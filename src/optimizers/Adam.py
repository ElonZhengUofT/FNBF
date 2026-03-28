import jax
import jax.numpy as jnp
import jax.flatten_util as jfu
import jax.tree_util as jtu
import optax


class AdamQuantax:
    """
    Quantax-like pure Adam wrapper.

    Public interface:
        optimizer = AdamQuantax(state, H, ...)
        step = optimizer.get_step(samples)
        state.update(step * lr)

    Notes
    -----
    1. This class is written for the common real / holomorphic case.
       If your state is truly non-holomorphic complex (Quantax VS_TYPE.non_holomorphic),
       the gradient packing rule should be adapted.

    2. learning rate is intentionally kept OUTSIDE the optimizer, to mimic Quantax:
           state.update(step * lr)
       So internally we use Adam-style scaling + weight decay transforms,
       but do not include optax.scale_by_learning_rate(...).
    """

    def __init__(
        self,
        state,
        hamiltonian,
        *,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        mu_dtype=None,
        weight_decay: float = 1e-4,
        mask=None,
        nesterov: bool = False,
        grad_clip: float | None = None,
    ):
        self._state = state
        self._hamiltonian = hamiltonian

        self._energy = None
        self._VarE = None

        transforms = []
        if grad_clip is not None:
            transforms.append(optax.clip_by_global_norm(grad_clip))

        # Keep LR outside, Quantax-style.
        # Final effective update applied by the caller is:
        #   theta <- theta - lr * step
        # where step is Adam-preconditioned gradient + decayed weights.
        transforms.append(
            optax.scale_by_adam(
                b1=b1,
                b2=b2,
                eps=eps,
                eps_root=eps_root,
                mu_dtype=mu_dtype,
                nesterov=nesterov,
            )
        )
        transforms.append(optax.add_decayed_weights(weight_decay, mask=mask))

        self._tx = optax.chain(*transforms)

        params_tree, _ = self._state.partition()
        self._opt_state = self._tx.init(params_tree)

    @property
    def energy(self):
        return self._energy

    @property
    def VarE(self):
        return self._VarE

    def _get_reweight(self, samples, dtype):
        rw = getattr(samples, "reweight_factor", None)
        if rw is None:
            return jnp.ones((samples.nsamples,), dtype=dtype)
        return jnp.asarray(rw, dtype=dtype)

    def _compute_energy_stats(self, Eloc, rw):
        E = jnp.mean(rw * Eloc)
        VarE = jnp.mean(rw * jnp.abs(Eloc - E) ** 2).real
        self._energy = E.real
        self._VarE = VarE
        return E, VarE

    def _grad_flat(self, samples):
        """
        Pure VMC energy gradient in the common real / holomorphic case:

            g = 2 Re < (E_loc - E) * O^* >

        where
            O = (1 / psi) d psi / d theta

        Quantax provides:
            state.jacobian(samples.spins) -> O
            hamiltonian.Oloc(state, samples) -> E_loc
        """
        O = self._state.jacobian(samples.spins)                  # (Ns, Np)
        Eloc = self._hamiltonian.Oloc(self._state, samples)     # (Ns,)

        # Match dtypes cleanly
        O = jnp.asarray(O)
        Eloc = jnp.asarray(Eloc, dtype=O.dtype)

        rw = self._get_reweight(samples, Eloc.real.dtype)

        E, _ = self._compute_energy_stats(Eloc, rw)

        O_mean = jnp.mean(rw[:, None] * O, axis=0)              # (Np,)
        O_centered = O - O_mean[None, :]
        E_centered = Eloc - E

        grad = 2.0 * jnp.real(
            jnp.einsum("s,sp,s->p", rw, jnp.conj(O_centered), E_centered)
        )

        if not jnp.issubdtype(self._state.dtype, jnp.complexfloating):
            grad = grad.real.astype(self._state.dtype)

        return grad

    def get_step(self, samples):
        """
        Return a flat update direction compatible with:

            state.update(step * lr)

        Since Quantax's state.update(step) already performs theta <- theta - step,
        we should return a positive descent direction here, not a negated Optax update.
        """
        grad_flat = self._grad_flat(samples)

        grad_tree = self._state.get_params_unflatten(
            grad_flat.astype(self._state.dtype)
        )
        params_tree, _ = self._state.partition()

        updates_tree, self._opt_state = self._tx.update(
            grad_tree,
            self._opt_state,
            params_tree,
        )

        step_flat, _ = jfu.ravel_pytree(updates_tree)

        if not jnp.issubdtype(self._state.dtype, jnp.complexfloating):
            step_flat = step_flat.real.astype(self._state.dtype)

        return step_flat