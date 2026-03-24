from quantax.model import MultiDet,UnrestrictedDet
import quantax as qtx
from quantax.nn import fermion_idx
import pdequinox as pdeqx
from pdequinox.arch import ClassicFNO
import equinox as eqx
from quantax.utils import LogArray
import jax
import jax.numpy as jnp
import jax.random as jr


def init_params(nsites: int, nup: int, ndn: int, key):
    k1, k2, k3 = jr.split(key, 3)

    phi_up = jr.normal(k1, (nsites, nup), dtype=jnp.float32)
    phi_dn = jr.normal(k2, (nsites, ndn), dtype=jnp.float32)

    nfmodes = 2 * nsites
    v = jr.normal(k3, (nfmodes, nfmodes), dtype=jnp.float32) / nfmodes
    return phi_up, phi_dn, v


def jastrow_log_factor(n: jax.Array, v: jax.Array) -> LogArray:
    """Stable Jastrow factor in log-amplitude form."""
    log_jastrow = - 0.5 * (n @ v @ n)
    return LogArray(jnp.ones_like(log_jastrow), log_jastrow)


def slater_forward_single(phi_up, phi_dn, n, nsites: int, nup: int):
    """Single-sample Slater determinant evaluation."""
    idx = fermion_idx(n)
    idx_up = idx[:nup]
    idx_dn = idx[nup:] - nsites

    M_up = phi_up[idx_up]         # (nup, nup)
    M_dn = phi_dn[idx_dn]         # (ndn, ndn)

    sign_up, logabs_up = jnp.linalg.slogdet(M_up)
    sign_dn, logabs_dn = jnp.linalg.slogdet(M_dn)

    psi_up = LogArray(sign_up, logabs_up)
    psi_dn = LogArray(sign_dn, logabs_dn)
    return psi_up * psi_dn


class JastrowSlater(eqx.Module):
    phi_up: jax.Array
    phi_dn: jax.Array
    v: jax.Array

    nsites: int = eqx.field(static=True)
    nup: int = eqx.field(static=True)
    ndn: int = eqx.field(static=True)
    nfmodes: int = eqx.field(static=True)

    def __init__(
        self,
        nsites: int,
        nup: int,
        ndn: int,
        key=None,
        Lx: int | None = None,
        Ly: int | None = None,
    ):
        if key is None:
            key = qtx.get_subkeys(1)[0]

        self.nsites = nsites
        self.nup = nup
        self.ndn = ndn
        self.nfmodes = 2 * nsites

        self.phi_up, self.phi_dn, self.v = init_params(nsites, nup, ndn, key)

    def single_forward(self, n: jax.Array):
        n = n.astype(self.phi_up.dtype)
        jastrow = jastrow_log_factor(n, self.v)
        slater = slater_forward_single(self.phi_up, self.phi_dn, n, self.nsites, self.nup)
        return jastrow * slater
    
    def __call__(self, n: jax.Array):
        if n.ndim == 1:
            return self.single_forward(n)
        elif n.ndim == 2:
            return jax.vmap(self.single_forward)(n)
        else:
            raise ValueError(f"Expected n.ndim in {{1,2}}, got {n.ndim}")


class FNBF(eqx.Module):
    Lx: int
    Ly: int
    phi_up: jax.Array
    phi_dn: jax.Array
    v: jax.Array
    
    fno_up: ClassicFNO
    fno_dn: ClassicFNO

    nsites: int = eqx.field(static=True)
    nup: int = eqx.field(static=True)
    ndn: int = eqx.field(static=True)
    nfmodes: int = eqx.field(static=True)

    def __init__(
        self,
        nsites: int,
        nup: int,
        ndn: int,
        Lx: int,
        Ly: int,
        key=None,
    ):
        if key is None:
            key = qtx.get_subkeys(1)[0]

        self.nsites = nsites
        self.nup = nup
        self.ndn = ndn
        self.nfmodes = 2 * nsites
        if Lx * Ly != nsites:
            raise ValueError(
                f"Expected Lx*Ly == nsites, got Lx={Lx}, Ly={Ly}, nsites={nsites}"
            )
        self.Lx = Lx
        self.Ly = Ly

        k_init, k_up, k_dn = jr.split(key, 3)
        self.phi_up, self.phi_dn, self.v = init_params(nsites, nup, ndn, k_init)

        self.fno_up = ClassicFNO(
            num_spatial_dims=2,
            in_channels=3,
            out_channels=1,
            key=k_up,
        )
        self.fno_dn = ClassicFNO(
            num_spatial_dims=2,
            in_channels=3,
            out_channels=1,
            key=k_dn,
        )

    def single_forward_from_phi(
        self, phi_up: jax.Array, phi_dn: jax.Array, n: jax.Array
    ):
        n = n.astype(self.phi_up.dtype)
        jastrow = jastrow_log_factor(n, self.v)
        slater = slater_forward_single(phi_up, phi_dn, n, self.nsites, self.nup)
        return jastrow * slater

    def _forward_single(self, n: jax.Array):
        # n: (2*nsites,)
        n = n.astype(self.phi_up.dtype)

        if n.shape[0] != 2 * self.nsites:
            raise ValueError(
                f"Expected input shape ({2*self.nsites},), got {n.shape}"
            )

        # split occupation
        n_up = n[:self.nsites]   # (nsites,)
        n_dn = n[self.nsites:]   # (nsites,)

        # reshape to grids
        n_up_grid = n_up.reshape(self.Lx, self.Ly)   # (Lx, Ly)
        n_dn_grid = n_dn.reshape(self.Lx, self.Ly)   # (Lx, Ly)

        # base orbitals -> grid fields
        # self.phi_up: (nsites, nup) -> (nup, Lx, Ly)
        phi_up_grid = self.phi_up.T.reshape(self.nup, self.Lx, self.Ly)
        phi_dn_grid = self.phi_dn.T.reshape(self.ndn, self.Lx, self.Ly)

        # each orbital is corrected independently with shared FNO parameters
        def correct_one_up(phi_i):
            # phi_i: (Lx, Ly)
            X_i = jnp.stack(
                [
                    n_up_grid,   # (Lx, Ly)
                    n_dn_grid,   # (Lx, Ly)
                    phi_i,       # (Lx, Ly)
                ],
                axis=0,
            )  # (3, Lx, Ly)

            delta_i = self.fno_up(X_i)   # expected (1, Lx, Ly)
            if delta_i.shape != (1, self.Lx, self.Ly):
                raise ValueError(
                    f"fno_up output shape mismatch: expected {(1, self.Lx, self.Ly)}, "
                    f"got {delta_i.shape}"
                )
            return delta_i[0]   # (Lx, Ly)

        def correct_one_dn(phi_i):
            # phi_i: (Lx, Ly)
            X_i = jnp.stack(
                [
                    n_up_grid,   # (Lx, Ly)
                    n_dn_grid,   # (Lx, Ly)
                    phi_i,       # (Lx, Ly)
                ],
                axis=0,
            )  # (3, Lx, Ly)

            delta_i = self.fno_dn(X_i)   # expected (1, Lx, Ly)
            if delta_i.shape != (1, self.Lx, self.Ly):
                raise ValueError(
                    f"fno_dn output shape mismatch: expected {(1, self.Lx, self.Ly)}, "
                    f"got {delta_i.shape}"
                )
            return delta_i[0]   # (Lx, Ly)

        phi_up_correction_grid = jax.vmap(correct_one_up)(phi_up_grid)   # (nup, Lx, Ly)
        phi_dn_correction_grid = jax.vmap(correct_one_dn)(phi_dn_grid)   # (ndn, Lx, Ly)
        return self.single_forward_from_phi(phi_up_correction_grid, phi_dn_correction_grid, n)

        # back to (nsites, nup)/(nsites, ndn)
        phi_up_correction = phi_up_correction_grid.reshape(self.nup, self.nsites).T
        phi_dn_correction = phi_dn_correction_grid.reshape(self.ndn, self.nsites).T

        phi_up_eff = self.phi_up + phi_up_correction
        phi_dn_eff = self.phi_dn + phi_dn_correction

        return self.single_forward_from_phi(phi_up_eff, phi_dn_eff, n)

    def __call__(self, n: jax.Array):
        # support single sample (2*nsites,) or batch (B, 2*nsites)
        if n.ndim == 1:
            return self._forward_single(n)
        elif n.ndim == 2:
            return jax.vmap(self._forward_single)(n)
        else:
            raise ValueError(f"Expected n.ndim in {{1,2}}, got {n.ndim}")

if __name__ == "__main__":
    FNO = ClassicFNO(
        num_spatial_dims=2,
        in_channels=2,
        out_channels=1,
        key=jax.random.PRNGKey(0),
    )
    lattice = qtx.sites.Square(
        4, particle_type=qtx.PARTICLE_TYPE.spinful_fermion, Nparticles=(7, 7)
    )
    N = lattice.Nsites

    H = qtx.operator.Hubbard(U=8)
    JastrowSlater = JastrowSlater(nsites=N, nup=7, ndn=7, key=jax.random.PRNGKey(0))
    state = qtx.state.Variational(JastrowSlater, max_parallel=8192*45)
    sampler = qtx.sampler.ParticleHop(state, 8192, sweep_steps=10 * N)
    optimizer = qtx.optimizer.SR(state, H)

    energy = qtx.utils.DataTracer()

    for i in range(500):
        samples = sampler.sweep()
        step = optimizer.get_step(samples)
        state.update(step * 0.05)
        e = optimizer.energy
        print(e)
        energy.append(e)
    print(energy.mean)

    model_fnbf = FNBF(nsites=N, nup=7, ndn=7, Lx=4, Ly=4, key=jax.random.PRNGKey(0))
    model_0 = state.model
    model_fnbf = eqx.tree_at(lambda model: model.phi_up, model_fnbf, model_0.phi_up)
    model_fnbf = eqx.tree_at(lambda model: model.phi_dn, model_fnbf, model_0.phi_dn)
    model_fnbf = eqx.tree_at(lambda model: model.v, model_fnbf, model_0.v)
    state_fnbf = qtx.state.Variational(model_fnbf, max_parallel=8192*45)
    sampler_fnbf = qtx.sampler.ParticleHop(state_fnbf, 8192, sweep_steps=10 * N)
    optimizer_fnbf = qtx.optimizer.SR(state_fnbf, H)

    energy_fnbf = qtx.utils.DataTracer()

    for i in range(500):
        samples = sampler_fnbf.sweep()
        step = optimizer_fnbf.get_step(samples)
        state_fnbf.update(step * 0.05)
        e = optimizer_fnbf.energy
        print(e)
        energy_fnbf.append(e)
    print(energy_fnbf.mean)