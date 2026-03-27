import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import quantax as qtx
from quantax.nn import fermion_idx
from quantax.utils import LogArray
from quantax.model import MultiDet


def init_params(nsites: int, nup: int, ndn: int, key):
    k1, k2, k3 = jr.split(key, 3)

    phi_up = jr.normal(k1, (nsites, nup), dtype=jnp.float32)
    phi_dn = jr.normal(k2, (nsites, ndn), dtype=jnp.float32)

    nfmodes = 2 * nsites
    v = jr.normal(k3, (nfmodes, nfmodes), dtype=jnp.float32) / nfmodes
    return phi_up, phi_dn, v


def scale_mlp_params(mlp: eqx.nn.MLP, scale: float) -> eqx.nn.MLP:
    if scale == 1.0:
        return mlp
    return jax.tree_util.tree_map(
        lambda x: x * scale if isinstance(x, jax.Array) else x,
        mlp,
    )


def jastrow_log_factor(n: jax.Array, v: jax.Array) -> LogArray:
    """Stable Jastrow factor in log-amplitude form."""
    log_jastrow = 0.5 * (n @ v @ n)
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


class Slater(eqx.Module):
    phi_up: jax.Array
    phi_dn: jax.Array

    nsites: int = eqx.field(static=True)
    nup: int = eqx.field(static=True)
    ndn: int = eqx.field(static=True)
    nfmodes: int = eqx.field(static=True)

    def __init__(self, nsites: int, nup: int, ndn: int, key=None):
        if key is None:
            key = qtx.get_subkeys(1)[0]

        self.nsites = nsites
        self.nup = nup
        self.ndn = ndn
        self.nfmodes = 2 * nsites

        self.phi_up, self.phi_dn, v = init_params(nsites, nup, ndn, key)

    def single_forward(self, n: jax.Array):
        n = n.astype(self.phi_up.dtype)
        return slater_forward_single(self.phi_up, self.phi_dn, n, self.nsites, self.nup)

    def __call__(self, n: jax.Array):
        if n.ndim == 1:
            return self.single_forward(n)
        elif n.ndim == 2:
            return jax.vmap(self.single_forward)(n)
        else:
            raise ValueError(f"Expected n.ndim in {{1,2}}, got {n.ndim}")


class JastrowSlater(eqx.Module):
    phi_up: jax.Array
    phi_dn: jax.Array
    v: jax.Array

    nsites: int = eqx.field(static=True)
    nup: int = eqx.field(static=True)
    ndn: int = eqx.field(static=True)
    nfmodes: int = eqx.field(static=True)

    def __init__(self, nsites: int, nup: int, ndn: int, key=None):
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

class SlaterBackflowJastrow(eqx.Module):
    mlp_up: eqx.nn.MLP
    mlp_dn: eqx.nn.MLP
    phi_up: jax.Array
    phi_dn: jax.Array
    v: jax.Array

    nsites: int = eqx.field(static=True)
    nup: int = eqx.field(static=True)
    ndn: int = eqx.field(static=True)
    nfmodes: int = eqx.field(static=True)

    def __init__(self, nsites: int, nup: int, ndn: int, width: int, depth: int = 1, key=None, backflow_scale: float = 1e-3):
        if key is None:
            key = qtx.get_subkeys(1)[0]

        self.nsites = nsites
        self.nup = nup
        self.ndn = ndn
        self.nfmodes = 2 * nsites

        k_init, k_up, k_dn = jr.split(key, 3)
        self.phi_up, self.phi_dn, self.v = init_params(nsites, nup, ndn, k_init)

        self.mlp_up = eqx.nn.MLP(
            in_size=self.nfmodes,
            out_size=self.phi_up.size,
            width_size=width,
            depth=depth,
            use_final_bias=False,
            activation=jnp.tanh,
            key=k_up,
        )

        self.mlp_dn = eqx.nn.MLP(
            in_size=self.nfmodes,
            out_size=self.phi_dn.size,
            width_size=width,
            depth=depth,
            use_final_bias=False,
            activation=jnp.tanh,
            key=k_dn,
        )
        self.mlp_up = scale_mlp_params(self.mlp_up, backflow_scale)
        self.mlp_dn = scale_mlp_params(self.mlp_dn, backflow_scale)

    def single_forward(self, n: jax.Array):
        n = n.astype(self.phi_up.dtype)
        jastrow = jastrow_log_factor(n, self.v)

        phi_up = self.phi_up + self.mlp_up(n).reshape(self.phi_up.shape)
        phi_dn = self.phi_dn + self.mlp_dn(n).reshape(self.phi_dn.shape)
        slater = slater_forward_single(phi_up, phi_dn, n, self.nsites, self.nup)
        return jastrow * slater

    def __call__(self, n: jax.Array):
        # single sample: (nfmodes,)
        if n.ndim == 1:
            return self.single_forward(n)

        # batch: (B, nfmodes)
        elif n.ndim == 2:
            return jax.vmap(self.single_forward)(n)

        else:
            raise ValueError(f"Expected n.ndim in {{1,2}}, got {n.ndim}")

class SlaterBackflow(eqx.Module):
    mlp_up: eqx.nn.MLP
    mlp_dn: eqx.nn.MLP
    phi_up: jax.Array
    phi_dn: jax.Array

    nsites: int = eqx.field(static=True)
    nup: int = eqx.field(static=True)
    ndn: int = eqx.field(static=True)
    nfmodes: int = eqx.field(static=True)

    def __init__(self, nsites: int, nup: int, ndn: int, width: int = 256, depth: int = 1, key=None, backflow_scale: float = 1e-3):
        if key is None:
            key = qtx.get_subkeys(1)[0]

        self.nsites = nsites
        self.nup = nup
        self.ndn = ndn
        self.nfmodes = 2 * nsites

        k_init, k_up, k_dn = jr.split(key, 3)
        self.phi_up, self.phi_dn, _ = init_params(nsites, nup, ndn, k_init)

        self.mlp_up = eqx.nn.MLP(
            in_size=self.nfmodes,
            out_size=self.phi_up.size,
            width_size=width,
            depth=depth,
            use_final_bias=False,
            activation=jnp.tanh,
            key=k_up,
        )

        self.mlp_dn = eqx.nn.MLP(
            in_size=self.nfmodes,
            out_size=self.phi_dn.size,
            width_size=width,
            depth=depth,
            use_final_bias=False,
            activation=jnp.tanh,
            key=k_dn,
        )
        self.mlp_up = scale_mlp_params(self.mlp_up, backflow_scale)
        self.mlp_dn = scale_mlp_params(self.mlp_dn, backflow_scale)

    def single_forward(self, n: jax.Array):
        n = n.astype(self.phi_up.dtype)

        phi_up = self.phi_up + self.mlp_up(n).reshape(self.phi_up.shape)
        phi_dn = self.phi_dn + self.mlp_dn(n).reshape(self.phi_dn.shape)

        return slater_forward_single(phi_up, phi_dn, n, self.nsites, self.nup)

    def __call__(self, n: jax.Array):
        if n.ndim == 1:
            return self.single_forward(n)
        elif n.ndim == 2:
            return jax.vmap(self.single_forward)(n)
        else:
            raise ValueError(f"Expected n.ndim in {{1,2}}, got {n.ndim}")


class MultiSlaterBackflow(eqx.Module):

    mlp_up: eqx.nn.MLP
    mlp_dn: eqx.nn.MLP
    phi_up: jax.Array
    phi_dn: jax.Array
    v: jax.Array

    nsites: int = eqx.field(static=True)
    nup: int = eqx.field(static=True)
    ndn: int = eqx.field(static=True)
    nfmodes: int = eqx.field(static=True)

    def __init__(self, nsites: int, nup: int, ndn: int, width: int, depth: int = 1, key=None, backflow_scale: float = 1e-3):
        if key is None:
            key = qtx.get_subkeys(1)[0]

        self.nsites = nsites
        self.nup = nup
        self.ndn = ndn
        self.nfmodes = 2 * nsites

        k_init, k_up, k_dn = jr.split(key, 3)
        self.phi_up, self.phi_dn, self.v = init_params(nsites, nup, ndn, k_init)

        self.mlp_up = eqx.nn.MLP(
            in_size=self.nfmodes,
            out_size=self.phi_up.size,
            width_size=width,
            depth=depth,
            use_final_bias=False,
            activation=jnp.tanh,
            key=k_up,
        )

        self.mlp_dn = eqx.nn.MLP(
            in_size=self.nfmodes,
            out_size=self.phi_dn.size,
            width_size=width,
            depth=depth,
            use_final_bias=False,
            activation=jnp.tanh,
            key=k_dn,
        )
        self.mlp_up = scale_mlp_params(self.mlp_up, backflow_scale)
        self.mlp_dn = scale_mlp_params(self.mlp_dn, backflow_scale)

    def single_forward(self, n: jax.Array):
        n = n.astype(self.phi_up.dtype)
        jastrow = jastrow_log_factor(n, self.v)

        phi_up = self.phi_up + self.mlp_up(n).reshape(self.phi_up.shape)
        phi_dn = self.phi_dn + self.mlp_dn(n).reshape(self.phi_dn.shape)
        slater = slater_forward_single(phi_up, phi_dn, n, self.nsites, self.nup)
        return jastrow * slater

    def __call__(self, n: jax.Array):
        # single sample: (nfmodes,)
        if n.ndim == 1:
            return self.single_forward(n)

        # batch: (B, nfmodes)
        elif n.ndim == 2:
            return jax.vmap(self.single_forward)(n)

        else:
            raise ValueError(f"Expected n.ndim in {{1,2}}, got {n.ndim}")


def init_generalized_params(nsites: int, nup: int, ndn: int, key):
    """Initialize full generalized Slater matrix and Jastrow factor."""
    nfmodes = 2 * nsites
    nelec = nup + ndn
    k1, k2 = jr.split(key, 2)
    phi = jr.normal(k1, (nfmodes, nelec), dtype=jnp.float32)
    v = jr.normal(k2, (nfmodes, nfmodes), dtype=jnp.float32) / nfmodes
    return phi, v


def generalized_slater_forward_single(phi, n):
    """Single-sample generalized Slater determinant evaluation."""
    idx = fermion_idx(n)
    M = phi[idx]  # (nelec, nelec)
    sign, logabs = jnp.linalg.slogdet(M)
    return LogArray(sign, logabs)


class GeneralizedSlater(eqx.Module):
    phi: jax.Array

    nsites: int = eqx.field(static=True)
    nup: int = eqx.field(static=True)
    ndn: int = eqx.field(static=True)
    nfmodes: int = eqx.field(static=True)
    nelec: int = eqx.field(static=True)

    def __init__(self, nsites: int, nup: int, ndn: int, key=None):
        if key is None:
            key = qtx.get_subkeys(1)[0]

        self.nsites = nsites
        self.nup = nup
        self.ndn = ndn
        self.nfmodes = 2 * nsites
        self.nelec = nup + ndn
        self.phi, _ = init_generalized_params(nsites, nup, ndn, key)

    def single_forward(self, n: jax.Array):
        n = n.astype(self.phi.dtype)
        slater = generalized_slater_forward_single(self.phi, n)
        return slater

    def __call__(self, n: jax.Array):
        if n.ndim == 1:
            return self.single_forward(n)
        elif n.ndim == 2:
            return jax.vmap(self.single_forward)(n)
        else:
            raise ValueError(f"Expected n.ndim in {{1,2}}, got {n.ndim}")

class GeneralizedJastrowSlater(eqx.Module):
    phi: jax.Array
    v: jax.Array

    nsites: int = eqx.field(static=True)
    nup: int = eqx.field(static=True)
    ndn: int = eqx.field(static=True)
    nfmodes: int = eqx.field(static=True)
    nelec: int = eqx.field(static=True)

    def __init__(self, nsites: int, nup: int, ndn: int, key=None):
        if key is None:
            key = qtx.get_subkeys(1)[0]

        self.nsites = nsites
        self.nup = nup
        self.ndn = ndn
        self.nfmodes = 2 * nsites
        self.nelec = nup + ndn

        self.phi, self.v = init_generalized_params(nsites, nup, ndn, key)

    def single_forward(self, n: jax.Array):
        n = n.astype(self.phi.dtype)
        jastrow = jastrow_log_factor(n, self.v)
        slater = generalized_slater_forward_single(self.phi, n)
        return jastrow * slater

    def __call__(self, n: jax.Array):
        if n.ndim == 1:
            return self.single_forward(n)
        elif n.ndim == 2:
            return jax.vmap(self.single_forward)(n)
        else:
            raise ValueError(f"Expected n.ndim in {{1,2}}, got {n.ndim}")


class GeneralizedSlaterBackflowJastrow(eqx.Module):
    mlp: eqx.nn.MLP
    phi: jax.Array
    v: jax.Array

    nsites: int = eqx.field(static=True)
    nup: int = eqx.field(static=True)
    ndn: int = eqx.field(static=True)
    nfmodes: int = eqx.field(static=True)
    nelec: int = eqx.field(static=True)

    def __init__(
        self,
        nsites: int,
        nup: int,
        ndn: int,
        width: int,
        depth: int = 1,
        key=None,
        backflow_scale: float = 1e-3,
    ):
        if key is None:
            key = qtx.get_subkeys(1)[0]

        self.nsites = nsites
        self.nup = nup
        self.ndn = ndn
        self.nfmodes = 2 * nsites
        self.nelec = nup + ndn

        k_init, k_mlp = jr.split(key, 2)
        self.phi, self.v = init_generalized_params(nsites, nup, ndn, k_init)

        self.mlp = eqx.nn.MLP(
            in_size=self.nfmodes,
            out_size=self.phi.size,
            width_size=width,
            depth=depth,
            use_final_bias=False,
            activation=jnp.tanh,
            key=k_mlp,
        )
        self.mlp = scale_mlp_params(self.mlp, backflow_scale)

    def single_forward(self, n: jax.Array):
        n = n.astype(self.phi.dtype)
        jastrow = jastrow_log_factor(n, self.v)
        phi = self.phi + self.mlp(n).reshape(self.phi.shape)
        slater = generalized_slater_forward_single(phi, n)
        return jastrow * slater

    def __call__(self, n: jax.Array):
        if n.ndim == 1:
            return self.single_forward(n)
        elif n.ndim == 2:
            return jax.vmap(self.single_forward)(n)
        else:
            raise ValueError(f"Expected n.ndim in {{1,2}}, got {n.ndim}")


class GeneralizedSlaterBackflow(eqx.Module):
    mlp: eqx.nn.MLP
    phi: jax.Array

    nsites: int = eqx.field(static=True)
    nup: int = eqx.field(static=True)
    ndn: int = eqx.field(static=True)
    nfmodes: int = eqx.field(static=True)
    nelec: int = eqx.field(static=True)

    def __init__(
        self,
        nsites: int,
        nup: int,
        ndn: int,
        width: int = 256,
        depth: int = 1,
        key=None,
        backflow_scale: float = 1e-3,
    ):
        if key is None:
            key = qtx.get_subkeys(1)[0]

        self.nsites = nsites
        self.nup = nup
        self.ndn = ndn
        self.nfmodes = 2 * nsites
        self.nelec = nup + ndn

        k_init, k_mlp = jr.split(key, 2)
        self.phi, _ = init_generalized_params(nsites, nup, ndn, k_init)

        self.mlp = eqx.nn.MLP(
            in_size=self.nfmodes,
            out_size=self.phi.size,
            width_size=width,
            depth=depth,
            use_final_bias=False,
            activation=jnp.tanh,
            key=k_mlp,
        )
        self.mlp = scale_mlp_params(self.mlp, backflow_scale)

    def single_forward(self, n: jax.Array):
        n = n.astype(self.phi.dtype)
        phi = self.phi + self.mlp(n).reshape(self.phi.shape)
        return generalized_slater_forward_single(phi, n)

    def __call__(self, n: jax.Array):
        if n.ndim == 1:
            return self.single_forward(n)
        elif n.ndim == 2:
            return jax.vmap(self.single_forward)(n)
        else:
            raise ValueError(f"Expected n.ndim in {{1,2}}, got {n.ndim}")


if __name__ == "__main__":
    nsites = 16
    nup = 4
    ndn = 4
    key = jr.PRNGKey(0)

    model = SlaterBackflowJastrow(
        nsites=nsites,
        nup=nup,
        ndn=ndn,
        width=64,
        depth=2,
        key=key,
    )
    print(model)