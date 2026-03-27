import os
import sys
import time
import csv
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("../")
root_dir = os.getcwd()
print(root_dir)
sys.path.insert(0, root_dir)

from src.model.NNBF import SlaterBackflowJastrow, SlaterBackflow, JastrowSlater, init_params, slater_forward_single, Slater
from src.make_hubbard_2d import hubbard_2d
import quantax as qtx
import jax.random as jr
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import equinox as eqx
from quantax.nn import fermion_idx
from quantax.utils import LogArray
from quantax.optimizer.solver import auto_shift_eig

def clip_tree_global_norm(tree, max_norm: float):
    leaves = [x for x in jax.tree_util.tree_leaves(tree) if isinstance(x, jax.Array)]
    if not leaves:
        return tree
    sq_norm = sum(jnp.sum(jnp.square(x)) for x in leaves)
    global_norm = jnp.sqrt(sq_norm + 1e-12)
    scale = jnp.minimum(1.0, max_norm / global_norm)
    return jax.tree_util.tree_map(
        lambda x: x * scale if isinstance(x, jax.Array) else x,
        tree,
    )

def transfer_params(pretrained_js: Slater, target_model: SlaterBackflowJastrow):
    """Transfer parameters from pretrained Slater to SlaterBackflowJastrow"""
    model = eqx.tree_at(lambda m: m.phi_up, target_model, pretrained_js.phi_up)
    model = eqx.tree_at(lambda m: m.phi_dn, model, pretrained_js.phi_dn)
    return model

def main():
    Lx = 8
    Ly = 8
    nsites = Lx * Ly
    nup = 28
    ndn = 28
    key = jr.PRNGKey(0)

    E_gs = -52.54

    H = hubbard_2d(Lx, Ly, U=8.0, t=1.0, filling=(nup, ndn), boundary=(1, 1))
    
    # ========== Stage 0: Pretrain JastrowSlater ==========
    slater_save_path = os.path.join(
        root_dir, f"pretrained_slater_{Lx}x{Ly}_{nup}up_{ndn}dn.eqx"
    )
    if os.path.exists(slater_save_path):
        print("=" * 60)
        print("Loading pretrained JastrowSlater parameters...")
        print("=" * 60)
        pretrain_model_template = Slater(nsites=nsites, nup=nup, ndn=ndn, key=key)
        pretrained_model = eqx.tree_deserialise_leaves(slater_save_path, pretrain_model_template)
        print(f"Pretrained model loaded from: {slater_save_path}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("Pretrained model not found, training from scratch...")
        print("=" * 60)
        pretrained_model = Slater(nsites=nsites, nup=nup, ndn=ndn, key=key)
    batch_size = 4096
    pretrain_state = qtx.state.Variational(pretrained_model, max_parallel=batch_size*45)
    pretrain_sampler = qtx.sampler.ParticleHop(pretrain_state, batch_size, sweep_steps=5*nsites)
    pretrain_optimizer = qtx.optimizer.AdamSR(pretrain_state, H, solver=auto_shift_eig(ashift=1e-3))
        
    n_iter_pretrain = 2000
    pretrain_lr = 5e-2
    pretrain_step_clip = 10.0
    e = float("inf")
    pretrain_energy = qtx.utils.DataTracer()
    pretrain_energy_history = []
    for i in range(n_iter_pretrain):
        iter_start = time.time()
            
        old_spins = pretrain_sampler._spins.copy()
        samples = pretrain_sampler.sweep()
        accept_rate = jnp.mean(jnp.any(old_spins != pretrain_sampler._spins, axis=1))
            
        lr = pretrain_lr * 1 / (1 + i / n_iter_pretrain)
        
        step = pretrain_optimizer.get_step(samples)
        step_has_nan = any(jnp.any(jnp.isnan(s)) for s in jax.tree_util.tree_leaves(step))
        step_has_inf = any(jnp.any(jnp.isinf(s)) for s in jax.tree_util.tree_leaves(step))
        if step_has_nan or step_has_inf:
            if i % 10 == 0:
                print(f"  Warning: Invalid step detected (NaN: {step_has_nan}, Inf: {step_has_inf}), skipping update")
            if len(pretrain_energy_history) > 0:
                e = pretrain_energy_history[-1]
            else:
                e = float("inf")
            iter_time = time.time() - iter_start
            if i % 10 == 0 or i < 10:
                print(f"  Pretrain {i:4d}   | {e:.8f} | Acc {float(accept_rate):.4f} | {iter_time:.3f}")
            continue

        step = clip_tree_global_norm(step, pretrain_step_clip)
        old_model = pretrain_state.model
        pretrain_state.update(step * lr)
        e = float(pretrain_optimizer.energy)
            
        if jnp.isnan(e) or jnp.isinf(e) or abs(e) > 1e6:
            pretrain_state._model = old_model
            iter_time = time.time() - iter_start
            if i % 10 == 0:
                print(f"  Warning: Invalid energy detected ({e})")
            if len(pretrain_energy_history) > 0:
                e = pretrain_energy_history[-1]
            else:
                e = float("inf")
            if i % 10 == 0 or i < 10:
                print(f"  Pretrain {i:4d}   | {e:.8f} | Acc {float(accept_rate):.4f} | {iter_time:.3f}")
            continue
            
        pretrain_energy.append(e)
        pretrain_energy_history.append(e)
            
        iter_time = time.time() - iter_start
        if i % 10 == 0 or i < 10:
            print(f"  Pretrain {i:4d}   | {e:.8f} | Acc {float(accept_rate):.4f} | {iter_time:.3f}")

    print(f"Pretraining completed. Final energy: {e:.8f}")
    print("=" * 60)
    
    # Save pretrained JastrowSlater parameters
    pretrained_slater = pretrain_state.model
    print("Saving pretrained JastrowSlater parameters...")
    slater_save_to_path = os.path.join(
        root_dir,
        f"pretrained_slater_{Lx}x{Ly}_{nup}up_{ndn}dn_{n_iter_pretrain}iter.eqx",
    )
    eqx.tree_serialise_leaves(slater_save_to_path, pretrained_slater)
    print(f"Pretrained Slater saved to: {slater_save_to_path}")
    print("=" * 60)

if __name__ == "__main__":
    print(jax.devices())
    if jax.devices()[0].platform == "gpu":
        print("Using GPU")
    else:
        print("Using CPU")
        sys.exit(1)
    main()