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
from src.model.FNBF import FNBF
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
from src.optimizers.Adam import AdamQuantax

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

def transfer_params_fnbf(pretrained_js: FNBF, target_model: FNBF):
    """Transfer parameters from pretrained smaller FNBF to larger FNBF"""
    model = eqx.tree_at(lambda m: m.fno_up, model, pretrained_js.fno_up)
    model = eqx.tree_at(lambda m: m.fno_dn, model, pretrained_js.fno_dn)
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
        root_dir, f"checkpoints/pretrained_slater_{Lx}x{Ly}_{nup}up_{ndn}dn.eqx"
    )
    fnbf_save_path = os.path.join(
        root_dir, f"checkpoints/fnbf.eqx"
    )
    if not os.path.exists(fnbf_save_path) or not os.path.exists(slater_save_path):
        raise ValueError("Pretrained model not found")
    
    pretrained_fnbf = eqx.tree_deserialise_leaves(fnbf_save_path, FNBF(nsites=16, nup=7, ndn=7, Lx=4, Ly=4, key=key))
    pretrained_slater = eqx.tree_deserialise_leaves(slater_save_path, Slater(nsites=nsites, nup=nup, ndn=ndn, key=key))

    test_model = Slater(nsites=nsites, nup=nup, ndn=ndn, key=key)
    test_model = eqx.tree_at(lambda m: m.phi_up, test_model, pretrained_slater.phi_up)
    test_model = eqx.tree_at(lambda m: m.phi_dn, test_model, pretrained_slater.phi_dn)
    test_state = qtx.state.Variational(test_model, max_parallel=4096)
    test_sampler = qtx.sampler.ParticleHop(test_state, 4096, sweep_steps=10 * nsites)
    test_optimizer = qtx.optimizer.AdamSR(test_state, H)
    test_energy = qtx.utils.DataTracer()
    for i in range(1):
        samples = test_sampler.sweep()
        step = test_optimizer.get_step(samples)
        test_state.update(step * 0.05)
        e = test_optimizer.energy
        print(e)

    model = FNBF(nsites=nsites, nup=nup, ndn=ndn, Lx=Lx, Ly=Ly, key=key)
    model = eqx.tree_at(lambda m: m.phi_up, model, pretrained_slater.phi_up)
    model = eqx.tree_at(lambda m: m.phi_dn, model, pretrained_slater.phi_dn)
    model = eqx.tree_at(lambda m: m.fno_up, model, pretrained_fnbf.fno_up)
    model = eqx.tree_at(lambda m: m.fno_dn, model, pretrained_fnbf.fno_dn)
    
    state = qtx.state.Variational(model, max_parallel=4096)
    sampler = qtx.sampler.ParticleHop(state, 2048, sweep_steps=10 * nsites)
    optimizer = qtx.optimizer.AdamSR(state, H)
    energy = qtx.utils.DataTracer()
    energy_csv_path = os.path.join(
        root_dir, f"logs/transfer8x8_energy.csv"
    )
    os.makedirs(os.path.dirname(energy_csv_path), exist_ok=True)
    lr_init = 0.05
    train_start = time.time()
    with open(energy_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "energy", "VarE", "iter_time", "elapsed_s", "lr"])
        for i in range(1000):
            iter_start = time.time()
            lr = lr_init / (1 + i / 1e3)
            samples = sampler.sweep()
            step = optimizer.get_step(samples)
            state.update(step * lr)
            e = optimizer.energy
            var_e = getattr(optimizer, "VarE", getattr(optimizer, "variance", float("nan")))
            iter_time = time.time() - iter_start
            elapsed_s = time.time() - train_start
            print(e)
            energy.append(e)
            writer.writerow([i, float(e), float(var_e), iter_time, elapsed_s, lr])
    print(f"Energy log saved to: {energy_csv_path}")
    print(energy.mean())

if __name__ == "__main__":
    print(jax.devices())
    if jax.devices()[0].platform == "gpu":
        print("Using GPU")
    else:
        print("Using CPU")
        sys.exit(1)
    main()