import quantax as qtx


def hubbard_2d(Lx, Ly, U, t=1.0, filling="half", boundary=(1, 1)):
    Nsites = Lx * Ly

    if filling == "half":
        n_up = Nsites // 2
        n_dn = Nsites // 2
    elif isinstance(filling, tuple) and len(filling) == 2:
        n_up, n_dn = filling
    else:
        raise ValueError("filling should be 'half' or a tuple (n_up, n_dn).")

    lattice = qtx.sites.Grid(
        extent=[Lx, Ly],
        boundary=boundary,
        particle_type="spinful_fermion",
        Nparticles=(n_up, n_dn),
        double_occ=True,
    )

    H = qtx.operator.Hubbard(U=U, t=t)

    return H

if __name__ == "__main__":
    H = hubbard_2d(8, 10, U=4.0, t=1.0, filling="half", boundary=(1, 0))
    print(H)