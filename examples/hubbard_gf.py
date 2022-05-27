# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
import lattpy as lp
import matplotlib.pyplot as plt
import exactdiag as ed


def main():
    num_sites = 5
    u, hop = 4.0, 1.0
    beta = 10.0

    latt = lp.finite_hypercubic(num_sites)
    model = ed.models.HubbardModel(latt, inter=u, hop=hop).hf()

    z = np.linspace(-10, +10, 1001) + 1e-1j
    gf = ed.gf_lehmann(model, z, beta, i=2, sigma=ed.UP)[0]

    fig, ax = plt.subplots()
    ax.plot(z.real, -gf.imag)
    ax.grid()
    ax.set_ylim(0, None)
    ax.set_xlim(z[0].real, z[-1].real)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$A(\omega)$")
    fig.set_size_inches((5, 3))
    fig.tight_layout()
    fig.savefig("hubbard_gf.png")
    plt.show()


if __name__ == "__main__":
    main()
