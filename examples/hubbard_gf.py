# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
import matplotlib.pyplot as plt
import exactdiag as ed


def main():
    model = ed.models.HubbardModel.chain(num_sites=7, inter=4.0, beta=10.0).hf()

    z = np.linspace(-8, +8, 1001) + 1e-1j
    gf = ed.gf_lehmann(model, z, i=3, sigma=ed.UP)[0]

    fig, ax = plt.subplots()
    ax.plot(z.real, -gf.imag / np.pi)
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
