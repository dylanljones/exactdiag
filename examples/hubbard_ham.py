# coding: utf-8
#
# This code is part of exactdiag.
#
# Copyright (c) 2022, Dylan Jones

import lattpy as lp
import matplotlib.pyplot as plt
import exactdiag as ed


def main():
    num_sites = 2
    u = 2.0

    latt = lp.finite_hypercubic(num_sites)
    model = ed.models.HubbardModel(latt, inter=u)

    sector = model.basis.get_sector()
    ham = model.hamiltonian(sector=sector)
    ax = ed.matshow(ham, ticklabels=sector.state_labels(), values=True)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig("hubbard_ham.png")
    plt.show()


if __name__ == "__main__":
    main()
