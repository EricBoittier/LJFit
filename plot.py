import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from opt import get_LJs, get_DEs, get_freeDEs
from load_data import OO, OH, HH, OO_dim, OH_dim, HH_dim, dimer_flux, dimer_kern, dimer_pc

# #  convert molpro csv
# converters = {i: lambda x: float(x.strip().replace("D", "E")) * 627.503 * 10 ** (-3) for i in range(1, 8)}
# converters[0] = lambda x: float(x.strip().replace("D", "E"))
# df = pd.read_csv("/home/boittier/pcbach/molpro_sapt/sapt.csv", converters=converters)




def plot_energy_MSE(df, key1, key2, FONTSIZE=14,
                    xlabel="NBOND energy\n(kcal/mol)",
                    ylabel="CCSD(T) interaction energy\n(kcal/mol)",
                    elec="ele",
                    CMAP="viridis",
                    cbar_label = "ELEC (kcal/mol)"):
    """Plot the energy MSE"""

    fig, ax = plt.subplots()
    # calculate MSE
    ERROR = df[key1] - df[key2]
    MSE = np.mean(ERROR ** 2)
    df["MSE"] = ERROR ** 2
    # add the MSE to the plot
    ax.text(0.00, 1.05, f"MSE = {MSE:.2f} kcal/mol\nRMSE = {np.sqrt(MSE):.2f} kcal/mol",
            transform=ax.transAxes, fontsize=FONTSIZE)
    # color points by MSE
    sc = ax.scatter(df[key1], df[key2],
                    c=df[elec], cmap=CMAP, alpha=0.5)

    #  make the aspect ratio square
    ax.set_aspect("equal")
    #  make the range of the plot the same
    ax.set_ylim(ax.get_xlim())
    # ax.set_xlim(-15,10)
    # ax.set_ylim(-15,10)
    #  plot the diagonal line
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    #  show the grid
    ax.grid(alpha=0.15)
    #  set the labels
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    plt.colorbar(sc, label=cbar_label)
    #  tight layout
    plt.tight_layout()
    # plt.show()


def eval_2d_LJ(df, res, type=None, elec="ele"):
    df = df.copy()
    Osig, Hsig, Oep, Hep = res.x

    if type is not None:
        if type == "6-12":
            a = 6
            b = 2
            c = 2
        elif type == "7-14":
            a = 7
            b = 2
            c = 2

    df["LJ"] = [get_LJs(i, Osig, Hsig, Oep, Hep,
                        a, b, c, OH_dim, HH_dim, OO_dim) for i in df.index]

    fig, axs = plt.subplots(3, sharex=True, figsize=(6, 7))
    fig.suptitle(
        f"O$_{{\sigma}}$ : {Osig:.1f} $\mathrm{{\AA}}$ | O$_{{\epsilon}}$ : {Oep:.4f} kcal/mol \nH$_{{\sigma}}$ : {Hsig:.1f} $\mathrm{{\AA}}$ | H$_{{\epsilon}}$ : {Hep:.4f} kcal/mol",
        fontsize=20, y=1.0)

    axs[2].set_xlim(1.25, 5)

    axs[2].plot(df["dist"], df["LJ"] + df[elec],
                alpha=0.7, label="LJ", c="firebrick")
    axs[2].plot(df["dist"], df["ETOT"], label="ETOT",
                alpha=1, c="firebrick", linestyle="--")
    axs[2].set_xlabel("$r ~~[\mathrm{\\AA}]$", fontsize=20)

    # axs[0].text(0.85,0.6, "LJ", fontsize=18, transform=axs[0].transAxes,)
    # axs[1].text(0.85,0.6, "Elec.", fontsize=18, transform=axs[1].transAxes,)
    # axs[2].text(0.85,0.6, "Total", fontsize=18, transform=axs[2].transAxes,)

    axs[2].axhline(0, color="k", linestyle="--", alpha=0.2)
    axs[1].set_ylabel("$E$ [kcal/mol]\n", fontsize=20)
    axs[2].set_ylim(-5, 10)

    axs[0].plot(df["dist"], df["LJ"], c="k")
    axs[0].plot(df["dist"], df["ETOT"] - df[elec], c="k", linestyle="-", alpha=0.25)
    axs[0].plot(df["dist"], df["ETOT"] - df["ELST"], c="k", linestyle="--", )
    axs[0].axhline(0, color="k", linestyle="--", alpha=0.2)
    axs[0].set_ylim(-1.5, 20)

    #  create an inset axes object that is 40% width and height of the
    #  parent axes
    axins = inset_axes(axs[0],
                       width="40%",  # width = 30% of parent_bbox
                       height="40%",  # height : 1 inch
                       loc='upper right',
                       )
    m1 = (df["ETOT"] - df[elec]).min()
    m2 = (df["ETOT"] - df["ELST"]).min()
    m3 = (df["LJ"]).min()
    axins.set_ylim(min([m1, m2, m3]), 0.1)

    axins.set_xlim(2, 4)
    axins.plot(df["dist"], df["LJ"], c="k")
    axins.plot(df["dist"], df["ETOT"] - df[elec], c="k", linestyle="-", alpha=0.25)
    axins.plot(df["dist"], df["ETOT"] - df["ELST"], c="k", linestyle="--", )
    axins.axhline(0, color="k", linestyle="--", alpha=0.2)

    axs[1].set_ylim(-10, 1)
    axs[1].plot(df["dist"], df[elec], c="orange")
    axs[1].plot(df["dist"], df["ELST"], c="orange", alpha=1, linestyle="--")
    axs[1].axhline(0, color="k", linestyle="--", alpha=0.2)

    plt.tight_layout()


def eval_2d_DE(df, res, type=None, elec="ele"):
    df = df.copy()
    Osig, Hsig, Oep, Hep = res.x

    df["LJ"] = [get_DEs(i, Osig, Hsig, Oep, Hep, OH_dim, HH_dim, OO_dim) for i in df.index]

    fig, axs = plt.subplots(3, sharex=True, figsize=(6, 7))
    fig.suptitle(
        f"O$_{{\sigma}}$ : {Osig:.1f} $\mathrm{{\AA}}$ | O$_{{\epsilon}}$ : {Oep:.4f} kcal/mol \nH$_{{\sigma}}$ : {Hsig:.1f} $\mathrm{{\AA}}$ | H$_{{\epsilon}}$ : {Hep:.4f} kcal/mol",
        fontsize=20, y=1.0)

    axs[2].set_xlim(1.25, 5)

    axs[2].plot(df["dist"], df["LJ"] + df[elec],
                alpha=0.7, label="LJ", c="firebrick")
    axs[2].plot(df["dist"], df["ETOT"], label="ETOT",
                alpha=1, c="firebrick", linestyle="--")
    axs[2].set_xlabel("$r ~~[\mathrm{\\AA}]$", fontsize=20)

    # axs[0].text(0.85,0.6, "LJ", fontsize=18, transform=axs[0].transAxes,)
    # axs[1].text(0.85,0.6, "Elec.", fontsize=18, transform=axs[1].transAxes,)
    # axs[2].text(0.85,0.6, "Total", fontsize=18, transform=axs[2].transAxes,)

    axs[2].axhline(0, color="k", linestyle="--", alpha=0.2)
    axs[1].set_ylabel("$E$ [kcal/mol]\n", fontsize=20)
    axs[2].set_ylim(-5, 10)

    axs[0].plot(df["dist"], df["LJ"], c="k")
    axs[0].plot(df["dist"], df["ETOT"] - df[elec], c="k", linestyle="-", alpha=0.25)
    axs[0].plot(df["dist"], df["ETOT"] - df["ELST"], c="k", linestyle="--", )
    axs[0].axhline(0, color="k", linestyle="--", alpha=0.2)
    # axs[0].set_ylim(-1.5,1.5)
    axs[0].set_ylim(-1.5, 20)

    #  create an inset axes object that is 40% width and height of the
    #  parent axes
    axins = inset_axes(axs[0],
                       width="50%",  # width = 30% of parent_bbox
                       height="50%",  # height : 1 inch
                       loc='upper right',
                       )
    m1 = (df["ETOT"] - df[elec]).min()
    m2 = (df["ETOT"] - df["ELST"]).min()
    m3 = (df["LJ"]).min()
    axins.set_ylim(min([m1, m2, m3]), 0.1)

    axins.set_xlim(2, 4)
    axins.plot(df["dist"], df["LJ"], c="k")
    axins.plot(df["dist"], df["ETOT"] - df[elec], c="k", linestyle="-", alpha=0.25)
    axins.plot(df["dist"], df["ETOT"] - df["ELST"], c="k", linestyle="--", )
    axins.axhline(0, color="k", linestyle="--", alpha=0.2)

    axs[1].set_ylim(-10, 1)
    axs[1].plot(df["dist"], df[elec], c="orange")
    axs[1].plot(df["dist"], df["ELST"], c="orange", alpha=1, linestyle="--")
    axs[1].axhline(0, color="k", linestyle="--", alpha=0.2)

    plt.tight_layout()


def eval_2d_freeDE(df, res, type=None, elec="ele"):
    df = df.copy()
    Osig, Hsig, Oep, Hep, a, b = res.x

    df["LJ"] = [get_freeDEs(i, Osig, Hsig, Oep, Hep, a, b, OH_dim, HH_dim, OO_dim) for i in df.index]

    fig, axs = plt.subplots(3, sharex=True, figsize=(6, 7))
    fig.suptitle(
        f"O$_{{\sigma}}$ : {Osig:.1f} $\mathrm{{\AA}}$ | O$_{{\epsilon}}$ : {Oep:.4f} kcal/mol \nH$_{{\sigma}}$ : {Hsig:.1f} $\mathrm{{\AA}}$ | H$_{{\epsilon}}$ : {Hep:.4f} kcal/mol",
        fontsize=20, y=1.0)

    axs[2].set_xlim(1.25, 5)

    axs[2].plot(df["dist"], df["LJ"] + df[elec],
                alpha=0.7, label="LJ", c="firebrick")
    axs[2].plot(df["dist"], df["ETOT"], label="ETOT",
                alpha=1, c="firebrick", linestyle="--")
    axs[2].set_xlabel("$r ~~[\mathrm{\\AA}]$", fontsize=20)

    # axs[0].text(0.85,0.6, "LJ", fontsize=18, transform=axs[0].transAxes,)
    # axs[1].text(0.85,0.6, "Elec.", fontsize=18, transform=axs[1].transAxes,)
    # axs[2].text(0.85,0.6, "Total", fontsize=18, transform=axs[2].transAxes,)

    axs[2].axhline(0, color="k", linestyle="--", alpha=0.2)
    axs[1].set_ylabel("$E$ [kcal/mol]\n", fontsize=20)
    axs[2].set_ylim(-5, 10)

    axs[0].plot(df["dist"], df["LJ"], c="k")
    axs[0].plot(df["dist"], df["ETOT"] - df[elec], c="k", linestyle="-", alpha=0.25)
    axs[0].plot(df["dist"], df["ETOT"] - df["ELST"], c="k", linestyle="--", )
    axs[0].axhline(0, color="k", linestyle="--", alpha=0.2)
    # axs[0].set_ylim(-1.5,1.5)
    axs[0].set_ylim(-1.5, 20)

    #  create an inset axes object that is 40% width and height of the
    #  parent axes
    axins = inset_axes(axs[0],
                       width="50%",  # width = 30% of parent_bbox
                       height="50%",  # height : 1 inch
                       loc='upper right',
                       )
    m1 = (df["ETOT"] - df[elec]).min()
    m2 = (df["ETOT"] - df["ELST"]).min()
    m3 = (df["LJ"]).min()
    axins.set_ylim(min([m1, m2, m3]), 0.1)

    axins.set_xlim(2, 4)
    axins.plot(df["dist"], df["LJ"], c="k")
    axins.plot(df["dist"], df["ETOT"] - df[elec], c="k", linestyle="-", alpha=0.25)
    axins.plot(df["dist"], df["ETOT"] - df["ELST"], c="k", linestyle="--", )
    axins.axhline(0, color="k", linestyle="--", alpha=0.2)

    axs[1].set_ylim(-10, 1)
    axs[1].plot(df["dist"], df[elec], c="orange")
    axs[1].plot(df["dist"], df["ELST"], c="orange", alpha=1, linestyle="--")
    axs[1].axhline(0, color="k", linestyle="--", alpha=0.2)

    plt.tight_layout()

