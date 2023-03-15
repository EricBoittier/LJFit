from load_data import OO, OH, HH, OO_dim, OH_dim, HH_dim, dimer_flux, dimer_kern, dimer_pc, \
    dimer_mdcm, cluster_flux, cluster_kern, cluster_pc, cluster_mdcm
from opt import EvalLoss, get_LJs, get_DEs, fit_func, get_freeDEs
from plot import plot_energy_MSE, eval_2d_LJ, eval_2d_DE, eval_2d_freeDE
import matplotlib.pyplot as plt
import pandas as pd

#  convert molpro csv
converters = {i: lambda x: float(x.strip().replace("D", "E")) * 627.503 * 10 ** (-3) for i in range(1, 8)}
converters[0] = lambda x: float(x.strip().replace("D", "E"))
#  read molpro csv
sapt_df = pd.read_csv("/home/boittier/pcbach/molpro_sapt/sapt.csv", converters=converters)
sapt_df = sapt_df.rename(columns=lambda x: x.strip())

bounds_4 = [(0.0001, 3.9), (0.0001, 3.9), (0.00001, 2), (0.00001, 2)]
bounds_6 = [(0.0001, 3.9), (0.0001, 3.9), (0.00001, 2), (0.00001, 2), (1, 20), (1, 20)]


def fit_and_eval_6_12(df,
                      df_dimer,
                      x0=(3.5, 2.5, 0.1, 0.1, 1, 1, 1),
                      title=None,
                      elec="ele"):
    #  define the loss function
    loss = EvalLoss(df, {"OO": OO, "OH": OH, "HH": HH}, get_LJs, elec=elec)
    #  fit the parameters
    res = fit_func(loss.get_loss_6_12, x0, bounds=bounds_4)
    Osig, Hsig, Oep, Hep = res.x
    print(res)

    df["LJ"] = [get_LJs(i, Osig, Hsig, Oep, Hep, 6, 2, 2, OH, HH, OO) for i in df.index]
    df["ETOT_LJ"] = df["LJ"] + df[elec]

    plot_energy_MSE(df, "ETOT_LJ", "ETOT")
    if title is not None:
        plt.savefig("plots/12-6-" + title + "_MSE_cluster.pdf", bbox_inches="tight")
    plt.show()

    df_dimer["dist"] = sapt_df["DIST"]
    eval_2d_LJ(df_dimer, res, type="6-12", elec=elec)
    if title is not None:
        plt.savefig("plots/12-6-" + title + "_scan_pots.pdf", bbox_inches="tight")
    plt.show()


def fit_and_eval_freeDE(df, df_dimer, x0=(3.5, 2.5, 0.1, 0.1, 1, 1, 1), title=None, elec="ele"):
    #  define the loss function
    loss = EvalLoss(df, {"OO": OO, "OH": OH, "HH": HH}, get_freeDEs, elec=elec)
    #  fit the parameters
    res = fit_func(loss.get_loss_freeDE, x0, bounds=bounds_6)
    Osig, Hsig, Oep, Hep, a, b = res.x
    print(res)

    df["DE"] = [get_freeDEs(i, Osig, Hsig, Oep, Hep, a, b, OH, HH, OO) for i in df.index]
    df["ETOT_DE"] = df["DE"] + df[elec]

    plot_energy_MSE(df, "ETOT_DE", "ETOT")
    if title is not None:
        plt.savefig("plots/freeDE-" + title + "_MSE_cluster.pdf", bbox_inches="tight")
    # plt.show()

    df_dimer["dist"] = sapt_df["DIST"]
    eval_2d_freeDE(df_dimer, res, elec=elec)
    if title is not None:
        plt.savefig("plots/freeDE-" + title + "_scan_pots.pdf", bbox_inches="tight")
    # plt.show()


def fit_and_eval_7_14(df, df_dimer, x0=(3.5, 2.5, 0.1, 0.1, 1, 1, 1), title=None, elec="ele"):
    #  define the loss function
    loss = EvalLoss(df, {"OO": OO, "OH": OH, "HH": HH}, get_LJs, elec=elec)
    #  fit the parameters
    res = fit_func(loss.get_loss_7_14, x0, bounds=bounds_4)
    Osig, Hsig, Oep, Hep = res.x
    print(res)

    df["LJ"] = [get_LJs(i, Osig, Hsig, Oep, Hep, 7, 2, 2, OH, HH, OO)
                for i in df.index]
    df["ETOT_LJ"] = df["LJ"] + df[elec]

    plot_energy_MSE(df, "ETOT_LJ", "ETOT")
    if title is not None:
        plt.savefig("plots/14-7-" + title + "_MSE_cluster.pdf", bbox_inches="tight")
    # plt.show()

    df_dimer["dist"] = sapt_df["DIST"]
    eval_2d_LJ(df_dimer, res, type="7-14", elec=elec)
    if title is not None:
        plt.savefig("plots/14-7-" + title + "_scan_pots.pdf", bbox_inches="tight")
    # plt.show()


def fit_and_eval_DE(df, df_dimer, x0=(3.5, 2.5, 0.1, 0.1, 1, 1, 1), title=None, elec="ele"):
    #  define the loss function
    print(df.keys())
    df = df[3:].copy()
    # df_dimer = df_dimer[4:].copy()
    print(df)
    loss = EvalLoss(df, {"OO": OO, "OH": OH, "HH": HH}, get_DEs, elec=elec)
    #  fit the parameters
    res = fit_func(loss.get_loss, x0, bounds=bounds_4)
    Osig, Hsig, Oep, Hep = res.x
    print(res)

    df["LJ"] = [get_DEs(i, Osig, Hsig, Oep, Hep, OH, HH, OO)
                for i in df.index]
    df["ETOT_LJ"] = df["LJ"] + df[elec]

    plot_energy_MSE(df, "ETOT_LJ", "ETOT")
    if title is not None:
        plt.savefig("plots/DE-" + title + "_MSE_cluster.pdf", bbox_inches="tight")
    # plt.show()

    df_dimer["dist"] = sapt_df["DIST"]
    eval_2d_DE(df_dimer, res, elec=elec)
    if title is not None:
        plt.savefig("plots/DE-" + title + "_scan_pots.pdf", bbox_inches="tight")
    # plt.show()


def pc_6_12():
    x0 = [2.0, 0.1, 1.703e-01, 0.0001]
    # fit_and_eval_6_12(cluster_pc, dimer_pc, x0, title="pc")
    fit_and_eval_6_12(dimer_pc, dimer_pc, x0, title="pc")

# pc_6_12()

def pc_7_14():
    x0 = [2.0, 0.1, 1.703e-01, 0.0001]
    fit_and_eval_7_14(cluster_pc, dimer_pc, x0, title="pc")


def pc_DE():
    x0 = [2.0, 0.1, 1.703e-01, 0.0001]
    fit_and_eval_DE(cluster_pc, dimer_pc, x0, title="pc")


def mdcm_6_12():
    x0 = [2.0, 0.1, 1.703e-01, 0.0001]
    fit_and_eval_6_12(cluster_mdcm, dimer_mdcm, x0, title="mdcm")


def mdcm_7_14():
    x0 = [2.0, 0.1, 1.703e-01, 0.0001]
    fit_and_eval_7_14(cluster_mdcm, dimer_mdcm, x0, title="mdcm")


def mdcm_DE():
    x0 = [2.0, 2.1, 1.703e-01, 0.0001]
    # fit_and_eval_DE(cluster_mdcm, dimer_mdcm, x0, title="mdcm")
    fit_and_eval_DE(dimer_mdcm, dimer_mdcm, x0, title="mdcm")

mdcm_DE()

def mdcm_freeDE():
    x0 = [2.0, 2.0, 1.703e-01, 0.0001, 16., 4.,]
    fit_and_eval_freeDE(cluster_mdcm, dimer_mdcm, x0, title="mdcm")


def kern_6_12():
    x0 = [2.0, 0.1, 1.703e-01, 0.0001]
    fit_and_eval_6_12(cluster_kern, dimer_kern, x0, title="kern")

def kern_7_14():
    x0 = [2.0, 0.1, 1.703e-01, 0.0001]
    fit_and_eval_7_14(cluster_kern, dimer_kern, x0, title="kern")

def kern_DE():
    x0 = [2.0, 0.1, 1.703e-01, 0.0001]
    fit_and_eval_DE(cluster_kern, dimer_kern, x0, title="kern")

def kern_freeDE():
    x0 = [2.0, 2., 1.703e-01, 0.0001, 16., 4., ]
    fit_and_eval_freeDE(cluster_kern, dimer_kern, x0, title="kern")



def flux_6_12():
    x0 = [2.0, 0.1, 1.703e-01, 0.0001]
    fit_and_eval_6_12(cluster_flux, dimer_flux, x0, title="flux")

def flux_7_14():
    x0 = [2.0, 0.1, 1.703e-01, 0.0001]
    fit_and_eval_7_14(cluster_flux, dimer_flux, x0, title="flux")

def flux_DE():
    x0 = [2.0, 0.1, 1.703e-01, 0.0001]
    fit_and_eval_DE(cluster_flux, dimer_flux, x0, title="flux")

def flux_freeDE():
    x0 = [2.0, 2., 1.703e-01, 0.0001, 16., 4., ]
    fit_and_eval_freeDE(cluster_flux, dimer_flux, x0, title="flux")


def elst_6_12():
    x0 = [2.0, 0.1, 1.703e-01, 0.0001]
    fit_and_eval_6_12(cluster_kern, dimer_kern, x0, title="elst", elec="ELST")


def elst_7_14():
    x0 = [2.0, 0.1, 1.703e-01, 0.0001]
    fit_and_eval_7_14(cluster_kern, dimer_kern, x0, title="elst", elec="ELST")


def elst_DE():
    x0 = [2.0, 0.1, 1.703e-01, 0.0001]
    fit_and_eval_DE(cluster_kern, dimer_kern, x0, title="elst", elec="ELST")


def elst_freeDE():
    x0 = [2.0, 2., 1.703e-01, 0.0001, 16., 4., ]
    fit_and_eval_freeDE(cluster_kern, dimer_kern, x0, title="elst", elec="ELST")





# pc_DE()
# pc_6_12()
# pc_7_14()

# mdcm_DE()
# mdcm_6_12()
# mdcm_freeDE()

# kern_DE()
# kern_6_12()
# kern_7_14()
# kern_freeDE()

# elst_DE()
# elst_7_14()
# elst_6_12()
# elst_freeDE()

# flux_6_12()
# flux_7_14()
# flux_DE()
# flux_freeDE()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--charge_model")
    parser.add_argument("-p", "--pot_model")
    args = vars(parser.parse_args())

    if args["pot_model"] == "DE":
        if args["charge_model"] == "pc":
            pc_DE()
        elif args["charge_model"] == "mdcm":
            mdcm_DE()
        elif args["charge_model"] == "kern":
            kern_DE()

    elif args["pot_model"] == "6-12":
        if args["charge_model"] == "pc":
            pc_6_12()
        elif args["charge_model"] == "mdcm":
            mdcm_6_12()
        elif args["charge_model"] == "kern":
            kern_6_12()

    elif args["pot_model"] == "7-14":
        if args["charge_model"] == "pc":
            pc_7_14()
        elif args["charge_model"] == "mdcm":
            mdcm_7_14()
        elif args["charge_model"] == "kern":
            kern_7_14()

    #  something went wrong
    else:
        print("Please specify the charge model and potential model")
