import pandas as pd
import numpy as np
import os 
import logging
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use(['science','nature','no-latex'])
logging.getLogger('matplotlib.font_manager').disabled = True
from load_data import *

from opt import get_LJs, get_DEs, fit_func, get_freeDEs, get_freeLJs, get_DEplus
from plot import plot_energy_MSE, eval_2d_LJ, eval_2d_DE, eval_2d_freeDE

from opt import FF

bounds_4 = [(0.0001, 3.9), (0.0001, 3.9), (0.00001, 2), (0.00001, 2)]
bounds_6 = [(0.0001, 3.9), (0.0001, 3.9), (0.00001, 2), (0.00001, 2), (1, 20), (1, 20)]
bounds_7 = [(0.0001, 3.9), (0.0001, 3.9), (0.00001, 2), (0.00001, 2), (1, 20), (1, 20), (1, 20)]
bounds_8 = [(0.0001, 3.9), (0.0001, 3.9), (0.00001, 2), (0.00001, 2), (1, 20), (1, 20), (1, 20),(-2000, 2000)]

def plot_2d(ff, lim=True):
    ff.set_dists({"OO": OO_dim, "OH": OH_dim, "HH": HH_dim})

    dimer_ccsdt = pd.read_csv("dimers_ccsdt.csv")
    len(dimer_ccsdt)
    dimer_ccsdt["ETOT"] = dimer_ccsdt["int_CCSDT"]
    dimer_ccsdt["ele"] = dimer_ccsdt["int_CCSDT"]
    dimer_ccsdt["ELST"] = dimer_ccsdt["int_CCSDT"]


    ff.data = dimer_ccsdt
    _ = ff.get_best_df()
    dist=[0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 7, 8]

    plt.plot(dist, _["LJ"])
    if lim:
        plt.xlim(0.75,5)
        plt.ylim(-1,1)
    plt.axhline(0, c="k")
    
    
def get_energy(path):
    lines = open(path).readlines()
    try:
        ccsdt, mp2, hf = (float(x) for x in lines[-3].split())
        return ccsdt, mp2, hf
    except ValueError:
        return None

bach_base = Path("/home/boittier/pcbach")
cluster_path = bach_base / "molpro_energies"
singles_path = bach_base / "3body/f12"

#  loading data

cluster_files = [_ for _ in cluster_path.iterdir() if _.is_dir()]
singles_files = [_ for _ in singles_path.iterdir() if _.is_dir()]

cluster_dict = {_.name : [x for x in _.iterdir() if x.name.endswith("out") and x.name.startswith("t")] for _ in cluster_files}
singles_dict = {_.name : [x for x in _.iterdir() if x.name.endswith("out") and x.name.startswith("t")] for _ in singles_files}

df = pd.DataFrame({"cluster_path": cluster_dict, "singles_paths": singles_dict})

failed = list(df[df["singles_paths"].str.len() == 0].index)
success = df[df["singles_paths"].str.len() != 0]

df["cluster_E"] = df["cluster_path"].apply(lambda x: get_energy(x[0]))
df["c_HF"] = df["cluster_E"] .apply(lambda x: x[2] if x is not None else None)
df["c_MP2"] = df["cluster_E"] .apply(lambda x: x[1] if x is not None else None)
df["c_CCSDT"] = df["cluster_E"] .apply(lambda x: x[0] if x is not None else None)

df["singles_E"] = df["singles_paths"].apply(lambda x: [get_energy(_) for _ in x])
df["s_HF"] = df["singles_E"].apply(lambda x: np.sum([_[2] for _ in x]) if x is not None else None)
df["s_MP2"] = df["singles_E"].apply(lambda x: np.sum([_[1] for _ in x]) if x is not None else None)
df["s_CCSDT"] = df["singles_E"].apply(lambda x: np.sum([_[0] for _ in x]) if x is not None else None)

H2kcalmol = 627.503
df["int_HF"] = (df["c_HF"]  - df["s_HF"]) * H2kcalmol
df["int_MP2"] = (df["c_MP2"]  - df["s_MP2"]) * H2kcalmol
df["int_CCSDT"] = (df["c_CCSDT"]  - df["s_CCSDT"]) * H2kcalmol

clean_df = df.dropna()

elec="ele"

def prepare_FF_DE(df, elec="ele", data="int_CCSDT"):
    cluster = df.copy()
    cluster.index = cluster["key"]

    flux_clean = pd.merge(cluster, clean_df, left_index=True, right_index = True)

    test_dict = {a: b for a,b in zip(cluster["key"], OO)}
    OO_df = pd.DataFrame(test_dict).T
    OO_df = pd.merge(OO_df, clean_df, left_index=True, right_index = True)
    OO_np = OO_df[[i for i in range(190)]].to_numpy()

    test_dict = {a: b for a,b in zip(cluster["key"], OH)}
    OH_df = pd.DataFrame(test_dict).T
    OH_df = pd.merge(OH_df, clean_df, left_index=True, right_index = True)
    OH_np = OH_df[[i for i in range(190)]].to_numpy()

    test_dict = {a: b for a,b in zip(cluster["key"], HH)}
    HH_df = pd.DataFrame(test_dict).T
    HH_df = pd.merge(HH_df, clean_df, left_index=True, right_index = True)
    HH_np = HH_df[[i for i in range(190)]].to_numpy()

    df = flux_clean
    # df["ETOT"] = df["int_HF"]
    df["ETOT"] = df[data]

    ff = FF(df, {"OO": OO_np, "OH": OH_np, "HH": HH_np}, get_DEs, elec=elec)
    return ff

def prepare_FF_freeDE(df, elec="ele",data="int_CCSDT"):
    cluster = df.copy()
    cluster.index = cluster["key"]

    flux_clean = pd.merge(cluster, clean_df, left_index=True, right_index = True)

    test_dict = {a: b for a,b in zip(cluster["key"], OO)}
    OO_df = pd.DataFrame(test_dict).T
    OO_df = pd.merge(OO_df, clean_df, left_index=True, right_index = True)
    OO_np = OO_df[[i for i in range(190)]].to_numpy()

    test_dict = {a: b for a,b in zip(cluster["key"], OH)}
    OH_df = pd.DataFrame(test_dict).T
    OH_df = pd.merge(OH_df, clean_df, left_index=True, right_index = True)
    OH_np = OH_df[[i for i in range(190)]].to_numpy()

    test_dict = {a: b for a,b in zip(cluster["key"], HH)}
    HH_df = pd.DataFrame(test_dict).T
    HH_df = pd.merge(HH_df, clean_df, left_index=True, right_index = True)
    HH_np = HH_df[[i for i in range(190)]].to_numpy()

    df = flux_clean
    # df["ETOT"] = df["int_HF"]
    df["ETOT"] = df[data]

    ff = FF(df, {"OO": OO_np, "OH": OH_np, "HH": HH_np}, get_freeDEs, elec=elec)
    return ff

def prepare_FF_LJ(df, elec="ele",data="int_CCSDT"):
    cluster = df.copy()
    cluster.index = cluster["key"]

    flux_clean = pd.merge(cluster, clean_df, left_index=True, right_index = True)

    test_dict = {a: b for a,b in zip(cluster["key"], OO)}
    OO_df = pd.DataFrame(test_dict).T
    OO_df = pd.merge(OO_df, clean_df, left_index=True, right_index = True)
    OO_np = OO_df[[i for i in range(190)]].to_numpy()

    test_dict = {a: b for a,b in zip(cluster["key"], OH)}
    OH_df = pd.DataFrame(test_dict).T
    OH_df = pd.merge(OH_df, clean_df, left_index=True, right_index = True)
    OH_np = OH_df[[i for i in range(190)]].to_numpy()

    test_dict = {a: b for a,b in zip(cluster["key"], HH)}
    HH_df = pd.DataFrame(test_dict).T
    HH_df = pd.merge(HH_df, clean_df, left_index=True, right_index = True)
    HH_np = HH_df[[i for i in range(190)]].to_numpy()

    df = flux_clean
    # df["ETOT"] = df["int_HF"]
    df["ETOT"] = df[data]

    ff = FF(df, {"OO": OO_np, "OH": OH_np, "HH": HH_np}, get_LJs, elec=elec)
    return ff

def prepare_FF_freeLJ(df, elec="ele", data="int_CCSDT"):
    cluster = df.copy()
    cluster.index = cluster["key"]

    flux_clean = pd.merge(cluster, clean_df, left_index=True, right_index = True)

    test_dict = {a: b for a,b in zip(cluster["key"], OO)}
    OO_df = pd.DataFrame(test_dict).T
    OO_df = pd.merge(OO_df, clean_df, left_index=True, right_index = True)
    OO_np = OO_df[[i for i in range(190)]].to_numpy()

    test_dict = {a: b for a,b in zip(cluster["key"], OH)}
    OH_df = pd.DataFrame(test_dict).T
    OH_df = pd.merge(OH_df, clean_df, left_index=True, right_index = True)
    OH_np = OH_df[[i for i in range(190)]].to_numpy()

    test_dict = {a: b for a,b in zip(cluster["key"], HH)}
    HH_df = pd.DataFrame(test_dict).T
    HH_df = pd.merge(HH_df, clean_df, left_index=True, right_index = True)
    HH_np = HH_df[[i for i in range(190)]].to_numpy()

    df = flux_clean
    # df["ETOT"] = df["int_HF"]
    df["ETOT"] = df[data]

    ff = FF(df, {"OO": OO_np, "OH": OH_np, "HH": HH_np}, get_freeLJs, elec=elec)
    return ff

def prepare_FF_DEplus(df, elec="ele", data="int_CCSDT"):
    cluster = df.copy()
    cluster.index = cluster["key"]

    flux_clean = pd.merge(cluster, clean_df, left_index=True, right_index = True)

    test_dict = {a: b for a,b in zip(cluster["key"], OO)}
    OO_df = pd.DataFrame(test_dict).T
    OO_df = pd.merge(OO_df, clean_df, left_index=True, right_index = True)
    OO_np = OO_df[[i for i in range(190)]].to_numpy()

    test_dict = {a: b for a,b in zip(cluster["key"], OH)}
    OH_df = pd.DataFrame(test_dict).T
    OH_df = pd.merge(OH_df, clean_df, left_index=True, right_index = True)
    OH_np = OH_df[[i for i in range(190)]].to_numpy()

    test_dict = {a: b for a,b in zip(cluster["key"], HH)}
    HH_df = pd.DataFrame(test_dict).T
    HH_df = pd.merge(HH_df, clean_df, left_index=True, right_index = True)
    HH_np = HH_df[[i for i in range(190)]].to_numpy()

    df = flux_clean
    # df["ETOT"] = df["int_HF"]
    df["ETOT"] = df[data]

    ff = FF(df, {"OO": OO_np, "OH": OH_np, "HH": HH_np}, get_DEplus, elec=elec)
    return ff

ff = prepare_FF_DEplus(cluster_mdcm)

method = "Nelder-Mead"
# bounds_8 = [(0.0001, 3.9), (0.0001, 3.9), (0.00001, 2), (0.00001, 2), (1, 20), (1, 20), (1, 20),(-20000, 20000)]
bounds_8 = [(0,4),(0,4),(0,2),(0,2),(-30000,30000),(-30000,30000),(-30000,30000),(-30000,30000)] 
ff.fit_repeat(1, bounds=bounds_8, method=method, quiet=False)

print(ff.opt_parm)

plot_energy_MSE(ff.df, "ETOT", "ETOT_LJ", 
                xlabel="$E_{\mathrm{(int., CCSD(T))}}$ [kcal/mol]", 
                ylabel="$E_{\mathrm{(int., LJ)}}$ [kcal/mol]")

plot_energy_MSE(ff.df, "ELST", elec, 
                CMAP="plasma",
                xlabel="$E_{\mathrm{(elec., CCSD(T))}}$ [kcal/mol]", 
                ylabel="$E_{\mathrm{(elec., LJ)}}$ [kcal/mol]")


