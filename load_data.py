import pandas as pd
import numpy as np


#  energies
dimer_flux = pd.read_csv("data/dimer_flux.csv")
dimer_kern = pd.read_csv("data/dimer_kern.csv")
dimer_pc = pd.read_csv("data/dimer_pc.csv")
dimer_mdcm = pd.read_csv("data/mdcm_dimer.csv")
cluster_flux = pd.read_csv("data/clusters_flux.csv")
cluster_kern = pd.read_csv("data/clusters_kern.csv")
cluster_pc = pd.read_csv("data/clusters_pc.csv")
cluster_mdcm = pd.read_csv("data/clusters_mdcm.csv")

# clusters = {""}

#  distances
l = np.load("data/dists.npz")
OO = l["OO"]
OH = l["OH"]
HH = l["HH"]
OO_dim = l["OO_dim"]
OH_dim = l["OH_dim"]
HH_dim = l["HH_dim"]

