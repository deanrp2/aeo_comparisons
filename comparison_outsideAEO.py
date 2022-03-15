import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from pathlib import Path
import xarray as xr


def stitch(df1, df2):
    a = df1.astype(str)
    b = " (" + df2.astype(int).astype(str) + "\%)"
    return a + b

np.set_printoptions(floatmode = "fixed")

outside_algos = ["DE", "GWO", "JAYA", "MFO", "PSO", "SSA", "WOA"]
inside_ensembles = ["animal", "DE", "large"]

dimsets = ["low", "med", "high"]

scores = []
firsts = []
csc_scores = []
csc_firsts = []
cec_scores = []
cec_firsts = []
for d in dimsets:
    dalist = []
    csc_dalist = []
    cec_dalist = []

    cfs = []

    for e in inside_ensembles:
        df = pd.read_csv(Path("comp_results_p1") / Path(f"e{e}_g3_d{d}.csv"),
                index_col = 0)
        df2 = pd.read_csv(Path("comp_results_p2") / Path(f"e{e}_g3_d{d}.csv"),
                index_col = 0)
        df[df2.columns] = df2
        cols = df.columns
        da = xr.DataArray(df.values, dims = ["trial", "f"],
                coords = {"f" : cols, "trial": list(range(df.shape[0]))})
        dalist.append(da)
        csc_dalist.append(da[:, :22])
        cec_dalist.append(da[:, 22:])
        cfs.append("AEO:" + e)
    for o in outside_algos:
        df = pd.read_csv(Path("comp_results_p1") / Path(f"e{o}_d{d}.csv"),
                index_col = 0)
        df2 = pd.read_csv(Path("comp_results_p2") / Path(f"e{o}_d{d}.csv"),
                index_col = 0)
        df[df2.columns] = df2
        cols = df.columns
        da = xr.DataArray(df.values, dims = ["trial", "f"],
                coords = {"f" : cols, "trial": list(range(df.shape[0]))})
        dalist.append(da)
        csc_dalist.append(da[:, :22])
        cec_dalist.append(da[:, 22:])
        cfs.append(o)

    master = xr.concat(dalist, pd.Index(cfs, name = "config"))
    s = utils.score(master)
    s2 = utils.firsts(master)
    scores.append(s)
    firsts.append(s2)

    csc_master = xr.concat(csc_dalist, pd.Index(cfs, name = "config"))
    s = utils.score(csc_master)
    s2 = utils.firsts(csc_master)
    csc_scores.append(s)
    csc_firsts.append(s2)

    cec_master = xr.concat(cec_dalist, pd.Index(cfs, name = "config"))
    s = utils.score(cec_master)
    s2 = utils.firsts(cec_master)
    cec_scores.append(s)
    cec_firsts.append(s2)

nicdims = ["Low Dimension Set", "Medium Dimension Set", "High Dimension Set"]
nicnames = ["AEO: Animal", "AEO: DE", "AEO: Large"] + outside_algos

#total
score_master_table = pd.DataFrame(np.ones((len(nicnames), 3)),
        columns = nicdims,
        index = cfs)

for s, c, n in zip(scores, dimsets, nicdims):
    score_master_table.loc[:, n] = s

firsts_master_table = pd.DataFrame(np.ones((len(nicnames), 3)),
        columns = nicdims,
        index = cfs)

for s, c, n in zip(firsts, dimsets, nicdims):
    firsts_master_table.loc[:, n] = np.round(s)

total = stitch(score_master_table, firsts_master_table)

#classic
score_master_table = pd.DataFrame(np.ones((len(nicnames), 3)),
        columns = nicdims,
        index = cfs)

for s, c, n in zip(csc_scores, dimsets, nicdims):
    score_master_table.loc[:, n] = s

firsts_master_table = pd.DataFrame(np.ones((len(nicnames), 3)),
        columns = nicdims,
        index = cfs)

for s, c, n in zip(csc_firsts, dimsets, nicdims):
    firsts_master_table.loc[:, n] = np.round(s)

classic = stitch(score_master_table, firsts_master_table)

#cec
score_master_table = pd.DataFrame(np.ones((len(nicnames), 3)),
        columns = nicdims,
        index = cfs)

for s, c, n in zip(cec_scores, dimsets, nicdims):
    score_master_table.loc[:, n] = s

firsts_master_table = pd.DataFrame(np.ones((len(nicnames), 3)),
        columns = nicdims,
        index = cfs)

for s, c, n in zip(cec_firsts, dimsets, nicdims):
    firsts_master_table.loc[:, n] = np.round(s)

cec17 = stitch(score_master_table, firsts_master_table)
print("Complete Set")
print(total)
print("\n")
print("Classic Set")
print(classic)
print("\n")
print("Cec2017 Set")
print(cec17)
print("\n")
###
print("Complete Set")
print(total.to_csv())
print("\n")
print("Classic Set")
print(classic.to_csv())
print("\n")
print("Cec2017 Set")
print(cec17.to_csv())
print("\n")
