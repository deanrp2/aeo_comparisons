import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from pathlib import Path
import xarray as xr


dimsets = ["low", "med", "high"]
esets = ["animal", "DE", "large"]
gpcsets = ["3", "10", "50"]

scores = []
firsts = []
for d in dimsets:
    dalist = []
    cfs = []
    for e in esets:
        for g in gpcsets:
            df = pd.read_csv(Path("comp_results_p1") / Path(f"e{e}_g{g}_d{d}.csv"),
                    index_col = 0)
            df2 = pd.read_csv(Path("comp_results_p2") / Path(f"e{e}_g{g}_d{d}.csv"),
                    index_col = 0)
            df[df2.columns] = df2
            cols = df.columns
            da = xr.DataArray(df.values, dims = ["trial", "f"],
                    coords = {"f" : cols, "trial": list(range(df.shape[0]))})
            dalist.append(da)
            cfs.append(e + g)
    master = xr.concat(dalist, pd.Index(cfs, name = "config"))
    s = utils.score(master)
    s2 = utils.firsts(master)
    scores.append(s)
    firsts.append(s2)

nicdims = ["Low Dimension", "Medium Dimension", "High Dimension"]
nicnames = ["Animal", "DE", "Large"]
nn = []
for e in nicnames:
    for g in gpcsets:
        nn.append(e + ", $N_g$: " + str(g))

score_master_table = pd.DataFrame(np.zeros((9, 3)),
        columns = nicdims,
        index = cfs)

for s, c, n in zip(scores, dimsets, nicdims):
    score_master_table.loc[:, n] = s
score_master_table.index = nn
#print(master_table.to_latex())


firsts_master_table = pd.DataFrame(np.zeros((9, 3)),
        columns = nicdims,
        index = cfs)

for s, c, n in zip(firsts, dimsets, nicdims):
    firsts_master_table.loc[:, n] = np.round(s)
firsts_master_table.index = nn

def stitch(df1, df2):
    a = df1.astype(str)
    b = " (" + df2.astype(int).astype(str) + "\%)"
    return a + b


c = stitch(score_master_table, firsts_master_table)
print(c.to_latex(escape=False, column_format = "cccc"))
print(c.to_csv())
