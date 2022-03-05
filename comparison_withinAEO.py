import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from pathlib import Path
import xarray as xr



results_dir = Path("comp_results_p1")

dimsets = ["low", "med", "high"]
esets = ["animal", "DE", "large"]
gpcsets = ["3", "10", "50"]

scores = []
for d in dimsets:
    dalist = []
    cfs = []
    for e in esets:
        for g in gpcsets:
            df = pd.read_csv(results_dir / Path(f"e{e}_g{g}_d{d}.csv"),
                    index_col = 0)
            cols = df.columns
            da = xr.DataArray(df.values, dims = ["trial", "f"],
                    coords = {"f" : cols, "trial": list(range(df.shape[0]))})
            dalist.append(da)
            cfs.append(e + g)
    master = xr.concat(dalist, pd.Index(cfs, name = "config"))
    s = utils.score(master)
    scores.append(s)

nicdims = ["Low Dimension", "Medium Dimension", "High Dimension"]
master_table = pd.DataFrame(np.zeros((9, 3)),
        columns = nicdims,
        index = cfs)

for s, c, n in zip(scores, dimsets, nicdims):
    master_table.loc[:, n] = s
print(master_table)
print(master_table.to_latex())
