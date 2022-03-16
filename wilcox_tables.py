import numpy as np
import xarray as xr
from scipy.stats import wilcoxon

short = True

#low p value means different

ds = xr.load_dataset("benchmark_results.nc")

compare = "animal,gpc3"

if short:
    dedrops = ["DE,gpc"+str(a) for a in [3,10,50]]
    ds = ds.drop(dedrops)
    gdrops = [10, 50]
    drps = []
    for g in gdrops:
        drps.append("animal,gpc" + str(g))
        drps.append("large,gpc" + str(g))
    ds = ds.drop(drps)

c = ds[compare]
ds = ds.drop(compare)

#making table
nrows = len(ds.coords["f"]) + 1
ncols = len(ds.keys()) + 1
tabl = np.zeros((nrows, ncols), dtype = str)
tabl = tabl.tolist()

#fill in row names
for i in range(1, nrows):
    tabl[i][0] = ds.coords["f"].data[i-1]

#fill in column names
for i in range(1, ncols):
    tabl[0][i] = list(ds.keys())[i-1]

#fill in contents of table
for d in ["low", "med", "high"]:
    for i, f in enumerate(ds.coords["f"]):
        x = c.sel(f = f, dim = d).data
        for j, a in enumerate(ds.keys()):
            y = ds[a].sel(f = f, dim = d).data
            if np.all(x == x[0]) and np.all(y == y[0]):
                p = "0"
            else:
                r = wilcoxon(x, y).pvalue
                if r > 0.05:
                    p = "0"
                elif np.mean(x) > np.mean(y):
                    p = "-"
                elif np.mean(y) > np.mean(x):
                    p = "+"
                else:
                    raise Exception("weird p value")
            tabl[i + 1][j + 1] += p

#print table nicely
max_len = np.zeros(ncols, dtype = int)
for i in range(len(tabl)):
    for j in range(len(tabl[0])):
        max_len[j] = max(max_len[j], len(tabl[i][j]))

pt = ""
for i in range(len(tabl)):
    for j in range(len(tabl[0])):
        pt += tabl[i][j].rjust(max_len[j] + 1)
        if j < ncols - 2:
            pt += ","

    pt += "\n"

print(pt)
