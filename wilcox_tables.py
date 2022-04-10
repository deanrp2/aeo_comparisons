import numpy as np
import xarray as xr
from scipy.stats import wilcoxon

short = True

#low p value means different

ds = xr.load_dataset("benchmark_results.nc")
ds = ds.drop(["PESA2", "EDEV", "EPSO", "HCLPSO"])

compare = "large,gpc3"

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
nrows = len(ds.coords["f"]) + 1 + 2
ncols = len(ds.keys()) + 1
tabl = np.zeros((nrows, ncols), dtype = str)
tabl = tabl.tolist()

#fill in row names
for i in range(1, nrows - 2):
    nm = ds.coords["f"].data[i-1]
    if len(nm) > 3:
        nm = nm.replace("_", " ").title()
    tabl[i][0] = nm

tabl[-2][0] = "Total ``+''"
tabl[-1][0] = "Total ``-''"

#fill in column names
for i in range(1, ncols):
    h = list(ds.keys())[i-1]
    if not ("gpc" in h):
        tabl[0][i] = h
    else:
        if short:
            if "animal" in h:
                tabl[0][i] = "AEO:Animal"
            elif "large" in h:
                tabl[0][i] = "AEO:Large"
        else:
            g = h.split("c")[-1]
            if "animal" in h:
                tabl[0][i] = "AEO:Animal $N_g$: " + g
            if "large" in h:
                tabl[0][i] = "AEO:Large $N_g$: " + g
            if "DE" in h:
                tabl[0][i] = "AEO:DE $N_g$: " + g


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
                    p = "0 "
                elif np.mean(x) > np.mean(y):
                    p = "- "
                elif np.mean(y) > np.mean(x):
                    p = "+ "
                else:
                    raise Exception("weird p value")
            tabl[i + 1][j + 1] += p

for j, a in enumerate(ds.keys()):
    npp = [0, 0, 0]
    nm = [0, 0, 0]
    for i, f in enumerate(ds.coords["f"]):
        c = [a == "+" for a in tabl[i+1][j+1].split(" ")[:-1]]
        c2 = [a == "-" for a in tabl[i+1][j+1].split(" ")[:-1]]

        for v in range(len(c)):
            npp[v] += int(c[v])
            nm[v] += int(c2[v])
    npp = [str(a) for a in npp]
    nm = [str(a) for a in nm]
    ep = "/".join(npp)
    em = "/".join(nm)
    tabl[-2][j + 1] = ep
    tabl[-1][j + 1] = em


#print table nicely
max_len = np.zeros(ncols, dtype = int)
for i in range(len(tabl)):
    for j in range(len(tabl[0])):
        max_len[j] = max(max_len[j], len(tabl[i][j]))

pt = ""
for i in range(len(tabl)):
    for j in range(len(tabl[0])):
        pt += tabl[i][j].rjust(max_len[j] + 1)
        if j < ncols - 1:
            pt += ","

    pt += "\n"

print(pt)
