import numpy as np
from num2tex import num2tex
import pandas as pd
import xarray as xr

short = True #whether or not to cut AEO:DE and AEO where gcp != 3

ds = xr.load_dataset("benchmark_results.nc")

if short:
    dedrops = ["DE,gpc" + str(a) for a in [3,10,50]]
    ds = ds.drop(dedrops)
    gdrops = [10, 50]
    drps = []
    for g in gdrops:
        drps.append("animal,gpc" + str(g))
        drps.append("large,gpc" + str(g))
    ds = ds.drop(drps)

ms = ds.mean("trial")
sds = ds.std("trial")

for d in ["low", "med", "high"]:
    print(d, "Dimension")
    ncols = 2 + len(ms.keys())
    nrows = 1 + 2*len(ms.coords["f"])
    master_lsts = np.zeros((nrows, ncols))
    master_lsts = master_lsts.tolist()

    #set header
    master_lsts[0][0] = ""
    master_lsts[0][1] = ""
    for i in range(2, ncols):
        master_lsts[0][i] = list(ms.keys())[i - 2]

    #set index labels
    for i in np.arange(1, nrows, 2):
        master_lsts[i][0] = ms.coords["f"].data[int((i - 1)/2)]
        master_lsts[i+1][0] = ""

    for i in range(1, nrows):
        if i % 2 == 0:
            master_lsts[i][1] = "Std."
        else:
            master_lsts[i][1] = "Mean"

    #fill in means
    for i,n in enumerate(ms.coords["f"].data):
        row = 2*i + 1
        for j,k in enumerate(list(ms.keys())):
            col = j + 2
            master_lsts[row][col] = float(ms[k].sel(f = n, dim = d).data)
            master_lsts[row+1][col] = float(sds[k].sel(f = n, dim = d).data)

    def latex_float(f):
        float_str = "{0:.2g}".format(f)
        if "e" in float_str:
            base, exponent = float_str.split("e")
            return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
        else:
            return float_str

    max_len = np.zeros(ncols, dtype = int)
    for i in range(len(master_lsts)):
        for j in range(len(master_lsts[0])):
            if isinstance(master_lsts[i][j], str):
                pass
            else:
                if master_lsts[i][j] < .1 or master_lsts[i][j] > 99.999999999:
                #master_lsts[i][j]= latex_float(master_lsts[i][j])
                    master_lsts[i][j] = "{:.1e}".format(num2tex(master_lsts[i][j]))
                elif master_lsts[i][j] > 10.:
                    master_lsts[i][j] = "%.0f"%master_lsts[i][j]
                elif master_lsts[i][j] > .1 and master_lsts[i][j] < .99999:
                    master_lsts[i][j] = "%.2f"%master_lsts[i][j]
                else:
                    master_lsts[i][j] = "%.1f"%master_lsts[i][j]
            max_len[j] = max(max_len[j], len(master_lsts[i][j]))


    pt = ""
    for i in range(len(master_lsts)):
        for j in range(len(master_lsts[0])):
            pt += master_lsts[i][j].rjust(max_len[j] + 1)
            if j < ncols - 2:
                pt += ","
        pt += "\n"

    print(pt)




