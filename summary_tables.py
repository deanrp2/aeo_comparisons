import numpy as np
from num2tex import num2tex
import pandas as pd
import xarray as xr

short = False #whether or not to cut AEO:DE and AEO where gcp != 3

cec_odds = True #reduce rows to include only CEC odd numbers

commas = False #whether to use latex delims of comma delimes

ds = xr.load_dataset("benchmark_results.nc")


if cec_odds:
    wanted = ["f" + str(int(a)) for a in np.arange(1, 30, 2)]
    ds = ds.loc[{"f" : wanted}]

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
        h = list(ms.keys())[i-2]
        if not "animal" in h and not "large" in h and not "gpc" in h:
            master_lsts[0][i] = h
        else:
            if short:
                if "animal" in h:
                    master_lsts[0][i] = "AEO:Animal"
                elif "large" in h:
                    master_lsts[0][i] = "AEO:Large"
            else:
                g = h.split("c")[-1]
                if "animal" in h:
                    master_lsts[0][i] = "AEO:Animal $N_g$: " + g
                elif "large" in h:
                    master_lsts[0][i] = "AEO:Large $N_g$: " + g
                elif "DE" in h:
                    master_lsts[0][i] = "AEO:DE $N_g$: " + g

    #set index labels
    shk = "{\small "
    for i in np.arange(1, nrows, 2):
        nmms = ms.coords["f"].data[int((i - 1)/2)]
        nmms = nmms.replace("_", " ")
        if len(nmms) > 3 or not "f" in nmms:
            nmms = nmms.title()
        master_lsts[i][0] = shk + nmms + "}"
        master_lsts[i+1][0] = ""

    for i in range(1, nrows):
        if i % 2 == 0:
            master_lsts[i][1] = "$\sigma$"
        else:
            master_lsts[i][1] = "$\mu$"

    #fill in means
    for i,n in enumerate(ms.coords["f"].data):
        row = 2*i + 1
        for j,k in enumerate(list(ms.keys())):
            col = j + 2
            master_lsts[row][col] = float(ms[k].sel(f = n, dim = d).data)
            master_lsts[row+1][col] = float(sds[k].sel(f = n, dim = d).data)

    #find columns where each star belongs
    mara = ms.sel(dim = d).to_array("algo")
    starcols = mara.argmin("algo").data

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
                if master_lsts[i][j] == -1.:
                    master_lsts[i][j] = "-1.0"
                elif master_lsts[i][j] == 0.:
                    master_lsts[i][j] = "0.0"
                elif master_lsts[i][j] < .1 or master_lsts[i][j] > 99.999999999:
                #master_lsts[i][j]= latex_float(master_lsts[i][j])
                    master_lsts[i][j] = "${:.1e}$".format(num2tex(master_lsts[i][j]))
                elif master_lsts[i][j] > 10.:
                    master_lsts[i][j] = "%.0f"%master_lsts[i][j]
                elif master_lsts[i][j] > .1 and master_lsts[i][j] < .99999:
                    master_lsts[i][j] = "%.2f"%master_lsts[i][j]
                else:
                    master_lsts[i][j] = "%.1f"%master_lsts[i][j]
            max_len[j] = max(max_len[j], len(master_lsts[i][j]))

    #adding stars into printed
    for i in range(len(starcols)):
        master_lsts[1 + 2*i][starcols[i] + 2] += "$^*$"


    max_len = np.zeros(ncols, dtype = int)
    for i in range(len(master_lsts)):
        for j in range(len(master_lsts[0])):
            max_len[j] = max(max_len[j], len(master_lsts[i][j]))

    h, _ = np.histogram(starcols, bins = np.arange(-.1, mara.shape[0], 1))
    print("Winn tallies")
    for n, hb in zip(mara.coords["algo"].data, h):
        print(n, hb)

    pt = ""
    for i in range(len(master_lsts)):
        for j in range(len(master_lsts[0])):
            pt += master_lsts[i][j].rjust(max_len[j] + 1)
            if j < ncols - 1:
                if commas:
                    pt += ","
                else:
                    pt += "&"
        if commas == False:
            pt += r"\\"
        pt += "\n"

    print(pt)




