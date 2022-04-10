import pandas as pd
from pandas import ExcelWriter
import xarray as xr
from pathlib import Path

esets = ["animal", "DE", "large"]
gpcsets = ["3", "10", "50"]

standalone = ["DE", "GWO", "JAYA", "MFO", "PSO", "SSA", "WOA"]#, "PESA2", "EDEV", "EPSO", "HCLPSO"]

algonames = []
for e in esets:
    for g in gpcsets:
        algonames.append(e + ",gpc" + g)

algonames += standalone

dims = ["low", "med", "high"]

wholedas = []
for a in algonames:
    if "," in a:
        aname = "e" + a.split(",")[0] + "_g" + a.split("c")[-1]
    else:
        aname = "e" + a

    das = []
    for i, d in enumerate(dims):
        df = pd.read_csv(Path("comp_results_p1") / Path(aname + "_d" + d + ".csv"), index_col = 0)
        df2 = pd.read_csv(Path("comp_results_p2") / Path(aname + "_d" + d + ".csv"), index_col = 0)
        df[df2.columns] = df2
        cols = [a.split(":")[0] for a in df.columns]
        da = xr.DataArray(df.values, dims = ["trial", "f"], coords = {"f" : cols, "trial" :
            list(range(df.shape[0]))}, name = a)
        das.append(da)

    wholeda = xr.concat(das, pd.Index(dims, name="dim"))
    wholedas.append(wholeda)

res = xr.merge(wholedas)
res.to_netcdf("benchmark_results.nc")

sum_mean = res.mean("trial")
sum_std = res.std("trial")

writer = ExcelWriter("benchmark_results_summary.xlsx")

for k, v in sum_mean.items():
    v.to_pandas().T.to_excel(writer, k + "_mean")
for k, v in sum_std.items():
    v.to_pandas().T.to_excel(writer, k + "_std")

writer.save()


