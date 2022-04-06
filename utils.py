import numpy as np
import random
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import math
import pandas as pd
import sys
import copy
from scipy.optimize import minimize_scalar
import xarray as xr

sys.path.append("../neorl/")

from neorl.benchmarks.classic import all_functions, all_names, all_bounds, all_ndim, all_minima_x
from neorl.benchmarks.cec17 import all_functions as cec_functions
from neorl.hybrid.aeo import get_algo_ngtonevals, get_algo_nmembers, detect_algo, AEO

import pathos.multiprocessing
import tqdm

class FitWrap:
    """little class to count number of times a function has been called"""
    def __init__(self, f):
        self.n = 0
        self.outs = []
        self.ins = []
        self.fxn = f

    def f(self,*inputs):
        ans = self.fxn(*inputs)
        self.n += 1
        self.ins.append(inputs)
        self.outs.append(ans)
        return ans

    def reset(self):
        self.n = 0
        self.outs = []
        self.ins = []

def get_benchmarks(benchset = "all"):
    #get and filter classic functions
    def remove_classic(ilist):
        for i in ilist[::-1]:
            all_functions.pop(i)
            all_names.pop(i)
            all_bounds.pop(i)
            all_ndim.pop(i)
            all_minima_x.pop(i)

    #filter unwanted classic functions
    dimension_criteria_remove = [i for i, a in enumerate(all_ndim) if not (a == "arbitrary" or "+" in a)]
    remove_classic(dimension_criteria_remove)
    xminima_criteria_remove = [i for i,a in enumerate(all_minima_x) if not (a in [-1, 0, 1])]
    remove_classic(xminima_criteria_remove)

    all_minima_y = [a(np.zeros(3) + all_minima_x[i]) for i,a in enumerate(all_functions)]

    #now work on cec17 functions
    cec_names = ["f%i"%(i+1) for i in range(len(cec_functions))]
    cec_bounds = [[-100, 100] for _ in cec_functions]
    cec_minima_y = [100*i for i in range(len(cec_functions))]

    names = all_names + cec_names
    functions = all_functions + cec_functions
    bounds = all_bounds + cec_bounds
    minima = all_minima_y + cec_minima_y

    def remove_bench(ilist):
        for i in ilist[::-1]:
            names.pop(i)
            functions.pop(i)
            bounds.pop(i)
            minima.pop(i)

    if benchset == "all":
        pass
    elif benchset == "classic":
        remove_bench([i + len(all_names) for i in range(len(cec_names))])
    elif benchset == "cec17":
        remove_bench(list(range(len(all_names))))
    elif benchset == "most":
        remove_bench([7,10,14,15,16,21] + list(range(27,31)) + list(range(37,41)) + list(range(48, 52)))
    elif benchset == "some":
        remove_bench(list(range(11,22)) + list(range(25,31)) + list(range(35,41)) + list(range(45, 52)))
    elif benchset == "few":
        remove_bench(list(range(8,22)) + list(range(24,31)) + list(range(34,41)) + list(range(44, 52)))
    elif benchset == "fewest":
        remove_bench(list(range(7,22)) + list(range(24,31)) + list(range(32,52)))

    functions = [FitWrap(f) for f in functions]
    return names, functions, bounds, minima

def run_battery(opt, ddict, fevals = 1000, trials = 5, dims = "all", benchset = "all", ngen = None, nproc = 1):
    """opt in optimizer obj, ddict is non-default parameters not fit our bounds"""
    #pull in all the benchmark functions
    names, functions, bounds, minima = get_benchmarks(benchset)

    clsc = ['sphere', 'cigar', 'rosenbrock', 'bohachevsky', 'griewank', 'rastrigin', 'ackley',
            'rastrigin_scaled', 'rastrigin_skew', 'schaffer', 'schwefel2', 'brown', 'expo', 'yang', 'yang2',
            'yang3', 'yang4', 'zakharov', 'salomon', 'powell', 'happycat', 'levy']

    #select dimension set
    if dims == "all":
        dims = []
        for i, n in enumerate(names):
            if n in clsc:
                dims.append([2, 3, 4, 5, 6, 8, 10, 15, 20, 50])
            elif n in ["f%i"%a for a in range(1,11)]:
                dims.append([2, 10, 20, 30, 50])
            elif n in ["f%i"%a for a in range(11,21)]:
                dims.append([10, 30, 50])
            elif n in ["f%i"%a for a in range(21, 29)]:
                dims.append([2, 10, 20, 30, 50])
            elif n in ["f29", "f30"]:
                dims.append([10, 30, 50])

    elif dims == "most":
        dims = []
        for i, n in enumerate(names):
            if n in clsc:
                dims.append([2, 3, 4, 8, 15, 20])
            elif n in ["f%i"%a for a in range(1,11)]:
                dims.append([2, 10, 20, 30])
            elif n in ["f%i"%a for a in range(11,21)]:
                dims.append([10, 30])
            elif n in ["f%i"%a for a in range(21, 29)]:
                dims.append([2, 10, 20, 30])
            elif n in ["f29", "f30"]:
                dims.append([10, 30])

    elif dims == "few":
        dims = []
        for i, n in enumerate(names):
            if n in clsc:
                dims.append([2, 3, 4, 8])
            elif n in ["f%i"%a for a in range(1,11)]:
                dims.append([2, 10])
            elif n in ["f%i"%a for a in range(11,21)]:
                dims.append([10])
            elif n in ["f%i"%a for a in range(21, 29)]:
                dims.append([2, 10])
            elif n in ["f29", "f30"]:
                dims.append([10])

    elif dims == "fewest":
        dims = []
        for i, n in enumerate(names):
            if n in clsc:
                dims.append([2])
            elif n in ["f%i"%a for a in range(1,11)]:
                dims.append([2])
            elif n in ["f%i"%a for a in range(11,21)]:
                dims.append([10])
            elif n in ["f%i"%a for a in range(21, 29)]:
                dims.append([2])
            elif n in ["f29", "f30"]:
                dims.append([10])

    if dims == "low":
        dims = []
        for i, n in enumerate(names):
            if n in clsc:
                dims.append([10])
            elif n in ["f%i"%a for a in range(1,11)]:
                dims.append([10])
            elif n in ["f%i"%a for a in range(11,21)]:
                dims.append([10])
            elif n in ["f%i"%a for a in range(21, 29)]:
                dims.append([10])
            elif n in ["f29", "f30"]:
                dims.append([10])

    if dims == "med":
        dims = []
        for i, n in enumerate(names):
            if n in clsc:
                dims.append([30])
            elif n in ["f%i"%a for a in range(1,11)]:
                dims.append([30])
            elif n in ["f%i"%a for a in range(11,21)]:
                dims.append([30])
            elif n in ["f%i"%a for a in range(21, 29)]:
                dims.append([30])
            elif n in ["f29", "f30"]:
                dims.append([30])

    if dims == "high":
        dims = []
        for i, n in enumerate(names):
            if n in clsc:
                dims.append([50])
            elif n in ["f%i"%a for a in range(1,11)]:
                dims.append([50])
            elif n in ["f%i"%a for a in range(11,21)]:
                dims.append([50])
            elif n in ["f%i"%a for a in range(21, 29)]:
                dims.append([50])
            elif n in ["f29", "f30"]:
                dims.append([50])

    #start making the pandas dataframe to store results
    colnames = []
    for i, n in enumerate(names):
        for d in dims[i]:
            colnames.append("%s:D%i"%(n, d))
    results = pd.DataFrame(np.zeros((trials, len(colnames))), columns = colnames)

    if not (opt is AEO) and (ngen is None):
        dummy_bounds = {"x%i"%ii:["float"] + bounds[0] for ii in range(dims[0][0])}
        dummy_optimizer = opt(mode = "min", fit = functions[0].f, bounds = dummy_bounds, **ddict)
        ngtoevals = get_algo_ngtonevals(dummy_optimizer)
        nmembers = get_algo_nmembers(dummy_optimizer)
        #    we have function which takes number generations and returns fevals
        #    need to solve inverse problem real quick to input fevals and output generations
        ngen = minimize_scalar(lambda i, a : np.abs(ngtoevals(i,a) - fevals), [2, 1e9], args = (nmembers))
        ngen = int(ngen.x)
    elif opt is AEO: #if opt is AEO
        dummy_bounds = {"x%i"%ii:["float"] + bounds[0] for ii in range(dims[0][0])}
        dummy_optimizer = opt(mode = "min", fit = functions[0].f, bounds = dummy_bounds, **ddict)
        algonames = [detect_algo(a) for a in dummy_optimizer.optimizers]
        tot_nmembers = 0
        for o in dummy_optimizer.optimizers:
            tot_nmembers += get_algo_nmembers(o)
        if "DE" in algonames:
            evals_per_cycle = 2*tot_nmembers*ddict["gen_per_cycle"]*0.98
        else:
            evals_per_cycle = tot_nmembers*ddict["gen_per_cycle"]*0.98

        ncyc = int(math.ceil(fevals/evals_per_cycle))
    #else ngen is given in the args

    #actually run the benchmarks
    argdicts = []
    for i, (f, b) in enumerate(zip(functions, bounds)):
        for j, d in enumerate(dims[i]):
            for k in range(trials):
                argdict = {"trial" : k,
                           "f" : copy.deepcopy(f),
                           "fxn_name" : names[i],
                           "bounds" : b,
                           "dims" : d,
                           "ddict" : ddict,
                           "fevals" : fevals,
                           "seed" : np.random.randint(3, 20000)}
                if opt is AEO:
                    argdict["is_aeo"] = True
                    argdict["n"] = ncyc
                else:
                    argdict["is_aeo"] = False
                    argdict["n"] = ngen

                argdicts.append(argdict)

    def wrapper(argdict):
        """
        argdict = {"trial" : int,
                   "f" : fxn_obj,
                   "fxn_name" : strname,
                   "bounds" : bnds,
                   "is_aeo" : bool,
                   "dims" : d,
                   "ddict" : ddict,
                   "fevals" : int
                   "n" : int
                   "seed" : int
        """
        def stop_crit():
            if len(argdict["f"].outs) > argdict["fevals"]:
                return True
            else:
                return False

        np.random.seed(argdict["seed"])
        random.seed(argdict["seed"])

        thisf = argdict["f"]

        bounds = {"x%i"%ii:["float"] + argdict["bounds"] for ii in range(argdict["dims"])}

        ddict = argdict["ddict"]

        optimizer = opt(mode = "min", fit = thisf.f, bounds = bounds, **ddict)

        if argdict["is_aeo"] == True:
            _, _, aeo_log = optimizer.evolute(argdict["n"], stop_criteria = stop_crit)
            print("Cycles run", aeo_log.attrs["Ncycles"])
            print("Cycles req", argdict["n"])
        else:
            optimizer.evolute(argdict["n"] + 1) #one extra in case of rounding when getting ngen
        y = min(thisf.outs[:argdict["fevals"]])
        if len(thisf.outs[:argdict["fevals"]]) != argdict["fevals"]:
            print("\n", len(thisf.outs),argdict["fxn_name"], "\n")
            aeo_log.to_netcdf(argdict["fxn_name"] + str(len(thisf.outs)))
            raise Exception("-- Error: not enough function evaluations run! This many run: " + str(len(thisf.outs)))
        thisf.reset()
        return [y, argdict["fxn_name"], argdict["dims"], argdict["trial"]]

    pool = pathos.multiprocessing.Pool(processes = nproc)
    res = list(tqdm.tqdm(pool.imap(wrapper, argdicts), total = len(argdicts)))

    for y, n, d, t in res:
        results.loc[t, n + ":D" + str(d)] = y
    return results

def proc_out(path, outfile, scorefile):
    """process big textfile into netcdf and calculate scores"""
    dfstarts = []
    with open(path, "r") as f:
        c = f.readlines()
    for i, l in enumerate(c):
        if l[0] == ",":
            dfstarts.append(i)
    numrows = dfstarts[1] - dfstarts[0] - 1
    das = []
    for j, i in enumerate(dfstarts):
        data_list = [ [float(b.strip()) for b in c[a].split(",")[1:]] for a in range(i + 1, i + numrows + 1)]
        data = np.array(data_list)
        columns = [a.strip() for a in c[i].split(",")[1:]]
        xaray = xr.DataArray(data, dims = ["trial", "f"],
                coords = {"f" : columns, "trial" : np.arange(numrows)})
        das.append(xaray)
    da = xr.concat(das, pd.Index(np.arange(len(das)),name = "config"))
    da.to_netcdf(outfile)

    s = score(da)
    ws = wt_score(da)

    S = np.vstack([s, ws])
    np.savetxt(scorefile, S)

def score(da):
    """do scoring for all configs"""
    ptara = rankdata(da, method = "min", axis = 0)
    config_scores = ptara.sum(2).sum(1)
    return pd.Series(config_scores, index = da.coords["config"])

def firsts(da):
    """give first place percent for all configs"""
    ptara = rankdata(da,  method = "min", axis = 0)
    config_scores = (ptara == 1).sum(2).sum(1)
    config_scores = config_scores/(da.shape[1]*da.shape[2])*100
    np.set_printoptions(threshold = np.inf)
    return pd.Series(config_scores, index = da.coords["config"])

def wt_score(da):
    """do scoring for all configs"""
    ptara = np.argsort(da, axis = 0)
    wts = [np.sqrt(int(a.split(":")[1][1:])) for a in ptara.coords["f"].data]
    config_scores = (wts*ptara).sum("trial").sum("f")
    return config_scores.data

if __name__ == "__main__":
    from neorl import DE
    import time
    ddict = {}
    opt = DE
    start = time.time()
    a = run_battery(opt, ddict, fevals = 2000, trials = 2, dims = "fewest", benchset = "classic", nproc = 60)
    stop = time.time()
    print(stop - start)
