import numpy as np
import pandas as pd
import itertools
import random
import sys
sys.path.append("../neorl")

from neorl import DE
from neorl import ES
from neorl import GWO
from neorl import PSO
from neorl import WOA
from neorl import MFO
from neorl.hybrid.aeo import AEO

from utils import run_battery
from utils import FitWrap

import pathos.multiprocessing
import tqdm

dims = "low"
fevals = 30000
dims = "med"
#fevals = 300000
fevals = 10000
#dims = "high"
#fevals = 3000000

#dummy fxn/variables
def f(x):
    return sum(a**2 for a in x)

f = FitWrap(f)

bounds = {"x%i"%a: ["float", -1, 1] for a in range(3)}

#algorithm initialization
#    diverse ensemble
de = DE(mode = "min", bounds = bounds, fit = f, F = 0.6, CR = 0.4)
gwo = GWO(mode = "min", bounds = bounds, fit = f)
pso = PSO(mode = "min", bounds = bounds, fit = f)
woa = WOA(mode = "min", bounds = bounds, fit = f)
mfo = MFO(mode = "min", bounds = bounds, fit = f)
diverse_algos = [de, gwo, pso, woa, mfo]

#    DE ensemble
de1 = DE(mode = "min", bounds = bounds, fit = f, F = 0.8, CR = 0.2)
de2 = DE(mode = "min", bounds = bounds, fit = f, F = 0.7, CR = 0.3)
de4 = DE(mode = "min", bounds = bounds, fit = f, F = 0.5, CR = 0.5)
de_ensemble = [de1, de2, de, de4]

#    large ensemble
pso2 = PSO(mode = "min", bounds = bounds, fit = f, speed_mech = "timew")
pso3 = PSO(mode = "min", bounds = bounds, fit = f, speed_mech = "globw")
large_ensemble = diverse_algos + [de1, de2, de4] + [pso2]#, pso3]

ensemble_set = {"diverse" : diverse_algos,
                "DE" : de_ensemble,
                "large" : large_ensemble}

#sepcifying gpc used
gpc_set = [3, 10, 50]

#specify different AEOs that will make comparison
aeo_comb = list(itertools.product(list(ensemble_set.keys()), gpc_set))
argdicts = []
for c in aeo_comb:
    d = {"algo" : AEO,
         "algo_name" : "AEO",
         "ensemble_set" : ensemble_set[c[0]],
         "ensemble_set_name" : c[0],
         "gpc" : c[1]}
    argdicts.append(d)


def battery_wrapper(argdict):
    battery_opts = {"fevals" : fevals,
                    "trials" : 15,
                    "dims" : dims, 
                    "benchset" : "all", 
                    "verbose" : True}
    ddict = {"gen_per_cycle" : argdict["gpc"],
            "optimizers" : argdict["ensemble_set"]}
    r = run_battery(argdict["algo"], ddict, **battery_opts)
    csv_name = "comp_results_p1/e%s_g%i_d%s.csv"%(argdict["ensemble_set_name"], 
            argdict["gpc"], dims)
    r.to_csv(csv_name)


pool = pathos.multiprocessing.Pool(processes = len(aeo_comb))
r = list(pool.imap(battery_wrapper, argdicts))


