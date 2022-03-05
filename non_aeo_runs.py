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
from neorl import SSA
from neorl import JAYA
from neorl.hybrid.aeo import AEO

from utils import run_battery
from utils import FitWrap

nproc = 30

dims = "low"
fevals = 30000
dims = "med"
fevals = 300000
dims = "high"
fevals = 3000000

def f(x):
    return sum(a**2 for a in x)

f = FitWrap(f)

bounds = {"x%i"%a: ["float", -1, 1] for a in range(3)}

#algorithm initialization
#    animal ensemble
gwo = GWO(mode = "min", bounds = bounds, fit = f)
woa = WOA(mode = "min", bounds = bounds, fit = f)
mfo = MFO(mode = "min", bounds = bounds, fit = f)
ssa = SSA(mode = "min", bounds = bounds, fit = f)
de = DE(mode = "min", bounds = bounds, fit = f)
pso = PSO(mode = "min", bounds = bounds, fit = f)
jaya = JAYA(mode = "min", bounds = bounds, fit = f)

algos = [GWO, WOA, MFO, SSA, DE, PSO, JAYA]
algo_name = ["GWO", "WOA", "MFO", "SSA", "DE", "PSO", "JAYA"]


def battery_wrapper(algo, algo_name):
    battery_opts = {"fevals" : fevals,
                    "trials" : 20,
                    "dims" : dims, 
                    "benchset" : "all", 
                     "nproc" : nproc}
    r = run_battery(algo, {}, **battery_opts)
    csv_name = "comp_results_p1/e%s_d%s.csv"%(algo_name, dims)
    r.to_csv(csv_name)

for a, n in zip(algos, algo_name):
    battery_wrapper(a, n)

