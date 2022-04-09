import numpy as np
import pandas as pd
import itertools
import random
import sys
sys.path.append("../neorl")

from neorl import PESA2
from neorl import EDEV
from neorl import HCLPSO
from neorl import EPSO

from utils import run_battery
from utils import FitWrap

nproc = 18

#sset = "cec17"
#dest = "1"
#dims = "low"
#fevals = 30000
#dims = "med"
#fevals = 300000
#dims = "high"
#fevals = 3000000

sset = "classic"
dest = "2"
#dims = "low"
#fevals = 20000
#dims = "med"
#fevals = 40000
dims = "high"
fevals = 60000


def f(x):
    return sum(a**2 for a in x)

f = FitWrap(f)

bounds = {"x%i"%a: ["float", -1, 1] for a in range(3)}

algos = [PESA2, EDEV, HCLPSO, EPSO]
algo_name = ["PESA2", "EDEV", "HCLPSO", "EPSO"]
algos = [EDEV]
algo_name = ["EDEV"]

def battery_wrapper(algo, algo_name):
    battery_opts = {"fevals" : fevals,
                    "trials" : 20,
                    "dims" : dims,
                    "benchset" : sset,
                     "nproc" : nproc}

    if algo_name == "PESA2":
        battery_opts["ngen"] = int(np.ceil((fevals-100)/150))
        ddict = {"nwhales" : 5, "memory_size" : 5000}
    elif algo_name == "EDEV":
        battery_opts["ngen"] = int(np.ceil((fevals)/100))
        ddict = {}
    elif algo_name == "HCLPSO":
        battery_opts["ngen"] = int(np.ceil(fevals/40 - 1))
        ddict = {}
    elif algo_name == "EPSO":
        battery_opts["ngen"] = int(np.ceil(fevals/40 - 1))
        ddict = {}

    r = run_battery(algo, ddict, **battery_opts)
    csv_name = "comp_results_p%s/e%s_d%s.csv"%(dest, algo_name, dims)
    r.to_csv(csv_name)

for a, n in zip(algos, algo_name):
    battery_wrapper(a, n)

