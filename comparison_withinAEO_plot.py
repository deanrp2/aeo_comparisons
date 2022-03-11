import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append("../neorl/")

import neorl.benchmarks.classic as cls
import neorl.benchmarks.cec17 as cec


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

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

dims = 10

fxns = [cls.schaffer, cec.f13, cec.f28]
fnames = ["Schaffer", "f13", "f28"]
f_bounds = [[-100, 100], [-100, 100], [-100, 100]]

bnds = []
for b in f_bounds:
    bt = {"x%i"%ii:["float"] + b for ii in range(dims)}
    bnds.append(bt)




fig, ax = plt.subplots(1, 3, figsize = (11, 3), sharex = True)
for i, (f, b) in enumerate(zip(fxns, bnds)):
    f = FitWrap(f)

    bounds = {"x%i"%a: ["float", -1, 1] for a in range(3)}

    #algorithm initialization
    #    animal ensemble
    gwo = GWO(mode = "min", bounds = bounds, fit = f)
    woa = WOA(mode = "min", bounds = bounds, fit = f)
    mfo = MFO(mode = "min", bounds = bounds, fit = f)
    ssa = SSA(mode = "min", bounds = bounds, fit = f)
    animal_ensemble = [gwo, woa, mfo, ssa]

    #    DE ensemble
    de1 = DE(mode = "min", bounds = bounds, fit = f, F = 0.8, CR = 0.2)
    de2 = DE(mode = "min", bounds = bounds, fit = f, F = 0.7, CR = 0.3)
    de3 = DE(mode = "min", bounds = bounds, fit = f, F = 0.6, CR = 0.4)
    de4 = DE(mode = "min", bounds = bounds, fit = f, F = 0.5, CR = 0.5)
    de_ensemble = [de1, de2, de3, de4]

    # large ensemble
    pso = PSO(mode = "min", bounds = bounds, fit = f)
    pso2 = PSO(mode = "min", bounds = bounds, fit = f, speed_mech = "timew")
    jaya = JAYA(mode = "min", bounds = bounds, fit = f)
    large_ensemble = animal_ensemble + [pso, pso2, jaya]

    #specifying ensemble set to use
    ensemble_set = {"animal" : animal_ensemble,
                    "DE" : de_ensemble,
                    "large" : large_ensemble}
    #sepcifying gpc used
    gpc_set = [3, 10, 50]

    #combos to run
    cruns = [["animal", 10], ["DE", 3], ["large", 10]]
    cycles = [10, 6, 3]

    cyclenums = []
    minpts = []
    def cheat_stop_criteria():
        cyclenums.append(len(f.outs))
        minpts.append(np.min(f.outs))
        return False

    colors = ["k", "r", "b", "c"]
    for j, (name, g) in enumerate(cruns):
        e = ensemble_set[name]
        print(name, g)
        a = AEO(mode = "min", bounds = b, optimizers = e, fit = f.f, gen_per_cycle = g)
        a.evolute(cycles[j], stop_criteria = cheat_stop_criteria)
        ax[i].semilogy(np.arange(len(f.outs)), np.minimum.accumulate(f.outs), "--", color = colors[j], linewidth = 1.0, label = name[0].upper() + name[1:] + r", $N_g$: " + str(g))
        ax[i].plot(cyclenums, minpts, "x", color = colors[j], markersize = 4)
        f.reset()
        cyclenums = []
        minpts = []
    ax[i].set_ylabel(fnames[i])
    ax[i].set_xlabel("F evaluations")



ax[2].legend()
plt.tight_layout()

fig.savefig("figures/withinfig.pdf")

plt.show()
