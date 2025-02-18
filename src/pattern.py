import sim_input
import flowtechlib as ft
from flowtechlib import exemples
from flowtechlib import dicflowpattern as dfp


def classify_pattern(model, j_l, j_g):
    if model == "Shoham2005":
        padrao = sim_input.phe.Shoham2005_function_point(abs(j_l),abs(j_g))
    
    return padrao 

