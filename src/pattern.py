import sim_input
import flowtechlib as ft
from flowtechlib import exemples
from flowtechlib import dicflowpattern as dfp

j_l = sim_input.j_l
j_g = sim_input.j_g

if sim_input.model == "Shoham2005":
    padrao = sim_input.phe.Shoham2005_function_point(abs(j_l),abs(j_g))