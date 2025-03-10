from flowtechlib import *
from flowtechlib import exemples
import matplotlib.pyplot as plt
import pandas as pd
import copy
from flowtechlib import dicflowpattern as dfp
import numpy as np
import scipy.interpolate as interp
import matplotlib.lines as mlines

if __name__ == "__main__":

    parms_bar = copy.copy(exemples.exemple_0_Barnea)
    pat_bar = Patterns(parms_bar)
    phe_bar = Phenom(parms_bar)
    phe_data_bar = PhenomDataDriven(parms_bar)
    phe_data_bar.DecisionTreeOptmizeHyperParms()
    phe_data_bar.RandomForestMapOptmizeHyperParms()
    phe_data_bar.KNNOptmizeHyperParms()
    phe_data_bar.LogisticRegressionOptmizeHyperParms()
    phe_data_bar.SVMOptmizeHyperParms()
    phe_data_bar.MLPCOptmizeHyperParms()
    
