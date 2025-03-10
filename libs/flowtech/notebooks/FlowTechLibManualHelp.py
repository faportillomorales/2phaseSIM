import flowtechlib as ft
from flowtechlib import exemples
import matplotlib.pyplot as plt
import pandas as pd
import copy
from flowtechlib import dicflowpattern as dfp
import numpy as np
import scipy.interpolate as interp
import matplotlib.lines as mlines


if __name__ == "__main__":

    ### Carregando um exemplo feito para a biblioteca
    parms = exemples.exemple_0_Shoham
    
    ### Criando instancia Geral
    pat = ft.Patterns(parms)

    ### Informacoes do teste
    pat.info()
    
    ### Criando instancia phenomenological
    phe = ft.Phenom(parms)

    phe.alpha = 1.0 * np.pi / 180.0
    padrao = phe.Shoham2005_function_point(0.8,1.0)

    if padrao == "Slug":
        print("Deu aqui...")
    
    # phe.alpha = 90 * np.pi / 180.0
    # print(phe.Shoham2005_function_point(0.8,1.0))

    # phe.PhenomPatternsMap()

    # dat = ft.PhenomDataDriven(parms)

    # dat.PhenomDataDrivenPatternsMap()

    # dat_hyb = ft.PhenomDataDrivenHybrid(parms, phe.pattern_map)

    # fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    # phe.plot_patterns(fig1, ax=ax1, titlefigure=["Phenomenological Model: "],fontsizeleg=7)
    # plt.show()

    # fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
    # dat.plot_patterns(fig2, ax=ax2, titlefigure=['Data Driven Model: '],fontsizeleg=7)

    # fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5))
    # dat_hyb.plot_patterns(fig3, ax=ax3, titlefigure=['Data Driven Hybrid Model: '],fontsizeleg=7)




