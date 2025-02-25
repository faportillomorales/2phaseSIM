import sim_input
import flowtechlib as ft
from flowtechlib import exemples
from flowtechlib import dicflowpattern as dfp
import matplotlib.pyplot as plt

## Instancias para Função de Padrão
parms = exemples.exemple_0_Shoham #input_teste.param            #  # Carregando um exemplo feito para a biblioteca
pat = ft.Patterns(parms)            # Criando instancia Geral
phe = ft.Phenom(parms)              # Criando instancia phenomenological

pat.alpha = sim_input.theta
pat.d = sim_input.D
phe.alpha = sim_input.theta
phe.d = sim_input.D
# pat.info()                          # Informacoes do teste
phe.info()
# phe.PhenomPatternsMap()

# fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
# phe.plot_patterns(fig1, ax=ax1, titlefigure=["Phenomenological Model: "],fontsizeleg=7)

def classify_pattern(model, j_l, j_g):
    if model == "Shoham2005":
        padrao =phe.Shoham2005_function_point(abs(j_l),abs(j_g))
        
    return padrao 

