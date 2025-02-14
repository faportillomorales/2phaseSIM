import numpy as np
import flowtechlib as ft
from flowtechlib import exemples
from flowtechlib import dicflowpattern as dfp


global D, P_i, T, R_g, rho_g, rho_l, mu_g, mu_l, g, beta, sigma, L, dL, j_g, j_l

# Parâmetros de entrada para a simulação 2phaseSIM

# Diâmetro da tubulação (m)
D = 0.1

#Pressão inicial (Pa)
P_i = 7.0E5

# Temperatura do sistema (K)
T = 300

# Constante do gás (J/kg.K)
R_g = 287

# Densidade das fases (kg/m³)
rho_g = abs(P_i)/(R_g*T)   # Gás
rho_l = 1000  # Líquido

# Viscosidade dinâmica das fases (Pa.s)
mu_g = 1.8e-5   # Gás
mu_l = 1e-3     # Líquido

# Gravidade (m/s)
g = 9.81

# Ângulo de inclinação do tubo (graus) com a horizontal
angulo = 0
beta = angulo*np.pi/180

# Tensão superficial entre as fases (N/m)
sigma = 0.072

# Comprimento do tubo (m)
L = 500

# Tamanho do volume de controle (m)
dL = 10

# Velocidades superficiais das fases (m/s)
j_g = 1.0  # Gás
j_l = 0.5  # Líquido

model = "Shoham2005"

#############################################################################
def init():
    """Inicializa campos adicionais e calcula parâmetros derivados."""
    global nVC, rho_g, P, dpdz, alfa, h, delta, F_strat, F_anular
    global A_t, alpha_g, alpha_l, A_g, A_l, m_dot_g, m_dot_l
    global phe
    
    nVC = int(L / dL)               # Número de volumes de controle
    rho_g = abs(P_i) / (R_g * T)    # Atualiza a densidade do gás se necessário
    
    # Inicialização de campos
    P = np.zeros(nVC + 1)           # Campo da pressão
    dpdz = np.zeros(nVC + 1)        # Campo de gradiente de pressão
    alfa = np.zeros(nVC + 1)        # Campo de fração de vazio (Void_fraction)
    h = [0.1 * D, 0.9 * D]          # Chutes iniciais para altura da interface (estratificado)
    delta = [0.01*D, 0.40*D]        # Chutes iniciais para espessura do filme de líquido (anular)
    F_strat = []                    # Vetor para salvar valores da equação de quantidade de movimento (estratificado)
    F_anular = []                   # Vetor para salvar valores da equação de quantidade de movimento (anular)

    A_t = np.pi * (D/2)**2            # Área total da tubulação

    # Propriedades do escoamento
    alpha_g = j_g / (j_g + j_l)     # Fração de vazio do gás
    alpha_l = 1 - alpha_g           # Fração de vazio do líquido
    
    # Áreas ocupadas por cada fase
    A_g = alpha_g * A_t             # Área gás
    A_l = alpha_l * A_t             # Área líquido

    # Cálculo da vazão mássica para cada fase
    m_dot_g = rho_g * A_g * j_g     # Vazão mássica do gás
    m_dot_l = rho_l * A_l * j_l     # Vazão mássica do líquido
    
    ## Instancias para Função de Padrão
    parms = exemples.exemple_0_Shoham   # Carregando um exemplo feito para a biblioteca
    pat = ft.Patterns(parms)            # Criando instancia Geral
    pat.info()                          # Informacoes do teste
    phe = ft.Phenom(parms)              # Criando instancia phenomenological


#############################################################################

# Esecuta inicialização das variáveis globais
init()