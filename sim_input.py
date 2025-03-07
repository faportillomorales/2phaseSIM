import numpy as np
import flowtechlib as ft
from flowtechlib import exemples
from flowtechlib import dicflowpattern as dfp
import input_teste


global D, P_i, T, R_g, rho_g, rho_l, mu_g, mu_l, g, beta, sigma, L, dL, j_g, j_l

# Parâmetros de entrada para a simulação 2phaseSIM

# Diâmetro da tubulação (m)
D = 0.060

# Comprimento do tubo (m)
L = 24

# Tamanho do volume de controle (m)
dL = 0.1

# Velocidades superficiais das fases (m/s)
j_g = 2.306 # 0.551  # Gás
j_l = 0.060  # Líquido

#Pressão inicial (Pa)
P_i = 101e3 #7.0E5

# Temperatura do sistema (K)
T = 288

# Constante do gás (J/kg.K)
R_g = 287.053

# Densidade das fases (kg/m³)
rho_g = abs(P_i)/(R_g*T)   # Gás
rho_l = 999  # Líquido

# Viscosidade dinâmica das fases (Pa.s)
mu_g = 2e-5   # Gás
mu_l = 1.14e-3     # Líquido

# Gravidade (m/s)
g = 9.81

# Ângulo de inclinação do tubo (graus) com a horizontal
angulo = 0
theta = angulo*np.pi/180

# Tensão superficial entre as fases (N/m)
sigma = 0.072

model = "Shoham2005"

output_dir_name = "TESTE1"      # optional

#############################################################################
def init():
    """Inicializa campos adicionais e calcula parâmetros derivados."""
    global nVC, rho_g, P, dpdz, alfa
    global A_t, alpha_g, alpha_l, A_g, A_l, m_dot_g, m_dot_l
    global phe, dL
    
    nVC = int(L / dL)               # Número de volumes de controle
    
    # Inicialização de campos
    P = np.zeros(nVC + 1)           # Campo da pressão
    dpdz = np.zeros(nVC + 1)        # Campo de gradiente de pressão
    alfa = np.zeros(nVC + 1)        # Campo de fração de vazio (Void_fraction)

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
    



#############################################################################

# Esecuta inicialização das variáveis globais
init()