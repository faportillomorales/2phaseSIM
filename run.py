import numpy as np
import sim_input
import src.pattern as pattern
import src.flow_calc as flw
import matplotlib.pyplot as plt
import print_out as prnt
from tqdm import tqdm  # Importa a barra de progresso




def run_separated_phases_model():
    # Importando variáveis do módulo de configuração
    g = sim_input.g
    D = sim_input.D
    A_t = sim_input.A_t
    dL = sim_input.dL
    P_i = sim_input.P_i
    rho_g = sim_input.rho_g
    rho_l = sim_input.rho_l
    mu_g = sim_input.mu_g
    mu_l = sim_input.mu_l
    theta = sim_input.theta
    nVC = sim_input.nVC
    T = sim_input.T
    R_g = sim_input.R_g
    j_g = sim_input.j_g 
    j_l = sim_input.j_l 
    pattern_model = sim_input.pattern_model
    m_g = j_g * sim_input.rho_g * A_t # Vazão mássica (constante)

    # Inicializando matrizes
    P = np.zeros(nVC+1)  
    dPdz = np.zeros(nVC)
    alpha = np.zeros(nVC)
    P[-1] = P_i  # Definir o valor inicial de P na saída da tubulação
    
    # Início do loop no espaço
    for i in tqdm(range(nVC,0,-1), desc="Simulation in process", unit="it"):

        
        prnt.msg("\n=============================================\n")
        prnt.msg(f"VC: {i} \n")
        prnt.msg(f"P:, {P[i]} \n")
        rho_g = abs(P[i]) / (T * R_g) 
        j_g= m_g/(A_t*rho_g)

        padrao = pattern.classify_pattern(pattern_model,abs(j_l), abs(j_g))
        prnt.msg(f"Pattern: {padrao} \n")

        # Classificar padrão de escoamento para a VC
        if padrao == 'Smooth Stratified':
            dPdz[i-1],alpha[i-1] = flw.separated_smooth_stratified(D, rho_g, rho_l, mu_g, mu_l, j_g, j_l, theta, g, maxit=1000)

        P[i-1]= P[i] - dL*dPdz[i-1];
        
        prnt.print_results(dPdz[i-1],alpha[i-1])

    # plt.figure()
    # plt.plot(alpha[:-1], marker='o', linestyle='-', color='b')
    # plt.title('Fração de vazio')
    # plt.xlabel('SEC')
    # plt.ylabel('Alfa')
    # plt.grid()
    # plt.show()

    # plt.figure()
    # plt.plot(abs(dPdz[:-1]), marker='o', linestyle='-', color='b')
    # plt.title('Gráfico de dp/dz total')
    # plt.xlabel('SEC')
    # plt.ylabel('Valor Absoluto de dp/dz')
    # plt.grid()
    # plt.show()

    # plt.figure()
    # plt.plot(P, marker='o', linestyle='-', color='b')
    # plt.title('Gráfico da Pressão')
    # plt.xlabel('SEC')
    # plt.ylabel('Pressão')
    # plt.grid()
    # plt.show()

def run_mixture_model():
    # Importando variáveis do módulo de configuração
    g = sim_input.g
    D = sim_input.D
    A_t = sim_input.A_t
    dL = sim_input.dL
    P_i = sim_input.P_i
    rho_g = sim_input.rho_g
    rho_l = sim_input.rho_l
    mu_g = sim_input.mu_g
    mu_l = sim_input.mu_l
    theta = sim_input.theta
    nVC = sim_input.nVC
    T = sim_input.T
    R_g = sim_input.R_g
    j_g = sim_input.j_g 
    j_l = sim_input.j_l 
    epsilon = sim_input.epsilon
    sigma = sim_input.sigma
    pattern_model = sim_input.pattern_model
    m_g = j_g * sim_input.rho_g * A_t # Vazão mássica (constante)

    # Inicializando matrizes
    P = np.zeros(nVC+1)  
    dPdz = np.zeros(nVC)
    alpha = np.zeros(nVC)
    P[:] = P_i  # Definir o valor inicial de P na saída da tubulação
    
    # Início do loop no espaço
    for i in tqdm(range(nVC,0,-1), desc="Simulation in process", unit="it"):

        
        prnt.msg("\n=============================================\n")
        prnt.msg(f"VC: {i} \n")
        prnt.msg(f"P:, {P[i]} \n")
        rho_g = abs(P[i]) / (T * R_g) 
        j_g= m_g/(A_t*rho_g)

        padrao = pattern.classify_pattern(pattern_model,abs(j_l), abs(j_g))
        prnt.msg(f"Pattern: {padrao} \n")

         # Classificar padrão de escoamento para a VC
        if padrao in ('Smooth Stratified', 'Annular'):
            # Cálculo do fator de entranhamento
            E_d = flw.entrainment_fraction(j_g, mu_g, rho_g, rho_l, sigma)
            prnt.msg(f"E_d = {E_d}")
            # dPdz[i-1],alpha[i-1] = flw.mixture_separated_phases(D, rho_g, rho_l, mu_g, mu_l, j_g, j_l, theta, g, maxit=1000)
