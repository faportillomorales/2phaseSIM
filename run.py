import numpy as np
import sim_input
import src.pattern as pattern
import src.flow_calc as flw
import matplotlib.pyplot as plt
import print_out as prnt
from tqdm import tqdm  # Importa a barra de progresso




def run_separated_phases():
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
    model = sim_input.model
    m_g = j_g * sim_input.rho_g * A_t

    # Inicializando matrizes
    P = np.zeros(nVC+1)  
    dPdz = np.zeros(nVC+1)
    alpha = np.zeros(nVC+1)
    P[0] = P_i  # Definir o valor inicial de P

    # Início do loop no espaço
    for i in tqdm(range(nVC), desc="Simulation in process", unit="it"):
        
        prnt.msg("\n=============================================\n")
        prnt.msg(f"VC: {i} \n")

        rho_g = abs(P[i]) / (T * R_g) 
        j_g= m_g/(A_t*rho_g)

        padrao = pattern.classify_pattern(model,abs(j_l), abs(j_g))
        prnt.msg(f"Pattern: {padrao} \n")


        if padrao == 'Smooth Stratified':
            dPdz[i],alpha[i] = flw.cal_smooth_stratified(D, rho_g, rho_l, mu_g, mu_l, j_g, j_l, theta, g, maxit=1000)

        P[i+1]= P[i] + dL*dPdz[i];

        prnt.print_results(dPdz[i],alpha[i])


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
