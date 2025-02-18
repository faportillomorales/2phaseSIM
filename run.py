import numpy as np
import sim_input
import src.pattern as pattern
import src.flow_calc as flw
import matplotlib.pyplot as plt


def run():
    # Importando variáveis do módulo de configuração
    g = sim_input.g
    D = sim_input.D
    A_t = sim_input.A_t
    A_g = sim_input.A_g
    A_l = sim_input.A_l
    dL = sim_input.dL
    R = sim_input.D / 2.0 
    rho_g = sim_input.rho_g
    rho_l = sim_input.rho_l
    mu_g = sim_input.mu_g
    mu_l = sim_input.mu_l
    theta = sim_input.theta
    nVC = sim_input.nVC
    T = sim_input.T
    R_g = sim_input.R_g
    P_i = sim_input.P_i  
    m_dot_g = sim_input.m_dot_g
    m_dot_l = sim_input.m_dot_l
    j_g = m_dot_g/(A_t*rho_g)
    j_l = m_dot_l/(A_t*rho_l)

    model = sim_input.model

    # Inicializando matrizes
    P = np.zeros(nVC+1)  
    dPdz = np.zeros(nVC+1)
    alpha = np.zeros(nVC+1)
    P[0] = P_i  # Definir o valor inicial de P

    # Início do loop no espaço
    for i in range(nVC):
        rho_g = abs(P[i]) / (T * R_g)  
        padrao = pattern.classify_pattern(model,abs(j_l), abs(j_g))
        print(padrao)

        if padrao == 'Smooth Stratified':
            dPdz[i],alpha[i] = flw.cal_estratificado(D, rho_g, rho_l, mu_g, mu_l, j_g, j_l, theta, g)

        P[i+1]= P[i] + dL*dPdz[i];
        print ("Pressão: ", P[i+1])    


    plt.figure()
    plt.plot(alpha[:-1], marker='o', linestyle='-', color='b')
    plt.title('Fração de vazio')
    plt.xlabel('SEC')
    plt.ylabel('Alfa')
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(abs(dPdz[:-1]), marker='o', linestyle='-', color='b')
    plt.title('Gráfico de dp/dz total')
    plt.xlabel('SEC')
    plt.ylabel('Valor Absoluto de dp/dz')
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(P, marker='o', linestyle='-', color='b')
    plt.title('Gráfico da Pressão')
    plt.xlabel('SEC')
    plt.ylabel('Pressão')
    plt.grid()
    plt.show()
