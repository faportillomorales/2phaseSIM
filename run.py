import numpy as np
import sim_input  # Renomeie input.py para evitar conflitos
import src.pattern as pattern
import src.flow_calc as flow_calc


def run():
    # Importando variáveis do módulo de configuração
    nVC = sim_input.nVC
    T = sim_input.T
    R_g = sim_input.R_g
    P_i = sim_input.P_i  

    print(f"Número de células de volume: {nVC}")
    P = np.zeros(nVC+1)  

    for i in range(nVC):
        P[0] = P_i  # Definir o valor inicial de P
        rho_g = abs(P[i]) / (T * R_g)  # Adicionando parênteses para garantir prioridade correta

        print(f"Iteração {i}: P = {P[i]:.2f}, rho_g = {rho_g:.5f}")

run()
