import numpy as np
import src.mat_functions as mat
import print_out as prnt


def cal_smooth_stratified(D, rho_g, rho_l, mu_g, mu_l, jg, jl, theta, g, maxit):
    
    """
    Calcula a altura da interface h, o gradiente de pressão e a fração de vazio
    para um escoamento estratificado gás-líquido.

    Entrada:
    D       - Diâmetro da tubulação (m)
    rho_g   - Densidade do gás (kg/m³)
    rho_l   - Densidade do líquido (kg/m³)
    mu_g    - Viscosidade dinâmica do gás (Pa.s)
    mu_l    - Viscosidade dinâmica do líquido (Pa.s)
    jg      - Velocidade superficial do gás (m/s)
    jl      - Velocidade superficial do líquido (m/s)
    theta   - Ângulo da tubulação com a horizontal (radianos)

    Local:
    delta   - Espessura de file líquido adimensional
    Lambda  - Ângulo formado pela interface plana
    
    Retorna:
    h       - Altura da interface (m)
    dPdx    - Gradiente de pressão (Pa/m)
    alpha   - Fração de vazio (adimensional)
    """
    h = [0.1*D, 0.9*D]      # Chute inicial
    A = np.pi * (D**2) / 4  # Área total da tubulação

    F = []

    for it in range(maxit):
        delta = h[it]/D                     # Espessura de file líquido adimensional
        Lambda = 2*np.arccos(1-2*delta)     # Ângulo formado pela interface plana

        # Cálculo da fração de vazio
        A_l = D**2 * (Lambda-np.sin(Lambda))/8  # Área líquido
        # A_l = (Lambda/2)*np.square(D/2) - (D/2)**2*np.sin((Lambda/2))*np.cos((Lambda/2))
        A_g = A - A_l                           # Área gás
        alpha = A_g / A                         # Fração de vazio (fração de gás)

        # Velocidades médias das fases (in situ)
        Ul = jl/(1-alpha)
        Ug = jg/alpha

        # Perímetro molhado por cada fase
        S_l = D * Lambda/2
        S_g = D * (np.pi - (Lambda/2))
        S_i = D * np.sin(Lambda/2)

        # Diâmetro hidráulico para cada fase
        Dl = 4*A_l/S_l                          #[m] diâmetro hidráulico do líquido
        Dg = 4*A_g/(S_g+S_i)                    #[m] diâmetro hidráulico do gás
        
        # Números de Reynolds
        Re_l = (rho_l * Ul * Dl) / mu_l
        Re_g = (rho_g * Ug * Dg) / mu_g
        
        # Fator de atrito
        f_l = 0.046 * (Re_l**(-0.2)) if Re_l > 2300 else 16 / Re_l        ###### DIVERGE DO ESPERADO
        # f_l = 16 / Re_l                                                     ###### CONSIDERANDO LAMINAR PARA O LÍQUIDO
        f_g = 0.046 * (Re_g**(-0.2)) if Re_g > 2300 else 16 / Re_g          #####  MUDANDO RE>2000 PARA 2300 SE APROXIMA DO VALOR
        # f_g = 16 / Re_g

        # Tensões cisalhantes (parietal e de interface)
        Tau_W_l = f_l * rho_l * (Ul**2) / 2              # Tensão parietal do líquido [Pa]
        Tau_W_g = f_g * rho_g * (Ug**2) / 2              # Tensão parietal do gás [Pa]

        # f_i = f_g   #For Smooth Stratified Flow   Taitel and Dukler 
        # f_i = 0.0142                #Cohen and Hanratty
        # f_i = 1.3*Re_g**(-0.57)   #Agrawal
        # Re_sg = rho_g*jg*D/mu_g ; f_i = 0.96*Re_sg**(-0.52)   #Kowalski
        # f_i = 0.0625*(np.log((15/Re_g)+(0.001/3.715*D)))**(-2)    #Crowley
        f_i = 5*f_g   #Hart
        Tau_i = f_i * rho_g * ((Ug - Ul)**2) / 2      # Tensão interfacial [Pa]

        # Equação de quantidade de movimento - residue has to be zero
        res = Tau_W_g * (S_g / A_g) - Tau_W_l * (S_l / A_l) + Tau_i * S_i * ((1/A_l) + (1/A_g)) + (rho_l - rho_g) * g * np.sin(theta)
        F.append(res)
        E = 0
        
        # Método da secante
        if it > 0: 
            h_new = h[it] - ((h[it]-h[it-1])/(F[it]-F[it-1]))*F[it]

            h.append(h_new)
            
            if h[it + 1] > 0.999 * D:
                h[it + 1] = 0.999 * D
            if h[it + 1] < 0.001 * D:
                h[it + 1] = 0.001 * D            
                
            E = (h[it+1] - h[it]) / h[it+1]

        # Critério de convergência
        if abs(F[it])<1e-10 and it>2:
            break

    prnt.msg(f"n iterações: {it} \n")
    prnt.msg(f"j_g: {jg} \n")
    prnt.msg(f"h: {h[it]} \n")
    prnt.msg(f"D_g: {Dg} \n")
    prnt.msg(f"Re_g: {Re_g} \n")
    prnt.msg(f"Re_l: {Re_l} \n")
    
    # Gradiente de pressão friccional
    dPdz_l_f = (-Tau_W_l * S_l + Tau_i * S_i )/A_l
    dPdz_g_f = (-Tau_W_g * S_g - Tau_i * S_i )/A_g

    # Gradiente de pressão por gravidade
    dPdz_l_g = (rho_l * A_l * g * np.sin(theta))/A_l
    dPdz_g_g = (rho_g * A_g * g * np.sin(theta))/A_g

    # Gradiente de pressão total
    dPdz_l = dPdz_l_f + dPdz_l_g
    dPdz_g = dPdz_g_f + dPdz_g_g

    return dPdz_g, alpha 


