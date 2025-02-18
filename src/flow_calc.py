import numpy as np
import src.mat_functions as mat


def cal_estratificado(D, rho_g, rho_l, mu_g, mu_l, jg, jl, theta, g):
    
    """
    Calcula a altura da interface h, o gradiente de pressão e a fração de vazio
    para um escoamento estratificado gás-líquido.

    Parâmetros:
    D     - Diâmetro da tubulação (m)
    rho_g - Densidade do gás (kg/m³)
    rho_l - Densidade do líquido (kg/m³)
    mu_g  - Viscosidade dinâmica do gás (Pa.s)
    mu_l  - Viscosidade dinâmica do líquido (Pa.s)
    jg    - Velocidade superficial do gás (m/s)
    jl    - Velocidade superficial do líquido (m/s)
    theta - Ângulo da tubulação com a horizontal (radianos)
    
    Retorna:
    h     - Altura da interface (m)
    dPdx  - Gradiente de pressão (Pa/m)
    alpha - Fração de vazio (adimensional)
    """

    A = np.pi * (D**2) / 4  # Área total da tubulação
    Qg = jg * A  # Vazão volumétrica do gás (m³/s)
    Ql = jl * A  # Vazão volumétrica do líquido (m³/s)

    def funcao_interface(h):
        """Função para encontrar a altura da interface"""
        A_g = (np.pi * D**2 / 4) - (D * h - h**2 / 2)
        alpha_h = A_g / A
        return alpha_h - (Qg / (Qg + Ql))

    # Resolver h usando o método da secante
    h = mat.metodo_secante(funcao_interface, 0.1*D, 0.9*D)
    print('h: ', h)
    if h is None:
        return None, None, None

    # Cálculo da fração de vazio
    A_g = (np.pi * D**2 / 4) - (D * h - h**2 / 2)
    alpha = A_g / A

    # Perímetro molhado por cada fase
    S_l = np.pi * D * (1 - alpha)
    S_g = np.pi * D * alpha

    # Diâmetro hidráulico para cada fase
    D_hl = 4 * (A * (1 - alpha)) / S_l
    D_hg = 4 * (A * alpha) / S_g

    # Números de Reynolds
    Re_l = (rho_l * jl * D_hl) / mu_l
    Re_g = (rho_g * jg * D_hg) / mu_g

    # Fator de atrito (Blasius para turbulento)
    f_l = 0.046 / (Re_l**0.2) if Re_l > 2000 else 16 / Re_l
    f_g = 0.046 / (Re_g**0.2) if Re_g > 2000 else 16 / Re_g

    # Gradiente de pressão por atrito
    dPdz_f = (f_l * rho_l * (jl/(1-alpha)))**2 / (2 * D_hl) + (f_g * rho_g * ((jg/alpha)**2) / (2 * D_hg))

    # Gradiente de pressão por aceleração (simplificação)
    dPdz_a = 0

    # Gradiente de pressão por gravidade
    dPdz_g = (rho_l * (1 - alpha) + rho_g * alpha) * g * np.sin(theta)

    # Gradiente de pressão total
    dPdz = -(dPdz_f + dPdz_a + dPdz_g)
    print('dPdz: ', dPdz)
    return dPdz, alpha