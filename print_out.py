import sim_input as cfg

def print_input():
    with open("output", "w", encoding="utf-8") as f:
        f.write("=============================================\n")
        f.write("        2phaseSIM - Simulação 1D\n")
        f.write("=============================================\n\n")
        f.write(">> Configuração da Simulação:\n\n")
        
        # Fluido 1 (Líquido)
        f.write("===== Fluido 1 (Líquido) =====\n")
        f.write(f" ⚖  Densidade: {cfg.rho_l} kg/m³\n")
        f.write(f" 💧  Viscosidade: {cfg.mu_l} Pa.s\n")
        f.write(f" 💨  Velocidade superficial: {cfg.j_l} m/s\n")
        f.write("=============================================" + "\n\n")

        # Aba Fluido 2 (Gás)
        f.write("===== Fluido 2 (Gás) =====\n")
        f.write(f" ⚖  Densidade: {cfg.rho_g} kg/m³\n")
        f.write(f" 💧  Viscosidade: {cfg.mu_g} Pa.s\n")
        f.write(f" 💨  Velocidade superficial: {cfg.j_g} m/s\n")
        f.write(f" 🏭  Constante do gás: {cfg.R_g} J/kg.K\n")
        f.write("=============================================" + "\n\n")

        # Outras propriedades gerais
        f.write("===== Propriedades Gerais =====\n")
        f.write(f" 🏗  Diâmetro do tubo: {cfg.D} m\n")
        f.write(f" 🌡  Temperatura: {cfg.T} K\n")
        f.write(f" 📐  Ângulo de inclinação: {cfg.beta} graus\n")
        f.write(f" 🌊  Tensão superficial: {cfg.sigma} N/m\n")
        f.write(f" 📏  Comprimento do tubo: {cfg.L} m\n")
        f.write(f" 🔬  Tamanho do volume de controle: {cfg.dL} m\n")
        f.write("\n=============================================" + "\n")

