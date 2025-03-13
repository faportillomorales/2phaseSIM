import os
import directories as dir
import sim_input as cfg
import datetime
import numpy as np

# Função para abrir o arquivo de saída
def open_output_file(id_file):
    """Abre o arquivo globalmente para escrita."""
    global out_file
    
    out_file = open(f"{dir.LOGS_PATH}\{id_file}.dat", "w", encoding="utf-8")  # Alterei para "output.txt" para evitar erro de formato

# Função para fechar o arquivo de saída
def close_output_file():
    """Fecha o arquivo de saída corretamente."""
    global out_file
    if out_file:
        out_file.close()

# Função para printar mensagens
def msg(msg):
    """Grava a mensagem inserida pelo desenvolvedor no arquivo de saída"""
    global out_file
    
    out_file.write(msg)

def write_header():
    global out_file

    sim_id = "out_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Obtém a data e hora da execução
    data_hora = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    # header formatado
    cabecalho = (
        f"# ID_simulation: {sim_id}\n"
        f"# Date and Time: {data_hora}\n"
        f"# -----------------------------\n"
    )

    # Escreve o cabeçalho no arquivo (modo de escrita 'w' sobrescreve o arquivo)
    open_output_file(sim_id)
    out_file.write(cabecalho)


# Função para imprimir os parâmetros de entrada no arquivo de saída
def print_input():
    """Escreve a configuração inicial da simulação no arquivo de saída."""
    global out_file
    
    out_file.write("============================================\n")
    out_file.write("        2phaseSIM - Simulação 1D\n")
    out_file.write("============================================\n\n")
    out_file.write(">> Configuração da Simulação:\n\n")
    
    out_file.write("================ Parâmetros =================\n")
    out_file.write(f" Método: {cfg.method} \n")
    out_file.write(f" Modelo de padrão: {cfg.pattern_model} \n")
    out_file.write("============================================\n\n")

    # Fluido 1 (Líquido)
    out_file.write("=========== Fluido 1 (Líquido) =============\n")
    out_file.write(f" ⚖  Densidade: {cfg.rho_l} kg/m³\n")
    out_file.write(f" 💧  Viscosidade: {cfg.mu_l} Pa.s\n")
    out_file.write(f" 💨  Velocidade superficial: {cfg.j_l} m/s\n")
    out_file.write("============================================\n\n")
    
    # Fluido 2 (Gás)
    out_file.write("============= Fluido 2 (Gás) ===============\n")
    out_file.write(f" ⚖  Densidade: {cfg.rho_g} kg/m³\n")
    out_file.write(f" 💧  Viscosidade: {cfg.mu_g} Pa.s\n")
    out_file.write(f" 💨  Velocidade superficial: {cfg.j_g} m/s\n")
    out_file.write(f" 🏭  Constante do gás: {cfg.R_g} J/kg.K\n")
    out_file.write("============================================\n\n")
    
    # Outras propriedades gerais
    out_file.write("===== Propriedades Gerais =====\n")
    out_file.write(f" 🏗  Diâmetro do tubo: {cfg.D} m\n")
    out_file.write(f" 🌡  Temperatura: {cfg.T} K\n")
    out_file.write(f" 📐  Ângulo de inclinação: {cfg.theta} graus\n")
    out_file.write(f" 🌊  Tensão superficial: {cfg.sigma} N/m\n")
    out_file.write(f" 📏  Comprimento do tubo: {cfg.L} m\n")
    out_file.write(f" 🔬  Tamanho do volume de controle: {cfg.dL} m\n")
    out_file.write("\n=============================================\n")
    
    out_file.flush()  # Garante que os dados sejam gravados imediatamente

# Função para imprimir os resultados 
def print_results(dPdz,alpha):
    """Função para imprimir os resultados da simulação."""
    global out_file
    
    out_file.write(f"dPdz: {dPdz} \n")
    out_file.write(f"alfa: {alpha}\n")
    out_file.flush()

