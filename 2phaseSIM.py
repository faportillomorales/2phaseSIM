import argparse
import os
import json
from check_inputs import carregar_config

  # Usa o primeiro arquivo JSON encontrado

def main():
    parser = argparse.ArgumentParser(description="2phaseSIM: Simulador de escoamentos multifásicos 1-D")
    args = parser.parse_args()
    
    # Encontrar automaticamente o arquivo de configuração
    path_input = "input.json"
    print(f"Usando arquivo de configuração: {path_input}")
    
    # Carregar os parâmetros da simulação
    config = carregar_config(path_input)
    print("Configuração carregada com sucesso.")
    
    # Chamaremos o solver e executaremos a simulação
    print("Executando a simulação...")
    
    # Após a execução, exibimos os resultados
    print("Simulação concluída. Gerando resultados...")
    
if __name__ == "__main__":
    main()
