import json

def carregar_config(caminho_arquivo):
    """
    Carrega os parâmetros da simulação a partir de um arquivo JSON.
    """
    try:
        with open(caminho_arquivo, 'r') as arquivo:
            config = json.load(arquivo)
        validar_config(config)
        return config
    except Exception as e:
        print(f"Erro ao carregar o arquivo de configuração: {e}")
        exit(1)

def validar_config(config):
    """
    Valida os parâmetros da simulação.
    """
    parametros_necessarios = [
        "diametro_tubo", "temperatura", "r_gas", "densidade_gas", "densidade_liquido",
        "viscosidade_gas", "viscosidade_liquido", "angulo_inclinacao", "tensao_superficial",
        "comprimento_tubo", "tamanho_vc", "velocidade_superficial_gas", "velocidade_superficial_liquido"
    ]
    
    for parametro in parametros_necessarios:
        if parametro not in config:
            raise ValueError(f"Parâmetro ausente no arquivo de configuração: {parametro}")
    
    print("Configuração carregada e validada com sucesso.")