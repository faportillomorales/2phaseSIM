# 2phaseSIM

## Descrição
O **2phaseSIM** é um código em Python desenvolvido para simulação de escoamentos multifásicos utilizando modelos unidimensionais (1-D). O código recebe parâmetros de entrada por meio de um arquivo de configuração e gera saídas contendo informações do escoamento, que podem ser visualizadas graficamente.

## Estrutura do Projeto
A estrutura de diretórios do projeto é a seguinte:

```
2phaseSIM/
├── output/              # Resultados da simulação
│   ├── logs/           # Arquivos de log gerados durante a execução
│   ├── resume_figs/    # Figuras e gráficos gerados
├── src/                # Código-fonte principal
├── 2phaseSIM.py        # Arquivo principal para execução da simulação
├── directories.py      # Configuração de diretórios do projeto
├── input.py            # Arquivo de configuração de entrada da simulação
├── input_teste.py      # Arquivo de entrada para testes
├── print_out.py        # Módulo de manipulação de saídas
├── run.py              # Script auxiliar para execução da simulação
├── README.md           # Este arquivo
```

## Como Usar

1. Configure os parâmetros da simulação no arquivo `input.py`.
2. Execute o script principal:

   ```bash
   python 2phaseSIM.py
   ```
3. Os resultados da simulação serão armazenados na pasta `output/`.
4. Para visualizar os logs e as figuras, acesse as subpastas `logs/` e `resume_figs/`.

## Dependências
O código requer Python 3 e bibliotecas como:

- NumPy
- Matplotlib

Para instalar as dependências, execute:

```bash
pip install -r requirements.txt
```

## Contato
Para dúvidas ou sugestões, entre em contato com o desenvolvedor do projeto.

