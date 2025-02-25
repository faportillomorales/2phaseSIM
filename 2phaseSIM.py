import os
import directories
import argparse
import sim_input
import run
import print_out as prnt
import flowtechlib as ft
from flowtechlib import exemples
from flowtechlib import dicflowpattern as dfp
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="2phaseSIM: Simulador de escoamentos multifásicos 1-D")
    args = parser.parse_args()

    # Abro unidade de gravação de saída
    prnt.write_header()

    # Salvar saída em arquivo
    prnt.print_input()
    
    #Roda código
    prnt.msg("🚀  Iniciando simulação...\n\n")
    run.run()


    
        # f.write("🔄  Executando cálculos...\n")
        # f.write("✅  Simulação concluída. Gerando resultados...\n")

if __name__ == "__main__":
    main()
