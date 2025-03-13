import argparse
import run
import print_out as prnt
import time
import sim_input

def main():
    parser = argparse.ArgumentParser(description="2phaseSIM: Simulador de escoamentos multifásicos 1-D")
    args = parser.parse_args()

    start_time = time.time()

    # Abro unidade de gravação de saída
    prnt.write_header()

    # Salvar saída em arquivo
    prnt.print_input()
    
    #Roda código
    prnt.msg("🚀  Iniciando simulação...\n\n")
    if (sim_input.method == "Separated"):
        run.run_separated_phases_model()
    
    if (sim_input.method == "Mixture"):
        run.run_mixture_model()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTempo total da simulação: {elapsed_time:.2f} segundos")


if __name__ == "__main__":
    main()
