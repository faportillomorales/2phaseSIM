import argparse
import input as cfg
import print_out as prnt

def main():
    parser = argparse.ArgumentParser(description="2phaseSIM: Simulador de escoamentos multifásicos 1-D")
    args = parser.parse_args()
    
    # Salvar saída em arquivo
    prnt.print_input()
    # with open("output.txt", "w", encoding="utf-8") as f:
        # f.write("🚀  Iniciando simulação...\n\n")
        # f.write("🔄  Executando cálculos...\n")
        # f.write("✅  Simulação concluída. Gerando resultados...\n")
    
if __name__ == "__main__":
    main()
