import argparse
import input as cfg
import print_out as prnt

def main():
    parser = argparse.ArgumentParser(description="2phaseSIM: Simulador de escoamentos multifásicos 1-D")
    args = parser.parse_args()
    
    # Salvar saída em arquivo
    prnt.print_input()
    
if __name__ == "__main__":
    main()
