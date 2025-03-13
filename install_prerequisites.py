import subprocess
import sys
import os
import directories as dir

def install_package(package):
    """Instala um pacote usando pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    
    flowTech =  os.path.join(dir.LIBS_PATH, "flowtech")
    
    packages = ["numpy", "matplotlib", flowTech]
    for package in packages:
        try:
            install_package(package)
            print(f"{package} instalado com sucesso!")
        except subprocess.CalledProcessError:
            print(f"Erro ao instalar {package}. Verifique sua conexão com a internet e tente novamente.")
