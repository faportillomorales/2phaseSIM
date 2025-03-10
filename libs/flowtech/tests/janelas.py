import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Função que recebe os valores dos parâmetros e plota o gráfico
def executar_simulacao(vl, vg):
    # Exemplo de gráfico simples com os valores passados (velocidade do líquido e do gás)
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Simular com os valores passados (aqui estamos apenas plotando um exemplo)
    ax.plot([0, 1, 2, 3], [vl, vg, vl * 2, vg * 2], label=f'Simulação: Vl={vl}, Vg={vg}')
    ax.set_title(f"Simulação com Vl={vl} e Vg={vg}")
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Valor Calculado")
    ax.legend()

    # Renderizar o gráfico na área do Tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Função para atualizar a simulação com base na seleção do OptionMenu
def atualizar_simulacao(opcao_selecionada):
    vl, vg = map(int, opcao_selecionada.split(", "))
    executar_simulacao(vl, vg)

# Criar a janela principal
root = tk.Tk()
root.title("Simulações FlowTech")
root.geometry("1200x600")  # Definir o tamanho da janela

# Dividir a janela em duas partes
frame_buttons = tk.Frame(root, width=200)
frame_buttons.pack(side=tk.LEFT, fill=tk.Y)

frame_plot = tk.Frame(root, width=800)
frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Adicionar label para instrução
label = tk.Label(frame_buttons, text="Escolha os Valores:", font=('Arial', 14))
label.pack(pady=10)

# Opções pré-definidas de velocidades de líquido e gás
opcoes = ["10, 5", "20, 10", "30, 15", "40, 20"]

# Valor inicial do OptionMenu
opcao_selecionada = tk.StringVar()
opcao_selecionada.set(opcoes[0])  # Define a primeira opção como padrão

# Criar o OptionMenu (Dropdown) com as opções
dropdown = ttk.OptionMenu(frame_buttons, opcao_selecionada, opcoes[0], *opcoes, command=atualizar_simulacao)
dropdown.pack(pady=10)

# Executar a janela Tkinter
root.mainloop()
