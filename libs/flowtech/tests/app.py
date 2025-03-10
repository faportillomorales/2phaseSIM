import tkinter as tk
from tkinter import messagebox
from flowtechlib import *  # Supondo que essas sejam classes de FlowTechLib
from flowtechlib import exemples
import matplotlib.pyplot as plt
import pandas as pd
import copy
from flowtechlib import dicflowpattern as dfp
import numpy as np
import scipy.interpolate as interp
import matplotlib.lines as mlines
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class carlos:

    def __init__(self, parms):
        
        self.vl_min = float(parms["vel_min_liq"])
        self.vl_max = float(parms["vel_max_liq"])
        self.vg_min = float(parms["vel_min_gas"])
        self.vg_max = float(parms["vel_max_gas"])
        
        self.angle = str(int(float(parms["incl"])))

        self.data_driven = parms["data_driven"]
        self.fenomenol = parms["fenomenol"]
        self.resol = int(parms["resol"])
        
        self.pattern_plot = []
        
        self.vel_l = np.logspace(np.log10(self.vl_max), np.log10(self.vl_min), num=self.resol)
        self.vel_g = np.logspace(np.log10(self.vg_max), np.log10(self.vg_min), num=self.resol)

        self.V_g, self.V_l = np.meshgrid(self.vel_g, self.vel_l)
        self.ext_point_vl = parms["ext_point_vl"]
        self.ext_point_vg = parms["ext_point_vg"]
        
        self.flow_dic_barnea = dfp.flow_dic_barnea
        self.flow_dic_shoham = dfp.flow_dic_shoham
        self.flow_dic_trallero = dfp.flow_dic_trallero

    def load_data_exp(self, namefile):
        files_load = pd.read_excel(namefile)
        degrees = files_load.loc[:, ["Pipe Angle of Inclination [DEGREE]"]].values
        degree = str(degrees[0][0])
        ext_point_vl = files_load.loc[:, ["Superficial oil Velocity [m/s]"]].values
        ext_point_vg = files_load.loc[:, ["Superficial Gas Velocity [m/s]"]].values
        flow_pattern = files_load.loc[:, ["Flow Pattern [-]"]].values
        return degree, ext_point_vg, ext_point_vl, flow_pattern

    def plotcarlosnew(self, pattern_map, flow_pattern_car, fig, ax, titlefigure, loc="lower center", framealpha=0.4, fontsizeleg=12):
        
        # Configurar o pattern_map
        self.pattern_plot = np.transpose(np.reshape(pattern_map, (self.resol, self.resol)))
        level_aux = np.copy(np.reshape(self.pattern_plot, self.resol * self.resol))
        level_aux = set(level_aux)
        flow_pattern_car_aux = []
        for pos, ele in enumerate(flow_pattern_car):
            flow_pattern_car_aux.append(ele[0])

        flow_pattern_car_aux_2 = set(flow_pattern_car_aux)
        
        # Dicionário de legendas para padrões de escoamento
        leg_dict = {
            "Annular": "AN",
            "Churn": "CH",
            "Dispersed": "DI",
            "Dispersed bubbles": "DB",
            "Dual Continuous": "DC",
            "Intermittent": "IN",
            "Stratified": "ST",
            "Stratified Wavy": "SW" 
        }

        # Definir a lista de classificações para a legenda
        self.legclasspoint = []
        for ele in flow_pattern_car_aux_2:
            self.legclasspoint.append(leg_dict[ele])

        col_aux = []
        lev_aux = [-1]
        self.legclass = []

        # Dicionários para correspondência de fenômenos
        fenomenos_dict = {
            "BARNEA1986": (dfp.col_dict_barnea, dfp.flow_dic_barnea),
            "SHOHAM2005": (dfp.col_dict_shoham, dfp.flow_dic_shoham),
            "TRALLERO1995": (dfp.col_dict_trallero, dfp.flow_dic_trallero)
        }

        # Verificar se o fenômeno existe
        fenomeno = self.fenomenol.upper().replace(" ", "")
        if fenomeno in fenomenos_dict:
            col_dict, flow_dict = fenomenos_dict[fenomeno]
            for ele in level_aux:
                lev_aux.append(ele)
                col_aux.append(col_dict[ele])
                if ele in flow_dict.values():
                    for key, value in flow_dict.items():
                        if value == ele:
                            self.legclass.append(key)
        else:
            print('Method not implemented!')
            return 0
        
        # Dicionário de marcadores para padrões de escoamento
        marker_dict = {
            "Annular": "o",
            "Churn": "s",
            "Dispersed": "^",
            "Dispersed bubbles": "+",
            "Dual Continuous": "x",
            "Intermittent": "D",
            "Stratified": "h",
            "Stratified Wavy": "*" 
        }

        # Verificação de pontos extras de velocidades
        if len(self.ext_point_vl) == len(self.ext_point_vg) > 0:
            # rotulos = [f"$P_{{{i}}}$" for i in range(len(self.ext_point_vl))]
            # rotulos = [f" " for i in range(len(self.ext_point_vl))]
            fig.patch.set_facecolor('white')  # Alterado o fundo para branco
            ax.clear()
            ax.set_xscale('log')
            ax.set_yscale('log')
            for i, rotulo in enumerate(self.ext_point_vg):
                # Plotar os pontos com o marcador correto
                ax.scatter(self.ext_point_vg[i], self.ext_point_vl[i], color='blue', marker=marker_dict[flow_pattern_car[i][0]])
                # Adicionar o rótulo de cada ponto P
                # ax.annotate(rotulo, (self.ext_point_vg[i], self.ext_point_vl[i]), textcoords="offset points", xytext=(0, 14), ha='center', fontsize=18)
        elif len(self.ext_point_vl) != len(self.ext_point_vg):
            print('The extra velocities of the gas and the liquid must have the same dimensions!')
        
        # Plotar o mapa de padrões de escoamento
        grafico = ax.contourf(self.V_g, self.V_l, self.pattern_plot, colors=col_aux, levels=lev_aux, alpha=0.6, extend='max')

        # Configurar a primeira legenda (padrões de escoamento no centro inferior)
        proxy = [plt.Rectangle((self.vg_min, self.vl_min), 1, 1, fc=pc.get_facecolor()[0], edgecolor='black') for pc in grafico.collections]

        # Criar nova legenda para os padrões de escoamento
        legenda = ax.legend(proxy, self.legclass, loc=loc, ncol=len(lev_aux)-1, framealpha=framealpha, fontsize=fontsizeleg)
        ax.add_artist(legenda)  # Adiciona a primeira legenda sem sobrescrever

        # Configurar a segunda legenda com as informações da legclasspoint e marcadores
        proxies_legclasspoint = [
            mlines.Line2D([], [], color='blue', marker=marker_dict[pattern], markersize=10, label=leg_dict[pattern])
            for pattern in flow_pattern_car_aux_2
        ]

        # Adicionar a segunda legenda no canto superior direito
        legenda2 = ax.legend(handles=proxies_legclasspoint, title="Flow Patterns: Carlos", loc='upper right', fontsize=fontsizeleg, framealpha=framealpha)
        ax.add_artist(legenda2)  # Adiciona a segunda legenda sem sobrescrever a primeira

        # Configurar o gráfico
        ax.grid(alpha=0.15, color='black')
        ax.set_title(titlefigure[0] + self.fenomenol + ": Degree= " + self.angle + "°", fontsize=fontsizeleg)
        ax.set_xlabel('Superficial Gas Velocity [m/s]', labelpad=1, fontsize=fontsizeleg)
        ax.set_ylabel('Superficial Liquid Velocity [m/s]', fontsize=fontsizeleg)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Ajustar o layout para garantir que o título e as legendas sejam incluídos
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Ajuste a área da plotagem, deixando espaço para o título
        fig.subplots_adjust(top=0.95)  # Ajustar o topo para não cortar o título

    def plotcarlos(self, pattern_map, flow_pattern_car, fig, ax, titlefigure, loc="lower center", framealpha=0.4, fontsizeleg=12):
        
        # Configurar o pattern_map
        self.pattern_plot = np.transpose(np.reshape(pattern_map, (self.resol, self.resol)))
        level_aux = np.copy(np.reshape(self.pattern_plot, self.resol * self.resol))
        level_aux = set(level_aux)
        flow_pattern_car_aux = []
        for pos, ele in enumerate(flow_pattern_car):
            flow_pattern_car_aux.append(ele[0])

        flow_pattern_car_aux_2 = set(flow_pattern_car_aux)
        
        # Dicionário de legendas para padrões de escoamento
        leg_dict = {
            "Annular": "AN",
            "Churn": "CH",
            "Dispersed": "DI",
            "Dispersed bubbles": "DB",
            "Dual Continuous": "DC",
            "Intermittent": "IN",
            "Stratified": "ST",
            "Stratified Wavy": "SW" 
        }

        # Definir a lista de classificações para a legenda
        self.legclasspoint = []
        for ele in flow_pattern_car_aux_2:
            self.legclasspoint.append(leg_dict[ele])

        col_aux = []
        lev_aux = [-1]
        self.legclass = []

        # Dicionários para correspondência de fenômenos
        fenomenos_dict = {
            "BARNEA1986": (dfp.col_dict_barnea, dfp.flow_dic_barnea),
            "SHOHAM2005": (dfp.col_dict_shoham, dfp.flow_dic_shoham),
            "TRALLERO1995": (dfp.col_dict_trallero, dfp.flow_dic_trallero)
        }

        # Verificar se o fenômeno existe
        fenomeno = self.fenomenol.upper().replace(" ", "")
        if fenomeno in fenomenos_dict:
            col_dict, flow_dict = fenomenos_dict[fenomeno]
            for ele in level_aux:
                lev_aux.append(ele)
                col_aux.append(col_dict[ele])
                if ele in flow_dict.values():
                    for key, value in flow_dict.items():
                        if value == ele:
                            self.legclass.append(key)
        else:
            print('Method not implemented!')
            return 0
        
        # Dicionário de marcadores para padrões de escoamento
        marker_dict = {
            "Annular": "o",
            "Churn": "s",
            "Dispersed": "^",
            "Dispersed bubbles": "+",
            "Dual Continuous": "x",
            "Intermittent": "D",
            "Stratified": "h",
            "Stratified Wavy": "*" 
        }

        # Verificação de pontos extras de velocidades
        if len(self.ext_point_vl) == len(self.ext_point_vg) > 0:
            rotulos = [f"$P_{{{i}}}$" for i in range(len(self.ext_point_vl))]
            fig.patch.set_facecolor('white')  # Alterado o fundo para branco
            ax.clear()
            ax.set_xscale('log')
            ax.set_yscale('log')
            for i, rotulo in enumerate(rotulos):
                # Plotar os pontos com o marcador correto
                ax.scatter(self.ext_point_vg[i], self.ext_point_vl[i], color='blue', marker=marker_dict[flow_pattern_car[i][0]])
                # Adicionar o rótulo de cada ponto P
                ax.annotate(rotulo, (self.ext_point_vg[i], self.ext_point_vl[i]), textcoords="offset points", xytext=(0, 14), ha='center', fontsize=18)
        elif len(self.ext_point_vl) != len(self.ext_point_vg):
            print('The extra velocities of the gas and the liquid must have the same dimensions!')
        
        # Plotar o mapa de padrões de escoamento
        grafico = ax.contourf(self.V_g, self.V_l, self.pattern_plot, colors=col_aux, levels=lev_aux, alpha=0.6, extend='max')

        # Configurar a primeira legenda (padrões de escoamento no centro inferior)
        proxy = [plt.Rectangle((self.vg_min, self.vl_min), 1, 1, fc=pc.get_facecolor()[0], edgecolor='black') for pc in grafico.collections]

        # Criar nova legenda para os padrões de escoamento
        legenda = ax.legend(proxy, self.legclass, loc=loc, ncol=len(lev_aux)-1, framealpha=framealpha, fontsize=fontsizeleg)
        ax.add_artist(legenda)  # Adiciona a primeira legenda sem sobrescrever

        # Configurar a segunda legenda com as informações da legclasspoint e marcadores
        proxies_legclasspoint = [
            mlines.Line2D([], [], color='blue', marker=marker_dict[pattern], markersize=10, label=leg_dict[pattern])
            for pattern in flow_pattern_car_aux_2
        ]

        # Adicionar a segunda legenda no canto superior direito
        legenda2 = ax.legend(handles=proxies_legclasspoint, title="Flow Patterns: Carlos", loc='upper right', fontsize=fontsizeleg, framealpha=framealpha)
        ax.add_artist(legenda2)  # Adiciona a segunda legenda sem sobrescrever a primeira

        # Configurar o gráfico
        ax.grid(alpha=0.15, color='black')
        ax.set_title(titlefigure[0] + self.fenomenol + ": Degree= " + self.angle + "°", fontsize=fontsizeleg)
        ax.set_xlabel('Superficial Gas Velocity [m/s]', labelpad=1, fontsize=fontsizeleg)
        ax.set_ylabel('Superficial Liquid Velocity [m/s]', fontsize=fontsizeleg)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Ajustar o layout para garantir que o título e as legendas sejam incluídos
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Ajuste a área da plotagem, deixando espaço para o título
        fig.subplots_adjust(top=0.95)  # Ajustar o topo para não cortar o título

# Função para calcular o padrão de escoamento com FlowTech
def calcular_fluxo():
    try:
        # Obter os valores inseridos pelo usuário
        vel_liq_min = float(entry_vel_liq_min.get())
        vel_liq_max = float(entry_vel_liq_max.get())
        vel_gas_min = float(entry_vel_gas_min.get())
        vel_gas_max = float(entry_vel_gas_max.get())
        viscosidade_liq = float(entry_visc_liq.get())
        viscosidade_gas = float(entry_visc_gas.get())
        densidade_liq = float(entry_dens_liq.get())
        densidade_gas = float(entry_dens_gas.get())
        tensao_superficial = float(entry_tensao.get())
        diametro = float(entry_diametro.get())
        inclinacao = float(entry_inclinacao.get())
        resolucao = float(entry_resolucao.get())
        
        # Definir os parâmetros para FlowTechLib
        # parametros = {
        #     "fluid1": "Líquido",
        #     "vel_min_liq": vel_liq_min,
        #     "vel_max_liq": vel_liq_max,
        #     "visc_liq": viscosidade_liq,
        #     "dens_liq": densidade_liq,
        #     "fluid2": "Gás",
        #     "vel_min_gas": vel_gas_min,
        #     "vel_max_gas": vel_gas_max,
        #     "visc_gas": viscosidade_gas,
        #     "dens_gas": densidade_gas,
        #     "inte_tens": tensao_superficial,
        #     "diam": diametro,
        #     "incl": inclinacao,
        #     "data_driven": False,  # Supondo que não seja data-driven
        #     "fenomenol": "BARNEA1986",  # Exemplo de modelo
        #     "resol": resolucao,
        #     "ext_point_vl": [],
        #     "ext_point_vg": []
        # }

        parametros = copy.copy(exemples.exemple_0_Barnea)
        parametros["fenomenol"] = "Barnea 1986"
        parametros["resol"] = resolucao
        
        car_bar = carlos(parametros)

        # Criar uma instância de PhenomDataDrivenHybrid (ou qualquer classe aplicável da FlowTechLib)
        
        parametros["incl"], parametros["ext_point_vg"], parametros["ext_point_vl"], flow_pattern = car_bar.load_data_exp("00_degree"+".xlsx")
        
        phenom = Phenom(parametros)
        phenom.PhenomPatternsMap()      # Calcular o padrão de escoamento
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        car_bar.plotcarlosnew(phenom.pattern_map, flow_pattern, fig1, ax=ax1, titlefigure=["Phenomenological Model: "], loc="lower center", framealpha=0.4, fontsizeleg=13)
        # Mostrar o gráfico na interface Tkinter
        canvas = FigureCanvasTkAgg(fig1, master=frame_grafico)  # Adicionar o gráfico ao frame do gráfico
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)

        # Mostrar uma mensagem com o padrão calculado (exemplo)
        messagebox.showinfo("Resultado", f"Fluxo calculado com o modelo {parametros['fenomenol']}.")
    except ValueError:
        messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos.")

# Interface Gráfica
root = tk.Tk()
# Interface Gráfica
root = tk.Tk()
root.title("Simulador de Padrões de Escoamento - FlowTechLib")
# root.geometry("900x500")  # Definir o tamanho inicial da janela
root.attributes("-fullscreen", True)

root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

# Frame esquerdo para os inputs
frame_inputs = tk.Frame(root)
frame_inputs.grid(row=0, column=0, padx=10, pady=10, sticky='nw')

# Frame direito para o gráfico
frame_grafico = tk.Frame(root)
frame_grafico.grid(row=0, column=1, padx=9, pady=9)

# Rótulos e Entradas
label_vel_liq_min = tk.Label(frame_inputs, text="Velocidade Minima do Líquido (m/s):")
label_vel_liq_min.pack(pady=5)
entry_vel_liq_min = tk.Entry(frame_inputs)
entry_vel_liq_min.pack(pady=5)

label_vel_liq_max = tk.Label(frame_inputs, text="Velocidade Maxima do Líquido (m/s):")
label_vel_liq_max.pack(pady=5)
entry_vel_liq_max = tk.Entry(frame_inputs)
entry_vel_liq_max.pack(pady=5)

label_vel_gas_min = tk.Label(frame_inputs, text="Velocidade Minima do Gás (m/s):")
label_vel_gas_min.pack(pady=5)
entry_vel_gas_min = tk.Entry(frame_inputs)
entry_vel_gas_min.pack(pady=5)

label_vel_gas_max = tk.Label(frame_inputs, text="Velocidade Maxima do Gás (m/s):")
label_vel_gas_max.pack(pady=5)
entry_vel_gas_max = tk.Entry(frame_inputs)
entry_vel_gas_max.pack(pady=5)

label_visc_liq = tk.Label(frame_inputs, text="Viscosidade do Líquido (Pa.s):")
label_visc_liq.pack(pady=5)
entry_visc_liq = tk.Entry(frame_inputs)
entry_visc_liq.pack(pady=5)

label_visc_gas = tk.Label(frame_inputs, text="Viscosidade do Gás (Pa.s):")
label_visc_gas.pack(pady=5)
entry_visc_gas = tk.Entry(frame_inputs)
entry_visc_gas.pack(pady=5)

label_dens_liq = tk.Label(frame_inputs, text="Densidade do Líquido (kg/m³):")
label_dens_liq.pack(pady=5)
entry_dens_liq = tk.Entry(frame_inputs)
entry_dens_liq.pack(pady=5)

label_dens_gas = tk.Label(frame_inputs, text="Densidade do Gás (kg/m³):")
label_dens_gas.pack(pady=5)
entry_dens_gas = tk.Entry(frame_inputs)
entry_dens_gas.pack(pady=5)

label_tensao = tk.Label(frame_inputs, text="Tensão Superficial (N/m):")
label_tensao.pack(pady=5)
entry_tensao = tk.Entry(frame_inputs)
entry_tensao.pack(pady=5)

label_diametro = tk.Label(frame_inputs, text="Diâmetro do Tubo (m):")
label_diametro.pack(pady=5)
entry_diametro = tk.Entry(frame_inputs)
entry_diametro.pack(pady=5)

label_inclinacao = tk.Label(frame_inputs, text="Inclinação (graus):")
label_inclinacao.pack(pady=5)
entry_inclinacao = tk.Entry(frame_inputs)
entry_inclinacao.pack(pady=5)

label_resolucao = tk.Label(frame_inputs, text="Resolução :")
label_resolucao.pack(pady=5)
entry_resolucao = tk.Entry(frame_inputs)
entry_resolucao.pack(pady=5)

# Botão para calcular o padrão de escoamento
button_calcular = tk.Button(frame_inputs, text="Calcular Padrão de Escoamento", command=calcular_fluxo)
button_calcular.pack(pady=20)

# Iniciar o loop da interface
root.mainloop()
