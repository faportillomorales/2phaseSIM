from flowtechlib import *
from flowtechlib import exemples
import matplotlib.pyplot as plt
import pandas as pd
import copy
from flowtechlib import dicflowpattern as dfp
import numpy as np
import scipy.interpolate as interp
import matplotlib.lines as mlines

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

    def plotcarlosold1(self, pattern_map, flow_pattern_car, fig, ax, titlefigure, loc="lower center", framealpha=0.4, fontsizeleg=12):
            
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
            "Dispersed": "D",
            "Dispersed bubbles": "DB",
            "Dual Continuous": "DC",
            "Intermittent": "I",
            "Stratified": "ST",
            "Stratified Wavy": "SW" 
        }

        # Definir a lista de classificações para a legenda
        self.legclasspoint = []
        for ele in flow_pattern_car_aux_2:
            self.legclasspoint.append(leg_dict[ele])

        print(self.legclasspoint)

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

        # Configurar a legenda para os padrões de escoamento (no centro inferior)
        proxy = [plt.Rectangle((self.vg_min, self.vl_min), 1, 1, fc=pc.get_facecolor()[0], edgecolor='black') for pc in grafico.collections]
        
        legenda = fig.legend('')
        legenda.remove()  # Remove a legenda anterior para evitar sobreposição

        # Criar nova legenda para os padrões de escoamento
        legenda = ax.legend(proxy, self.legclass, loc=loc, ncol=len(lev_aux)-1, framealpha=framealpha, fontsize=fontsizeleg)

        # Configurar a legenda superior direita com as informações da legclasspoint e marcadores
        proxies_legclasspoint = [
            mlines.Line2D([], [], color='blue', marker=marker_dict[pattern], markersize=10, label=leg_dict[pattern])
            for pattern in flow_pattern_car_aux_2
        ]

        # Adicionar a segunda legenda no canto superior direito
        legenda2 = ax.legend(handles=proxies_legclasspoint, title="Flow Patterns: Carlos", loc='upper right', fontsize=fontsizeleg, framealpha=framealpha)
        ax.add_artist(legenda2)

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

    def plotcarlosold0(self, pattern_map, flow_pattern_car, fig, ax, titlefigure, loc="lower center", framealpha=0.4, fontsizeleg=12):
        
        # Configurar o pattern_map
        self.pattern_plot = np.transpose(np.reshape(pattern_map, (self.resol, self.resol)))
        level_aux = np.copy(np.reshape(self.pattern_plot, self.resol * self.resol))
        level_aux = set(level_aux)
        flow_pattern_car_aux = []
        for pos,ele in enumerate(flow_pattern_car):
            flow_pattern_car_aux.append(ele[0])

        flow_pattern_car_aux_2 = set(flow_pattern_car_aux)
        
        leg_dict = {
            "Annular" : "AN",
            "Churn" : "CH",
            "Dispersed" : "D",
            "Dispersed bubbles" : "DB",
            "Dual Continuous" : "DC",
            "Intermittent" : "I",
            "Stratified" : "ST",
            "Stratified Wavy" : "SW" 
        }

        self.legclasspoint = []
        for ele in flow_pattern_car_aux_2:
            self.legclasspoint.append(leg_dict[ele])
        
        print(self.legclasspoint)

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
        
        marker_dict = {
            "Annular" : "o",
            "Churn" : "s",
            "Dispersed" : "^",
            "Dispersed bubbles" : "+",
            "Dual Continuous" : "x",
            "Intermittent" : "D",
            "Stratified" : "h",
            "Stratified Wavy" : "*" 
        }

        # Verificação de pontos extras de velocidades
        if len(self.ext_point_vl) == len(self.ext_point_vg) > 0:
            rotulos = [f"$P_{{{i}}}$" for i in range(len(self.ext_point_vl))]
            fig.patch.set_facecolor('white')  # Alterado o fundo para branco
            ax.clear()
            ax.set_xscale('log')
            ax.set_yscale('log')
            for i, rotulo in enumerate(rotulos):
                ax.scatter(self.ext_point_vg[i], self.ext_point_vl[i], color='blue', marker=marker_dict[flow_pattern_car[i][0]])
                ax.annotate(rotulo, (self.ext_point_vg[i], self.ext_point_vl[i]), textcoords="offset points", xytext=(0, 14), ha='center', fontsize=18)
            
        elif len(self.ext_point_vl) != len(self.ext_point_vg):
            print('The extra velocities of the gas and the liquid must have the same dimensions!')
        
        # Plotar o mapa de padrões de escoamento
        grafico = ax.contourf(self.V_g, self.V_l, self.pattern_plot, colors=col_aux, levels=lev_aux, alpha=0.6, extend='max')

        # Configurar legenda
        proxy = [plt.Rectangle((self.vg_min, self.vl_min), 1, 1, fc=pc.get_facecolor()[0], edgecolor='black') for pc in grafico.collections]
        
        legenda = fig.legend('')
        legenda.remove()  # Remove a legenda anterior para evitar sobreposição

        # Criar nova legenda
        legenda = ax.legend(proxy, self.legclass, loc=loc, ncol=len(lev_aux)-1, framealpha=framealpha, fontsize=fontsizeleg)

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

    def plotcarlosold(self, pattern_map, fig, ax, titlefigure, loc="lower center", framealpha=0.4, fontsizeleg=12):
        
        # Configurar o pattern_map
        self.pattern_plot = np.transpose(np.reshape(pattern_map, (self.resol, self.resol)))
        level_aux = np.copy(np.reshape(self.pattern_plot, self.resol * self.resol))
        level_aux = set(level_aux)

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

        # Verificação de pontos extras de velocidades
        if len(self.ext_point_vl) == len(self.ext_point_vg) > 0:
            rotulos = [f"$P_{{{i}}}$" for i in range(len(self.ext_point_vl))]
            fig.patch.set_facecolor('white')  # Alterado o fundo para branco
            ax.clear()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.scatter(self.ext_point_vg, self.ext_point_vl, color='blue', marker='o')
            for i, rotulo in enumerate(rotulos):
                ax.annotate(rotulo, (self.ext_point_vg[i], self.ext_point_vl[i]), textcoords="offset points", xytext=(0, 14), ha='center', fontsize=18)
        elif len(self.ext_point_vl) != len(self.ext_point_vg):
            print('The extra velocities of the gas and the liquid must have the same dimensions!')
        
        # Plotar o mapa de padrões de escoamento
        grafico = ax.contourf(self.V_g, self.V_l, self.pattern_plot, colors=col_aux, levels=lev_aux, alpha=0.6, extend='max')

        # Configurar legenda
        proxy = [plt.Rectangle((self.vg_min, self.vl_min), 1, 1, fc=pc.get_facecolor()[0], edgecolor='black') for pc in grafico.collections]
        
        legenda = fig.legend('')
        legenda.remove()  # Remove a legenda anterior para evitar sobreposição
        
        # Criar nova legenda
        legenda = ax.legend(proxy, self.legclass, loc=loc, ncol=len(lev_aux)-1, framealpha=framealpha, fontsize=fontsizeleg)
        
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

if __name__ == "__main__":

    parms_bar = copy.copy(exemples.exemple_carlos)
    parms_sho = copy.copy(exemples.exemple_carlos)
    
    parms_bar["resol"] = "300"
    parms_sho["resol"] = "300"

    parms_bar["vel_max_gas"] = "4.0"
    parms_sho["vel_max_gas"] = "4.0"

    pat_bar = Patterns(parms_bar)
    pat_sho = Patterns(parms_sho)

    car_bar = carlos(parms_bar)
    car_sho = carlos(parms_sho)

    cases_test = ["00_degree","05_degree","10_degree","90_degree"]
    
    for filecase in cases_test:
        print(filecase)
        parms_bar["incl"], parms_bar["ext_point_vg"], parms_bar["ext_point_vl"], flow_pattern = car_bar.load_data_exp(filecase+".xlsx")
        parms_sho["incl"], parms_sho["ext_point_vg"], parms_sho["ext_point_vl"], flow_pattern = car_sho.load_data_exp(filecase+".xlsx")

        car_bar = carlos(parms_bar)
        car_sho = carlos(parms_sho)

        phe_sho = Phenom(parms_sho)
        phe_sho.PhenomPatternsMap()

        parms_bar["fenomenol"] = "Barnea 1986"
        
        phe_bar = Phenom(parms_bar)
        phe_bar.PhenomPatternsMap()

        fig1, ax1 = plt.subplots(1, 1, figsize=(9, 9))
        car_bar.plotcarlosnew(phe_bar.pattern_map, flow_pattern, fig1, ax=ax1, titlefigure=["Phenomenological Model: "], loc="lower center", framealpha=0.4, fontsizeleg=13)
        plt.savefig(filecase+"_Barnea_Point.png")
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(9, 9))
        car_sho.plotcarlosnew(phe_sho.pattern_map, flow_pattern, fig2, ax=ax2, titlefigure=["Phenomenological Model: "], loc="lower center", framealpha=0.4, fontsizeleg=13)
        plt.savefig(filecase+"_Shoham_Point.png")

        # plt.show()


