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

if __name__ == "__main__":

    # parms_bar = copy.copy(exemples.exemple_carlos)
    parms_bar = copy.copy(exemples.exemple_0_Barnea)
    
    # parms_bar["resol"] = "30"
    parms_bar["vel_min_gas"] = "0.01"
    parms_bar["vel_max_gas"] = "100.0"
    parms_bar["vel_min_liq"] = "0.001"
    parms_bar["vel_max_liq"] = "10.0"

    pat_bar = Patterns(parms_bar)
    car_bar = carlos(parms_bar)


    cases_test = ["00_degree", "05_degree", "10_degree", "90_degree"]
    # cases_test = ["00_degree"]
    
    for filecase in cases_test:
        # print(filecase)
        # parms_bar["incl"], parms_bar["ext_point_vg"], parms_bar["ext_point_vl"], flow_pattern = car_bar.load_data_exp(filecase+".xlsx")
        parms_bar["incl"], _, _, _ = car_bar.load_data_exp(filecase+".xlsx")

        # parms_bar["fenomenol"] = "Shoham 2005"
        parms_bar["fenomenol"] = "Barnea 1986"
        parms_bar["resol"] = "250"

        phe_bar = Phenom(parms_bar)
        car_bar = carlos(parms_bar)
        
        # print(phe_bar.Barnea1986_function_point(0.5,0.5))
        
        phe_bar.PhenomPatternsMap()
        
        # phe_data_bar = PhenomDataDriven(parms_bar)
        # phe_data_bar.PhenomDataDrivenPatternsMap()
        # # phe_data_bar.RandomForestMapOptmizeParms()

        # phe_hyb_bar = PhenomDataDrivenHybrid(parms_bar, phe_bar.pattern_map)
        # phe_hyb_bar.PhenomDataDrivenHybridPatternsMap()
        # # phe_hyb_bar.RandomForestHybridMapOptmizeParms()

        fig1, ax1 = plt.subplots(1, 1, figsize=(9, 9))
        phe_bar.plot_patterns(fig1, ax=ax1, titlefigure=["Phenomenological Model: "], loc="lower center", framealpha=0.4, fontsizeleg=13)
        # car_bar.plotcarlosnew(phe_bar.pattern_map, flow_pattern, fig1, ax=ax1, titlefigure=["Phenomenological Model: "], loc="lower center", framealpha=0.4, fontsizeleg=13)
        # plt.savefig(filecase+"_Shoham_Froude_Compare_2.png")
        plt.savefig(filecase+"_Barnea_Froude_Compare_2.png")
        
        # fig2, ax2 = plt.subplots(1, 1, figsize=(9, 9))
        # car_bar.plotcarlosnew(phe_data_bar.pattern_map, flow_pattern, fig2, ax=ax2, titlefigure=["Phenomenological Model: "], loc="lower center", framealpha=0.4, fontsizeleg=13)
        # plt.savefig(filecase+"_Barnea_Data_Driven_New_best_par.png")

        # fig3, ax3 = plt.subplots(1, 1, figsize=(9, 9))
        # car_bar.plotcarlosnew(phe_hyb_bar.pattern_map, flow_pattern, fig3, ax=ax3, titlefigure=["Phenomenological Model: "], loc="lower center", framealpha=0.4, fontsizeleg=13)
        # plt.savefig(filecase+"_Barnea_Data_Hybr_New_best_par.png")

        # plt.show()


