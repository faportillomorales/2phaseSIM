#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Set 09 20:23:09 2023

@author: LEMI Laboratory
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines
import h5py

import textwrap

from flowtechlib import dicflowpattern as dfp
import importlib.resources as pkg_resources
from . import Data

class Patterns:
    
    ### okokok
    def __init__(self,parms):
        """
        Description
        -------
        This is the constructor method for the class, which initializes the required parameters for 
        two-phase flow analysis, based on a set of input parameters (`parms`). The function sets up 
        the properties for the liquid and gas phases, including their velocities, viscosities, 
        densities, and surface tension, as well as other flow-related properties such as pipe diameter 
        and inclination. It also prepares arrays for velocity grids and initializes variables used 
        for further flow pattern calculations.

        Inputs
        ----------
        parms : dict -> A dictionary containing the input parameters required for 
                        initializing the class. The expected keys in the dictionary are:
            - "fluid1"       : str -> Name of the liquid phase fluid.
            - "vel_min_liq"  : float -> Minimum superficial velocity of the liquid phase [m/s].
            - "vel_max_liq"  : float -> Maximum superficial velocity of the liquid phase [m/s].
            - "visc_liq"     : float -> Viscosity of the liquid phase [Pa.s].
            - "dens_liq"     : float -> Density of the liquid phase [kg/m^3].
            - "fluid2"       : str -> Name of the gas phase fluid.
            - "vel_min_gas"  : float -> Minimum superficial velocity of the gas phase [m/s].
            - "vel_max_gas"  : float -> Maximum superficial velocity of the gas phase [m/s].
            - "visc_gas"     : float -> Viscosity of the gas phase [Pa.s].
            - "dens_gas"     : float -> Density of the gas phase [kg/m^3].
            - "inte_tens"    : float -> Surface tension between the liquid and gas phases [N/m].
            - "diam"         : float -> Hydraulic diameter of the pipe [m].
            - "incl"         : float -> Inclination angle of the pipe [degrees].
            - "data_driven"  : str -> Whether to use a data-driven model.
            - "fenomenol"    : str -> The phenomenological model to use (e.g., "BARNEA1986").
            - "resol"        : int -> Resolution for velocity grid calculations.
            - "ext_point_vl" : list -> List of external liquid velocities for plotting.
            - "ext_point_vg" : list -> List of external gas velocities for plotting.

        Outputs
        -------
        None
        The constructor initializes various class attributes required for flow pattern analysis.
        
        Attributes Initialized
        -----------------------
        - self.fluid1 : str -> The name of the liquid phase.
        - self.vl_min : float -> Minimum superficial velocity of the liquid phase [m/s].
        - self.vl_max : float -> Maximum superficial velocity of the liquid phase [m/s].
        - self.mu_l : float -> Viscosity of the liquid phase [Pa.s].
        - self.rho_l : float -> Density of the liquid phase [kg/m^3].
        - self.fluid2 : str -> The name of the gas phase.
        - self.vg_min : float -> Minimum superficial velocity of the gas phase [m/s].
        - self.vg_max : float -> Maximum superficial velocity of the gas phase [m/s].
        - self.mu_g : float -> Viscosity of the gas phase [Pa.s].
        - self.rho_g : float -> Density of the gas phase [kg/m^3].
        - self.sigma : float -> Surface tension between the liquid and gas phases [N/m].
        - self.d : float -> Hydraulic diameter of the pipe [m].
        - self.alpha : float -> Inclination angle of the pipe in radians.
        - self.resol : int -> Resolution for velocity grids (number of points for interpolation).
        - self.V_g, self.V_l : np.ndarray -> Meshgrid arrays for gas and liquid velocities.
        - self.epd : float -> Relative roughness of the pipe.
        - self.pattern_plot : list -> An empty list initialized for storing pattern data.
        - self.flow_dic_barnea, self.flow_dic_shoham, self.flow_dic_trallero : dict -> Dictionaries 
        mapping flow pattern names to specific models.
        """
        self.fluid1 = parms["fluid1"]
        self.vl_min = float(parms["vel_min_liq"])
        self.vl_max = float(parms["vel_max_liq"])
        self.mu_l = float(parms["visc_liq"])
        self.rho_l = float(parms["dens_liq"])

        self.fluid2 = parms["fluid2"]
        self.vg_min = float(parms["vel_min_gas"])
        self.vg_max = float(parms["vel_max_gas"])
        self.mu_g = float(parms["visc_gas"])
        self.rho_g = float(parms["dens_gas"])
        
        self.sigma = float(parms["inte_tens"])
        self.d = float(parms["diam"])
        angle = float(parms["incl"])
        self.data_driven = parms["data_driven"]
        self.fenomenol = parms["fenomenol"]
        
        self.resol = int(parms["resol"])

        self.grav = 9.80665
        
        ### Rad
        self.alpha = (angle*np.pi)/180
        
        self.pattern_plot = []
        # Rugosidade relativa [ad] -> acrílico médio = 0.5*(1.5 + 0.7)*10**-6/d
        self.epd = 0.5*(1.5 + 0.7)*10**-6/self.d

        self.vel_l = np.logspace(np.log10(self.vl_max), np.log10(self.vl_min), num=self.resol)
        self.vel_g = np.logspace(np.log10(self.vg_max), np.log10(self.vg_min), num=self.resol)

        self.V_g, self.V_l = np.meshgrid(self.vel_g, self.vel_l)
        self.ext_point_vl = parms["ext_point_vl"]
        self.ext_point_vg = parms["ext_point_vg"]
        
        self.flow_dic_barnea = dfp.flow_dic_barnea
        self.flow_dic_shoham = dfp.flow_dic_shoham
        self.flow_dic_trallero = dfp.flow_dic_trallero
        
        self.pattern_map = []

    ### okokok
    def info(self):
        """
        Description
        -------
        This function displays the current fluid and system properties, including the velocities, 
        viscosities, densities of the liquid and gas phases, interfacial tension, pipe diameter, 
        and inclination. It also shows the selected data-driven and phenomenological models, as well as the resolution used for calculations.

        Inputs
        ----------
        None

        Outputs
        -------
        None
        The function prints the following details to the console with improved formatting for readability.
        """

        # Título geral
        print("\n" + "="*40)
        print("          PROPERTIES     ")
        print("="*40 + "\n")
        
        # Propriedades do fluido 1 (líquido)
        print("Fluid 1 (Liquid):")
        print(f"  Name: {self.fluid1}")
        print(f"  Velocity [m/s]: Min = {self.vl_min:.3f} / Max = {self.vl_max:.3f}")
        print(f"  Viscosity [Pa.s]: {self.mu_l:.5f}")
        print(f"  Density [kg/m^3]: {self.rho_l:.2f}")
        
        print("\n" + "-"*40 + "\n")

        # Propriedades do fluido 2 (gás)
        print("Fluid 2 (Gas):")
        print(f"  Name: {self.fluid2}")
        print(f"  Velocity [m/s]: Min = {self.vg_min:.3f} / Max = {self.vg_max:.3f}")
        print(f"  Viscosity [Pa.s]: {self.mu_g:.5f}")
        print(f"  Density [kg/m^3]: {self.rho_g:.2f}")

        print("\n" + "-"*40 + "\n")

        # Tensão superficial
        print("Interfacial Tension:")
        print(f"  Sigma [N/m]: {self.sigma:.5f}")

        print("\n" + "-"*40 + "\n")

        # Propriedades do tubo
        print("Pipe Properties:")
        print(f"  Diameter [m]: {self.d:.4f}")
        print(f"  Inclination [degrees]: {self.alpha * 180 / np.pi:.2f}")

        print("\n" + "-"*40 + "\n")

        # Modelo baseado em dados e fenomenológico
        print("Models:")
        print(f"  Data Driven Model: {self.data_driven}")
        print(f"  Phenomenological Model: {self.fenomenol}")

        print("\n" + "-"*40 + "\n")

        # Resolução
        print("Simulation Resolution:")
        print(f"  Grid Resolution: {self.resol}")

        print("\n" + "="*40 + "\n")

    def plot_patterns(self, fig, ax, titlefigure, loc="lower center", framealpha=0.4, fontsizeleg=12):
        """
        Description
        -------
        
        Inputs
        ----------
        inp :       float -> 
        
        Outputs
        -------
        f :         float -> 
        """
        self.pattern_plot = np.transpose(np.reshape(self.pattern_map,(self.resol,self.resol)))
        level_aux = np.copy(np.reshape(self.pattern_plot,self.resol*self.resol))
        level_aux = set(level_aux)
        
        col_aux = []
        lev_aux = [-1]
        self.legclass = []
        if (self.fenomenol.upper().replace(" ","") == "BARNEA1986"):
            for pos, ele in enumerate(level_aux):
                lev_aux.append(ele)
                col_aux.append(dfp.col_dict_barnea[ele])
                if ele in dfp.flow_dic_barnea.values():
                    for key, value in dfp.flow_dic_barnea.items():
                        if value == ele:
                            self.legclass.append(key)
        elif (self.fenomenol.upper().replace(" ","") == "SHOHAM2005"):
            for pos, ele in enumerate(level_aux):
                lev_aux.append(ele)
                col_aux.append(dfp.col_dict_shoham[ele])
                if ele in dfp.flow_dic_shoham.values():
                    for key, value in dfp.flow_dic_shoham.items():
                        if value == ele:
                            self.legclass.append(key)
        elif (self.fenomenol.upper().replace(" ","") == "TRALLERO1995"):
            for pos, ele in enumerate(level_aux):
                lev_aux.append(ele)
                col_aux.append(dfp.col_dict_trallero[ele])
                if ele in dfp.flow_dic_trallero.values():
                    for key, value in dfp.flow_dic_trallero.items():
                        if value == ele:
                            self.legclass.append(key)
        else:
            print('Method not implemented!')
            return 0

        if ((len(self.ext_point_vl) > 0 ) and (len(self.ext_point_vg) > 0 ) and (len(self.ext_point_vl)==len(self.ext_point_vg))):
            rotulos = [f"$P_{ i }$" for i in range(len(self.ext_point_vl))]
            fig.tight_layout()
            fig.patch.set_facecolor((0.157, 0.388, 0.525, 1))
            legenda = fig.legend('')
            ax.clear()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.scatter(self.ext_point_vg, self.ext_point_vl, label='', color='blue', marker='o')
            for i, rotulo in enumerate(rotulos):
                ax.annotate(rotulo, (self.ext_point_vg[i], self.ext_point_vl[i]), textcoords="offset points",  xytext=(0,14), ha='center')
        elif ((len(self.ext_point_vl) == 0) and (len(self.ext_point_vg) == 0)):
            print('')
        else:
            print('The extra velocities of the gas and the liquid must have the same dimensions!')

        grafico = ax.contourf(self.V_g, self.V_l, self.pattern_plot, colors=col_aux, levels=lev_aux, alpha = 0.6, extend = 'max')

        proxy = [plt.Rectangle((self.vg_min,self.vl_min),1,1,fc = pc.get_facecolor()[0], edgecolor = 'black') for pc in grafico.collections]
        
        legenda.remove()
        
        legenda = ax.legend(proxy, self.legclass, loc=loc, ncol=lev_aux[-1]+1, framealpha=framealpha, fontsize=fontsizeleg)
        
        ax.grid(alpha=0.15, color='black')
        ax.set_title(titlefigure[0]+self.fenomenol, fontsize=fontsizeleg)
        ax.set_xlabel('Superficial Gas Velocity [m/s]', labelpad = 1, fontsize=fontsizeleg)
        ax.set_ylabel('Superficial Liquid Velocity [m/s]', fontsize=fontsizeleg)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    def load_data_compare(self):
        with pkg_resources.path(Data,'Flow_Database_Carlos.xlsx') as file:
            files_load = pd.read_excel(file)
        degrees = files_load.loc[:, ["Pipe Angle of Inclination [DEGREE]"]].values
        degree = str(degrees[0][0])
        self.ext_point_vl = files_load.loc[:, ["Superficial oil Velocity [m/s]"]].values
        self.ext_point_vg = files_load.loc[:, ["Superficial Gas Velocity [m/s]"]].values
        flow_pattern = files_load.loc[:, ["Flow Pattern [-]"]].values
        return flow_pattern

    def plot_patterns_compare(self, fig, ax, titlefigure, loc="lower center", framealpha=0.4, fontsizeleg=12):
        
        # Configurar o pattern_map
        self.pattern_plot = np.transpose(np.reshape(self.pattern_map, (self.resol, self.resol)))
        level_aux = np.copy(np.reshape(self.pattern_plot, self.resol * self.resol))
        level_aux = set(level_aux)

        flow_pattern_comp = self.load_data_compare()
        flow_pattern_comp_aux = []
        for pos, ele in enumerate(flow_pattern_comp):
            flow_pattern_comp_aux.append(ele[0])

        flow_pattern_comp_aux_2 = set(flow_pattern_comp_aux)
        
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
        for ele in flow_pattern_comp_aux_2:
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
                ax.scatter(self.ext_point_vg[i], self.ext_point_vl[i], color='blue', marker=marker_dict[flow_pattern_comp[i][0]])
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
            for pattern in flow_pattern_comp_aux_2
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

    def plot_patterns_2(self, fig, ax, titlefigure):
        
        self.pattern_plot = np.transpose(np.reshape(self.pattern_map,(self.resol,self.resol)))
        level_aux = np.copy(np.reshape(self.pattern_plot,self.resol*self.resol))
        level_aux = set(level_aux)
        
        if (self.fenomenol.upper().replace(" ","") == "BARNEA1986"):
            col_dict = dfp.col_dict_barnea
            labels = []
            for key, value in dfp.flow_dic_barnea.items():
                labels.append(key)
        elif (self.fenomenol.upper().replace(" ","") == "SHOHAM2005"):
            col_dict = dfp.col_dict_shoham
            labels = []
            for key, value in dfp.flow_dic_shoham.items():
                labels.append(key)
        else:
            print('Method not implemented!')
            return 0

        # define-se um colormap a partir do dicionário de cores
        cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
        
        # Lista com os labels associados à cada cor
        labels = np.array(labels)
        
        len_lab = len(labels)
        
        # preparar normalizador
        ## faixas para o normalizador - > -+0.5 entorno do valor
        norm_bins = np.sort([*col_dict.keys()]) + 0.5
        norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
        
        ## normaliza e formata
        norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
        fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])        
                
        ax.set_xscale('log')
        ax.set_yscale('log')

        cs = ax.pcolor(self.V_g, self.V_l, self.pattern_plot, cmap = cm, norm = norm, shading='auto')

        diff = norm_bins[1:] - norm_bins[:-1]
        tickz = norm_bins[:-1] + diff / 2
        cb = fig.colorbar(cs, format=fmt, ticks=tickz)
        
        # fig.suptitle('Mapa Padrão de Escoamento: '+r'$\theta$'+' = '+'{:.1f}'.format(self.alpha*180/np.pi)+'º')
        ax.set_title(titlefigure[0]+self.fenomenol, fontsize=12)
        ax.set_xlabel('Superficial Gas Velocity [m/s]', labelpad = 1, fontsize=9)
        ax.set_ylabel('Superficial Liquid Velocity [m/s]', fontsize=9)
        ax.set_xlim([self.vg_min,self.vg_max])
        ax.set_ylim([self.vl_min,self.vl_max])