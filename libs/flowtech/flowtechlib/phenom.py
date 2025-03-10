#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Set 09 20:23:09 2023

@author: LEMI Laboratory
"""
import os
import math
import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import fsolve, bisect, brentq
from sklearn import preprocessing, model_selection, preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import neural_network, metrics
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import SVC

#from sklearn.neural_network import MLPRegressor
#from sklearn.ensemble import GradientBoostingRegressor

from tqdm import tqdm
import copy
from copy import deepcopy

import warnings
warnings.filterwarnings('ignore')

import importlib.resources as pkg_resources
from . import Data
from .patterns import *

class Phenom(Patterns):

    def __init__(self,parms):
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
        super().__init__(parms)
    
    ### okokok
    def dimensionless(self):
        """
        Description
        -------
        This function calculates various dimensionless parameters for gas and liquid phases 
        based on a given height-to-diameter ratio (`h_adm`). The parameters include areas, 
        perimeters, velocities, and diameters for both the liquid and gas phases, in addition 
        to the interfacial perimeter. These parameters are essential for two-phase flow analysis.

        Inputs
        ----------
        self.h_adm :   float -> The dimensionless height ratio, typically the height of the 
                                liquid phase normalized by the diameter of the pipe.

        Outputs
        -------
        self.A_adm_l : float -> Dimensionless area of the liquid phase.
        self.A_adm_g : float -> Dimensionless area of the gas phase.
        self.S_adm_l : float -> Dimensionless wetted perimeter of the liquid phase.
        self.S_adm_g : float -> Dimensionless wetted perimeter of the gas phase.
        self.S_adm_i : float -> Dimensionless interfacial perimeter between the gas and liquid phases.
        self.v_adm_l : float -> Dimensionless velocity of the liquid phase, defined as the ratio 
                                of total area to liquid area.
        self.v_adm_g : float -> Dimensionless velocity of the gas phase, defined as the ratio 
                                of total area to gas area.
        self.D_adm_l : float -> Dimensionless hydraulic diameter for the liquid phase.
        self.D_adm_g : float -> Dimensionless hydraulic diameter for the gas phase.
        self.A_adm :   float -> Total cross-sectional area of the system, normalized (constant).
        """

        hdm1 = 2*self.h_adm-1
        self.A_adm_l = 0.25*(np.pi-np.arccos(hdm1) + hdm1*np.sqrt(1 - hdm1**2)) 
        self.A_adm_g = 0.25*(np.arccos(hdm1) - hdm1*np.sqrt(1 - hdm1**2))
        self.S_adm_l = np.pi-np.arccos(hdm1)
        self.S_adm_g = np.arccos(hdm1)
        self.S_adm_i = np.sqrt(1 - hdm1**2)
        self.A_adm = np.pi/4 
        self.v_adm_l = self.A_adm/self.A_adm_l
        self.v_adm_g = self.A_adm/self.A_adm_g
        self.D_adm_l = 4*self.A_adm_l/self.S_adm_l
        self.D_adm_g = 4*self.A_adm_g/(self.S_adm_g + self.S_adm_i)
    
    ### okokok
    def funcao_h_adm(self, xh_adm, vs_l, vs_g):
        """
        Description
        -------
        This function calculates various dimensionless parameters for gas and liquid phases based 
        on a given height-to-diameter ratio (`xh_adm`), as well as the velocities of the gas and 
        liquid phases (`vs_g`, `vs_l`). It computes dimensionless areas, perimeters, velocities, 
        hydraulic diameters, Reynolds numbers, and shear stress coefficients. The final result is 
        a calculated function (`func`) involving these parameters, which is essential for analyzing 
        two-phase flow.
        
        Inputs
        ----------
        xh_adm : float -> Dimensionless height-to-diameter ratio, typically the height of the 
                          liquid phase normalized by the diameter of the pipe.
        vs_l :   float -> Superficial velocity of the liquid phase (m/s).
        vs_g :   float -> Superficial velocity of the gas phase (m/s).
        
        Outputs
        -------
        func :   float -> A dimensionless result of the calculations, which is used to characterize 
                          two-phase flow conditions based on velocity ratios, Reynolds numbers, 
                          and other dimensionless parameters.
        """
        ### Adimensionais variaveis
        hdm1 = 2*xh_adm-1
        xA_adm_l = 0.25*(np.pi-np.arccos(hdm1) + hdm1*np.sqrt(1 - hdm1**2)) 
        xA_adm_g = 0.25*(np.arccos(hdm1) - hdm1*np.sqrt(1 - hdm1**2))
        xS_adm_l = np.pi-np.arccos(hdm1)
        xS_adm_g = np.arccos(hdm1)
        xS_adm_i = np.sqrt(1 - hdm1**2)
        xA_adm = np.pi/4 
        xv_adm_l = xA_adm/xA_adm_l
        xv_adm_g = xA_adm/xA_adm_g
        xD_adm_l = 4*xA_adm_l/xS_adm_l
        xD_adm_g = 4*xA_adm_g/(xS_adm_g + xS_adm_i)

        # Reynolds Numbers
        Re_sl = self.rho_l*vs_l*self.d/self.mu_l
        Re_sg = self.rho_g*vs_g*self.d/self.mu_g
        
        # Dimensionless Velocities
        v_g = xv_adm_g*vs_g
        v_l = xv_adm_l*vs_l

        d_l = self.d*4*xA_adm_l/xS_adm_l
        d_g = self.d*4*xA_adm_g/(xS_adm_g + xS_adm_i)
        
        # Calculation of dimensionless Reynolds numbers
        Re_l = self.rho_l * v_l * d_l / self.mu_l
        Re_g = self.rho_g * v_g * d_g / self.mu_g
        
        # Calculation of shear stress coefficients
        if Re_l > 2000:
            C_l = 0.046
            n = 0.2
        else:
            C_l = 16
            n = 1.0
            
        if Re_g > 2000:
            C_g = 0.046
            m = 0.2
        else:
            C_g = 16
            m = 1.0
        
        # Cálculo das grandezas X^2 e Y
        X2 = (C_l / C_g) * (Re_sg ** m / Re_sl ** n) * (self.rho_l / \
                    self.rho_g) * (vs_l / vs_g) ** 2

        Y = (self.rho_l - self.rho_g) * self.grav * np.sin(self.alpha) / (4 * C_g / \
            self.d * (vs_g * self.d * self.rho_g / self.mu_g) ** (-m) * \
            self.rho_g * vs_g ** 2 / 2)
        
        # Cálculo da função final
        func = X2 * (((xv_adm_l * xD_adm_l) ** (-n)) * \
                    (xv_adm_l ** 2) * xS_adm_l / xA_adm_l) - \
                    ((xv_adm_g * xD_adm_g) ** (-m)) * (xv_adm_g ** 2) * \
                    (xS_adm_g / xA_adm_g + xS_adm_i / xA_adm_l + \
                    xS_adm_i / xA_adm_g) + 4 * Y
        
        return func
    
    ### okokok
    def root_h_adm(self, vs_l : float, vs_g  : float) -> float:
        """
        Description
        -------
        This function uses the bisection method to compute the root of the function `funcao_h_adm`,
        which calculates the dimensionless height-to-diameter ratio (`h_adm`) for two-phase flow. 
        It takes the superficial velocities of the gas and liquid phases as inputs and iteratively 
        finds the value of `h_adm` that satisfies the conditions defined in `funcao_h_adm`. 
        The function employs precise tolerances and iteration limits for convergence.

        Inputs
        ----------
        vs_l :       float -> Superficial velocity (or volumetric flow rate per unit area) of the liquid phase [m/s].
        vs_g :       float -> Superficial velocity (or volumetric flow rate per unit area) of the gas phase [m/s].

        Outputs
        -------
        self.h_adm : float -> The dimensionless height-to-diameter ratio (`h_adm`), typically the height of the 
                              liquid phase normalized by the diameter of the pipe, calculated using the bisection method.
        
        Method Details
        ----------------
        - `bisect`: The bisection method is used to find the root of the function `funcao_h_adm`.
        - The search is conducted between very small bounds [0.0000001, 0.9999999].
        - `xtol`: Absolute tolerance for convergence, set to `2e-12`.
        - `rtol`: Relative tolerance for convergence, set to `8.88e-16`.
        - `maxiter`: Maximum number of iterations allowed for the bisection method, set to `2000`.
        """
        self.h_adm = bisect(partial(self.funcao_h_adm, vs_g=vs_g, vs_l=vs_l), 0.0000001, 0.9999999, xtol=2e-12, rtol=8.881784197001252e-16, maxiter=2000)
        
        return self.h_adm
    
    ### okokok
    def Barnea1986_function_point(self, vs_l : float, vs_g : float) -> str:
        """
        Description
        -------
        The function classifies the type of flow pattern according to the properties using Barnea (1986) 

        Inputs
        ----------
        vs_l :      float -> superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> mass density of the liquid phase [kg/m^3]
        rho_g :     float -> mass density of the gas phase [kg/m^3]
        mu_l :      float -> viscosity of the liquid phase [Pa.s]
        mu_g :      float -> viscosity of the gas phase [Pa.s]
        sigma :     float -> surface tension between the liquid and gas phases [N/m]
        d :         float -> equivalent hydraulic diameter of the pipe [m]
        alpha :     float -> inclination angle of the pipe [rad]
        g :         float -> acceleration due to gravity [m/s^2]

        Outputs
        -------
        pattern :   str -> flow pattern (Annular, Dispersed, Intermittent, Stratified)
        """
        
        A = np.pi*(self.d**2)/4 #area da seção transversal do duto
        P_total = 0
        self.root_h_adm(vs_l, vs_g)
        self.dimensionless()
        
        # Criteria obtained following Barnea 1986
        # Transition from Stratified to Non-Stratified Pattern
        v_g = self.v_adm_g*vs_g
        v_l = self.v_adm_l*vs_l
        
        # Froude Number
        Fro = np.sqrt(self.rho_g/(self.rho_l-self.rho_g))*(vs_g)/ \
            (np.sqrt(self.d*self.grav*np.cos(self.alpha)))
        
        # Fro = np.sqrt(1.0/(self.rho_l-self.rho_g))*(vs_g)/ \
        #     (np.sqrt(self.d*self.grav*np.cos(self.alpha)))

        # Fro = 0.5 * (vs_g + vs_l) / \
        #     (np.sqrt(self.d*self.grav*np.cos(self.alpha)))
        
        c2 = 1 - self.h_adm
        
        # The transition line from stratified to non-stratified flow
        Crit_strat = ((Fro**2) * (c2**(-2)) * self.v_adm_g**2 * self.S_adm_i / self.A_adm_g)

        if np.abs(self.alpha) == np.pi/2: 
            Crit_strat = 10
        
        # Transição de bolhas dispersas
        # Velocidade mistura
        vs_m = vs_g + vs_l 
        fraction = vs_g / vs_m #fração volumetrica gas
        f_m = 0.046 * np.power((vs_m * self.d * self.rho_l / self.mu_l),-0.2)
        d_cd = 2*np.power((0.4*self.sigma / ((self.rho_l - self.rho_g)*self.grav)),0.5)
        # Programa do Davi está como + as densidades
        d_cb = 3/8*(self.rho_l / (self.rho_l-self.rho_g))*f_m*vs_m**2/(self.grav*np.cos(self.alpha))
        d_c = min(d_cd,d_cb)
            
        aux1 = (0.725 + 4.15*fraction**0.5)
        aux2 = (self.sigma / self.rho_l)**0.6
        aux3 = (2*f_m/self.d * vs_m**3)**(-0.4)
        Crit_disp_1 = aux1*aux2*aux3
        # Linha transição caso a fração de gas >0.52
        Crit_disp_2 = vs_g*((1-0.52)/0.52)

        # Criterio 1 para existência de bolhas, mudado intencionalmente de 19 para 18 para atender o range de 2 inc vertical
        Crit_bubble_1 = 18*((self.rho_l - self.rho_g) * self.sigma / (self.rho_l**2*self.grav))**0.5
        v_o = 1.53*(self.grav*((self.rho_l-self.rho_g)*self.sigma/self.rho_l**2))**0.25
        #criterio 2 para existência de bolhas, CL = 1.1 e gamma 1.5 (existe erro na equação da barnea 1987, verificar dedução no artigo 1985)
        Crit_bubble_2 = 3/4*np.pi*np.cos(np.pi/4)*v_o**2/self.grav*1.2*1.5**2/self.d*self.rho_l/(self.rho_l-self.rho_g)
        # Linha de transição para bolhas
        Crit_bubble_3 = 3*vs_g-0.75*v_o*np.sin(self.alpha)
            
        #Transição do padrão anular
        Re_sl = self.rho_l*vs_l*self.d/self.mu_l
        Re_sg = self.rho_g*vs_g*self.d/self.mu_g
        d_l = self.d*4*self.A_adm_l/self.S_adm_l
        d_g = self.d*4*self.A_adm_g/(self.S_adm_g + self.S_adm_i)
        
        # Calculation of dimensionless Reynolds numbers
        Re_l = self.rho_l * v_l * d_l / self.mu_l
        Re_g = self.rho_g * v_g * d_g / self.mu_g

        # Caso seja turbulento ou laminar fornece os coeficientes para tensão cisalhamento
        if Re_l > 2000:
            C_l = 0.046
            n = 0.2
        else:
            C_l = 16
            n = 1.0
        
        #caso seja turbulento ou laminar fornece os coeficientes para tensão cisalhamento
        if Re_g > 2000:
            C_g = 0.046
            m = 0.2
        else:
            C_g = 16
            m = 1.0
        
        X2 = (C_l/C_g)*(Re_sg**m/Re_sl**n)*(self.rho_l/self.rho_g)*(vs_l/vs_g)**2
        Y = (self.rho_l-self.rho_g)*self.grav*np.sin(self.alpha)/(4*C_g/self.d*(vs_g*self.d*self.rho_g/self.mu_g)**(-m)*self.rho_g*vs_g**2/2)    
        
        f_l = C_l*(self.rho_l*d_l*v_l/self.mu_l)**(-n)
        
        def f(frac_l):
            return - Y - (np.power(frac_l, -3)) * X2 + (1 + 75 * frac_l) / ((np.power(1 - frac_l, 2.5) * frac_l))
        
        # try:
        #     frac_l = bisect(f, 0.0001, 0.99999, args=(), xtol=2e-12, rtol=8.881784197001252e-16, maxiter=2000)
        # except ValueError:
        #     frac_l = 1
        
        frac_l = bisect(f, 0.0001, 0.99999, args=(), xtol=2e-12, rtol=8.881784197001252e-16, maxiter=2000)

        #Criterio transição anular para altas inclinações descendentes
        Crit_annular_1 = self.grav*self.d*c2*np.cos(self.alpha)/f_l
        Crit_annular_2 = X2*(2-3/2*frac_l)/(frac_l**3*(1-3/2*frac_l))
        Crit_annular_3 = 0.24

        #Criterios de transição
        if ((fraction < 0.52 and d_c > Crit_disp_1) or (fraction >= 0.52 and vs_l > Crit_disp_2)): #dispersed bubbles
            # self.pattern_map.append("Dispersed")
            self.pattern_map.append(self.flow_dic_barnea["Dispersed"])
            pattern = "Dispersed"
        else:
            if ((Crit_strat < 1) and (v_l**2 < Crit_annular_1)): # stratified-nonstratified transition
                # self.pattern_map.append("Stratified")
                self.pattern_map.append(self.flow_dic_barnea["Stratified"])
                pattern = "Stratified"
            else:  #annular-intermittent transition  
                if (Y < Crit_annular_2) and (frac_l < Crit_annular_3):
                    # self.pattern_map.append("Annular")
                    self.pattern_map.append(self.flow_dic_barnea["Annular"])
                    pattern = "Annular"
                else: #bubbles-intermittent transition
                    if (self.d > Crit_bubble_1 and vs_l > Crit_bubble_3 and (np.cos(self.alpha)/(np.sin(self.alpha)**2) < Crit_bubble_2)):
                        # self.pattern_map.append("Dispersed")
                        self.pattern_map.append(self.flow_dic_barnea["Dispersed"])
                        pattern = "Dispersed"
                    else:    
                        # self.pattern_map.append("Intermittent")
                        self.pattern_map.append(self.flow_dic_barnea["Intermittent"])
                        pattern = "Intermittent"

        P_total_new = self.rho_l*v_l**2/2 + self.rho_g*v_g**2/2+101325*15

        if P_total_new > P_total:
            P_total = P_total_new
            vel_g_p = vs_g
            vel_l_p = vs_l
        
        return pattern

    ### okokok
    def Barnea1986Map(self) -> list:
        """
        Description
        -------
        Function that implements the decision tree for the unified model flow pattern (upward and downward) as proposed by Barnea (1986)

        Inputs
        ----------
        vs_l :      float -> superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> mass density of the liquid phase [kg/m^3]
        rho_g :     float -> mass density of the gas phase [kg/m^3]
        mu_l :      float -> viscosity of the liquid phase [Pa.s]
        mu_g :      float -> viscosity of the gas phase [Pa.s]
        sigma :     float -> surface tension between the liquid and gas phases [N/m]
        d :         float -> equivalent hydraulic diameter of the pipe [m]
        alpha :     float -> inclination angle of the pipe [rad]
        g :         float -> acceleration due to gravity [m/s^2]

        Outputs
        -------
        self.pattern_map :   list -> flow pattern (Annular : 0, Dispersed : 1, Intermittent : 2, Stratified : 3)
        """
        
        self.pattern_map = []
        ntotal = len(self.vel_g)*len(self.vel_l)
        progress_bar = tqdm(total=ntotal, desc="Barnea1986 Progress")
        for vs_g in self.vel_g:
            for vs_l in self.vel_l:
                progress_bar.update(1)
                self.Barnea1986_function_point(vs_l, vs_g)

        progress_bar.close()
    
    ### okokok
    def f_blasius(self,Re : float) -> float:
        """
        Description
        -------
        Function that calculates the Fanning friction factor for smooth tubes

        Inputs
        ----------
        rho :       float -> Specific mass of the fluid [kg/m^3]
        v :         float -> Fluid speed [m/s]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        mu :        float -> Fluid viscosity [Pa.s]

        Outputs
        -------
        f :         float -> Fanning friction factor
        """
        
        if (Re < 2000):
            a = 16
            n = 1
        elif (Re < 100000):
            a = 0.079
            n = 0.25
        else:
            a = 0.046
            n = 0.2
        
        # fator de fricção de fanning
        f = a*(Re**(-n))
        
        return f

    ### okokok
    def f_friccao_interface_xiao1990(self, hlt : float, ULt : float, UGt : float, DGt : float, vs_l : float, vs_g : float) -> float:
        """
        Description
        -------
        Function that calculates the interface friction factor in the stratified pattern - Shoham (2005) pages 79 to 82 - Xiao et al. (1990)
        
        Inputs
        ----------
        hlt :       float -> dimensionless liquid height
        ULt:        float -> ratio between in situ and surface velocity of the liquid phase (=1/alpha1) dimensionless
        UGt:        float -> ratio between in situ and surface velocity of the gas phase (=1/alpha2) dimensionless
        DGt :       float -> equivalent hydraulic diameter of the dimensionless gas phase
        vs_l :      float -> superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> mass density of the liquid phase [kg/m^3]
        rho_g :     float -> mass density of the gas phase [kg/m^3]
        mu_l :      float -> viscosity of the liquid phase [Pa.s]
        mu_g :      float -> viscosity of the gas phase [Pa.s]
        sigma :     float -> surface tension between the liquid and gas phases [N/m]
        d :         float -> equivalent hydraulic diameter of the pipe [m]
        epd :       float -> relative roughness of the pipe [-]
        alpha :     float -> inclination angle of the pipe [rad]
        g :         float -> acceleration due to gravity [m/s^2]

        Outputs
        -------
        f_i :       float -> Interface fanning friction factor
        """
    
        Re_2 = (self.rho_g*abs(UGt*vs_g)*(DGt*self.d))/self.mu_g   # número de Reynolds da fase gasosa
        f_2 = self.fatoratrito(Re_2,self.epd/DGt)    # fator de atrito da fase gasosa
        
        if (self.d <= 0.127):        # tubos de pequenos diâmetros (d <= 0.127] [m])
            
            # OBS.: prevê um critério para testar se é ondulado que só depende da pressão (sem tenão interfacial, massas específicas e escorregamento) - nada a ver com KH, também difícil de acreditar!
            #vs_g_ond = 5*(101.325/p)**0.5 # p-> pressão [Pa]
            
            # "efeito vento" ou ondas induzidas pelo cisalhamento provocado pelo gas - Taitel & Dukler (1976) - Barnea (1987) - Ref.: Shoham (2005) Eq. 3.53
            s = 0.01    # sheltering coefficient
            vs_g_ond = (1/UGt)*np.sqrt(4*self.mu_l*(self.rho_l-self.rho_g)*self.grav*np.cos(self.alpha)/(self.rho_l*s*self.rho_g*(ULt*vs_l))) # velocidade sup do gás de transição para ondulado

            if vs_g < vs_g_ond:   # padrão estratificado liso
                f_i = f_2
                
            else:               # padrão estratificado ondulado
                f_i = f_2*(1 + 15*(hlt**0.5)*(vs_g/vs_g_ond - 1))
                
                if (f_i < 0): f_i = f_2 # só uma prevenção anti-bug caso a rotina para calcular f_i falhe -> daí Taitel & Dukler (1976) padrão
            
        else:                   # tubos de grandes diâmetros [d >= 0.127]
            def rug_abs_interface(e_i): # a rugosidade absoluta da interface
                
                Nwe = self.rho_g*((ULt*vs_l)**2)*e_i/self.sigma     # número de weber
                Nmu = (self.mu_l**2)/(self.rho_l*self.sigma*e_i)        # número de viscosidade
                
                if (Nwe*Nmu <= 0.005):
                    fei = e_i - 34*self.sigma/(self.rho_g*((ULt*vs_l)**2))
                    
                else:
                    fei = e_i - 170*self.sigma*((Nwe*Nmu)**0.3)/(self.rho_g*((ULt*vs_l)**2))
                    
                return fei
            
            # Resolve-se a rugosidade absoluta da interface -> e_i [min(self.epd*self.d,0.25*(hlt)**0.5), max(self.epd*self.d,0.25*(hlt)**0.5)] é a raiz da equação rug_abs_interface(e_i)
            # OBS.1: como busca-se a solução em um intervalo fechado, o método de Brent é o melhor, pois combina biseção com secante, mas função deve ter sinais opostos nos limites do intervalo de busca
            
            # intervalo de busca
            # a rugosidade da interface deve ficar entre a da tubulação e uma fração da altura de líquido adm
            e_i_min = min(self.epd*self.d,0.25*((hlt)**0.5)) # mínimo
            e_i_max = max(self.epd*self.d,0.25*((hlt)**0.5)) # máximo
                    
            # se os sinais dos extremos do intervalo forem diferentes, chama o método de brent
            if (np.sign(rug_abs_interface(e_i_min)) != np.sign(rug_abs_interface(e_i_max))):
                
                try: # como pode haver problemas de convergência, deve-se pensar numa excessão!
                    e_i = brentq(rug_abs_interface,e_i_min,e_i_max)
                    
                except: # se falhar
                    e_i = self.epd*self.d # se falhar, retorna a rugosidade do duto e daí fi=fg como Taitel & Dukler (1976)
                    
                    # print("MAPA - Aviso: o método de brent não convergiu: e_i - f_friccao_interface_xiao1990")
                    
            else: # caso os sinais dos extremos sejam iguais 
            
                # o fsolve faz uma busca mais ampla e pode extrapolar o intervalo, necessário testar as raízes! (Powell hybrid method)
                try: # como pode haver problemas de convergência, deve-se pensar numa excessão!
                    e_i_i = 0.5*(e_i_min + e_i_max) # estimativa inicial - > meio do intervalo 
                    e_i = fsolve(rug_abs_interface, e_i_i) #solução é um array -> sol[0]
                    
                    if ((e_i < e_i_min) or (e_i > e_i_max)): e_i = self.epd*self.d # se falhar, retorna a rugosidade do duto e daí fi=fg como Taitel & Dukler (1976)
                    
                    # print("MAPA - Aviso: fsolve - e_i - f_friccao_interface_xiao1990")
                    
                except: # se falhar
                    e_i = self.epd*self.d # se falhar, retorna a rugosidade do duto e daí fi=fg como Taitel & Dukler (1976)
                    
                    # print("MAPA - Aviso: o método Powell Hybrid (fsolve) não convergiu: e_i - f_friccao_interface_xiao1990")
            
            # calcula-se o fator de atrito da interface usando a rugosidade absoluta da interface com o número de Reynolds e diâmetro hid eq da fase gasosa
            f_i = self.fatoratrito(Re_2, e_i/(DGt*self.d))         # fator de atrito da interface
        
        return f_i

    ### okokok
    def SwameeJain1976(self,Re : float) -> float:
        """
        Description
        -------
        Function to calculate the Darcy friction factor for rough pipes 
        -> Swamee-Jain (1976): turbulent friction factor 
        -> Re 5e3 to 1e8 and epd 5e-2 to 1e-6

        Inputs
        ----------
        rho :       float -> fluid density [kg/m^3]
        v :         float -> Fluid velocity [m/s]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        epd :       float -> Relative roughness of the pipe [-]
        mu :        float -> Fluid viscosity [Pa.s]

        Outputs
        -------
        f :         float -> (!)Darcy friction factor
        """
        
        f = 0.25/(np.log10((self.epd/3.7) + (5.74/(Re**0.9)))**2)
        
        return f

    ### okokok
    def Colebrokwhite1937(self,Re : float, epdaux : float) -> float:
        """
        Description
        -------
        Function to calculate the Darcy friction factor for rough pipes 
        -> turbulent region Colebrook-White model (1937)

        Inputs
        ----------
        rho :       float -> fluid density [kg/m^3]
        v :         float -> Fluid velocity [m/s]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        epd :       float -> Relative roughness of the pipe [-]
        mu :        float -> Fluid viscosity [Pa.s]

        Outputs
        -------
        f :         float -> (!)Darcy friction factor
        """

        def eqf(f): #função com log e raiz onde se procura Real -> restrição de domínio
            return (-(1 / np.sqrt(f)) - 2 * np.log10((epdaux / 3.7) + (2.51 / (Re * np.sqrt(f)))))
    
        soli = self.SwameeJain1976(Re) # estimativa inicial para o solver (usar correlação para fator de atrito turbulento)

        #sol = scipy.optimize.root(eqf, soli) #solução é um obj -> sol.x[0]
        sol = fsolve(eqf, soli) #solução é um array -> sol[0]
        
        return sol[0]

    ### okokok
    def Swamee1993(self,Re : float, epdaux : float) -> float:
        """
        Description
        -------
        Function to calculate the Darcy friction factor for rough pipes 
        -> Swamee (1993): laminar and turbulent friction factor

        Inputs
        ----------
        rho :       float -> fluid density [kg/m^3]
        v :         float -> Fluid velocity [m/s]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        epd :       float -> Relative roughness of the pipe [-]
        mu :        float -> Fluid viscosity [Pa.s]

        Outputs
        -------
        f :         float -> (!)Darcy friction factor
        """
    
        f = ((64/Re)**8 + 9.5*(np.log((epdaux/3.7) + (5.74/(Re**0.9))) - (2500/Re)**6)**(-16))**0.125
        
        return f
    
    ### okokok
    def Churchill1977(self,Re : float, epdaux : float) -> float:
        """
        Description
        -------
        Function to calculate the Darcy friction factor for rough pipes 
        -> Churchill (1977): laminar and turbulent friction factor ->

        Inputs
        ----------
        rho :       float -> fluid density [kg/m^3]
        v :         float -> Fluid velocity [m/s]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        epd :       float -> Relative roughness of the pipe [-]
        mu :        float -> Fluid viscosity [Pa.s]

        Outputs
        -------
        f :         float -> (!)Darcy friction factor
        """
        
        Af = (2.457*np.log(1/((7/Re)**0.9+0.27*epdaux)))**16
            
        Bf = (37530/Re)**16
        
        f = 8*((((8/Re)**12)+(1/((Af+Bf)**(3/2))))**(1/12))
        
        return f
    
    ### okokok
    def fatoratrito(self,Re : float, epdaux : float) -> float:
        """
        Description
        -------
        Function to calculate the (!)Fanning friction factor for rough pipes

        Inputs
        ----------
        rho :       float -> fluid density [kg/m^3]
        v :         float -> Fluid velocity [m/s]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        epd :       float -> Relative roughness of the pipe [-]
        mu :        float -> Fluid viscosity [Pa.s]

        Outputs
        -------
        f :         float -> (!)Darcy friction factor
        """
        
        if (Re < 2300): #Laminar (Re<2300)
            sol = (64/Re)
            
        elif (Re < 4000): #Transição (2300<Re<4000)- média entre laminar e turbulento (...incerto! e sem ref!)
            sol = 0.5*(self.Churchill1977(Re,epdaux) + self.Swamee1993(Re,epdaux))
            
        else: #Turbulento (Re>4000)
            sol = self.Colebrokwhite1937(Re,epdaux)
            
        return sol/4 #sol é darcy -> converte-se para (!)Fanning

    ### okokok
    def bolhas_dispersas(self, vs_l : float, vs_g : float) -> tuple:
        """
        Description
        -------
        Function to calculate the parameters of the dispersed bubble pattern according to Taitel et al. (1980) and Barnea (1986)
        
        Inputs
        ----------
        vs_l :      float -> superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> mass density of the liquid phase [kg/m^3]
        rho_g :     float -> mass density of the gas phase [kg/m^3]
        mu_l :      float -> viscosity of the liquid phase [Pa.s]
        mu_g :      float -> viscosity of the gas phase [Pa.s]
        sigma :     float -> surface tension between the liquid and gas phases [N/m]
        d :         float -> equivalent hydraulic diameter of the pipe [m]
        epd :       float -> relative roughness of the pipe [-]
        alpha :     float -> inclination angle of the pipe [rad]
        g :         float -> acceleration due to gravity [m/s^2]

        Outputs
        -------
        dt :        float -> Maximum possible bubble diameter under intense turbulence: Modified Hinze
        dcd :       float -> Maximum diameter of non-deformable bubble: perfectly spherical
        dcb :       float -> Critical diameter for bubble migration to the top: cream is a balance between drag and buoyancy on a bubble
        alfa :      float -> Void fraction calculation - homogeneous model
        fm :        float -> Mixture friction factor
        J :         float -> Mixture velocity - center of volume
        """
        
        # velocidade da mistura - centro de volume
        J = vs_l + vs_g
        
        # Cálculo da fração de vazio - modelo homogêneo 
        alfa = vs_g/J
        
        # # massa específica da mistura
        # rho_m = (1-alfa)*self.rho_l + alfa*self.rho_g
        
        # # viscosidade da mistura
        # mu_m = (1-alfa)*self.mu_l + alfa*self.mu_g
        
        Re_m_1 = self.rho_l*J*self.d/self.mu_l # número de Reynolds do Líquido com J da Mistura
        
        # fator de fricção da mistura de FANNING    
        fm = self.fatoratrito(Re_m_1, self.epd) #tubo rugoso 

        # máximo diâmetro de bolha não deformável: perfeitamente esférica
        dcd = 2*np.sqrt((0.4*self.sigma)/((self.rho_l-self.rho_g)*self.grav))
        
        # diâmetro crítico onde haverá migração das bolhas para parte superior:cream
        # trata-se de um equilíbrio entre arrasto e empuxo em uma bolha (Shoham, 2005, pág 208 - Barnea, 1987)
        dcb = (3/8)*(self.rho_l/(self.rho_l-self.rho_g))*((fm*J**2)/(self.grav*np.cos(self.alpha)))
        
        # máximo diâmetro possível de bolha sob intensa turbulência:Hinze modificado (Shoham, 2005, Pág 236 -Barnea, 1987)
        dt = (0.725 + 4.15*np.sqrt(alfa))*(self.sigma/self.rho_l)**(3/5)*((2*fm*J**3)/self.d)**(-2/5)
        
        return dt, dcd, dcb, alfa, fm, J

    ### okokok
    def trans_bolhas_dispersas(self, vs_l : float, vs_g : float) -> bool:
        """
        Description
        -------
        Function to check if the expected pattern is dispersed bubbles according to Taitel et al. (1980) and Barnea (1986)

        Inputs
        ----------
        vs_l :      float -> superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> mass density of the liquid phase [kg/m^3]
        rho_g :     float -> mass density of the gas phase [kg/m^3]
        mu_l :      float -> viscosity of the liquid phase [Pa.s]
        mu_g :      float -> viscosity of the gas phase [Pa.s]
        sigma :     float -> surface tension between the liquid and gas phases [N/m]
        d :         float -> equivalent hydraulic diameter of the pipe [m]
        epd :       float -> relative roughness of the pipe [-]
        alpha :     float -> inclination angle of the pipe [rad]
        g :         float -> acceleration due to gravity [m/s^2]

        Outputs
        -------
        e_bd :      bool -> Is it a dispersed bubble pattern?
        """
        
        [dt, dcd, dcb, alfa, _, _] = self.bolhas_dispersas(vs_l,vs_g)

        # é bolhas dispersas se a fração de vazio for menor que 52% (sphere packing) e se a bolha for menor que os diâmetros críticos (sem deformação - esteira e sem migração para o topo-dorso)
        if ((dt < min(dcd,dcb)) and (alfa < 0.52)):
            e_bd = True
        else:
            e_bd = False
        
        return e_bd

    ### okokok
    def nivel_estratificado(self, vs_l : float, vs_g : float) -> tuple:
        """
        Description
        -------
        Function that calculates the equilibrium liquid height and all other geometric properties to be used in other routines
        
        Inputs
        ----------
        vs_l :      float -> Superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> Superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> Mass density of the liquid phase [kg/m^3]
        rho_g :     float -> Mass density of the gas phase [kg/m^3]
        mu_l :      float -> Viscosity of the liquid phase [Pa.s]
        mu_g :      float -> Viscosity of the gas phase [Pa.s]
        sigma :     float -> Surface tension between the liquid and gas phases [N/m]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        epd :       float -> Relative roughness of the pipe [-]
        alpha :     float -> Inclination angle of the pipe [rad]
        g :         float -> Acceleration due to gravity [m/s^2]

        Outputs
        -------
        hlt :       float -> Dimensionless liquid height
        SLt :       float -> Parietal perimeter of the dimensionless liquid phase
        SGt :       float -> Parietal perimeter of the dimensionless gas phase
        SIt :       float -> Interfacial perimeter between dimensionless phases
        ALt :       float -> Area of the dimensionless liquid phase
        AGt :       float -> Area of the dimensionless gas phase
        ULt :       float -> Ratio between in situ and surface velocity of the liquid phase (=1/alpha1) dimensionless
        UGt :       float -> Ratio between in situ and surface velocity of the gas phase (=1/alpha2) dimensionless
        DLt :       float -> Equivalent hydraulic diameter of the dimensionless liquid phase
        DGt :       float -> Equivalent hydraulic diameter of the dimensionless gas phase
        """
        
        flag_1 = True  # flag do fator de fricção da fase líquido - False é a recomendação de Shoham (2005) // True é a recomendação de Taitel & Duckler (1976)
        flag_2 = True  # flag do fator de fricção da interface - False é a recomendação de Shoham (2005) // True é a recomendação de Taitel & Duckler (1976)
        flag_3 = False   # flag da busca com multiplicidade - False é a busca simples //True trata multiplicidade das raízes para o caso inclinado ascendente 
        
        # fator de fricção da fase líquida
        def f_liquido(ULt,DLt): # fator de atrito da fase líquida
        
            Re_1 = (self.rho_l*abs(ULt*vs_l)*(DLt*self.d))/self.mu_l        # número de Reynolds da fase líquida
            
            if flag_1:
                f_1 = self.fatoratrito(Re_1, self.epd/DLt)                 # colebrok-white para Taitel & Duckler (1976)
                
            else: # OBS.: difícil de acreditar num modelo que não depende da rugosidade 
                Re_S1 = self.rho_l*abs(vs_l)*self.d/self.mu_l
                f_1 = 1.6291*((vs_g/vs_l)**0.0926)/(Re_S1**0.5161)     # Shoham (2005) pag 84 - Ouyang & Aziz (1996) -> ERRO!
                #f_1 = 1.6291*((vs_g/vs_l)**0.0926)/(Re_S1)            # Shoham (2005) pag 240 - Ouyang & Aziz (1996) -> ERRO!
                #f_1 = 1.6291*((vs_g/vs_l)**0.0926)/(Re_1**0.5161)     # Gomez Shoham (2000) pag 4 - Ouyang & Aziz (1996) -> ERRO!
                
            return f_1
        
        ### okokok
        # fator de fricção da interface - Shoham (2005) pag 79 a 82 - Xiao et al. (1990)
        def f_interface(f_2 : float, hlt : float, ULt : float, UGt : float, DGt : float) -> float:
            
            if flag_2:
                
                f_i =  f_2 # [=f_2] Shoham (2005) pag 77 - Taitel & Duckler (1976) ->  estratificado liso
                
                # "efeito vento" ou ondas induzidas pelo cisalhamento provocado pelo gas - Taitel & Dukler (1976) - Barnea (1987) - Ref.: Shoham (2005) Eq. 3.53 pag 72
                s = 0.01 # sheltering coefficient
                vs_g_ond = (1/UGt)*np.sqrt(4*self.mu_l*(self.rho_l-self.rho_g)*self.grav*np.cos(self.alpha)/(self.rho_l*s*self.rho_g*(ULt*vs_l))) # velocidade sup do gás de transição para ondulado

                if ((((ULt*vs_l)/np.sqrt(self.grav*hlt*self.d)) >= 1.5) and (self.alpha<0)):   #  padrão estratificado ondulado  - roll waves -> só descentente e Fr>= 1.5 Barnea (1987)
                    f_i = 0.03                                              # Shoham (2005) pag 86 - Amavaradi (1993) - fator de fricção da interface para roll waves
                        
                elif (vs_g >= vs_g_ond):              # padrão estratificado ondulado  - wind effect
                    f_i = 0.0142                    # Shoham (2005) pag 78 - Cohen Hanratty (1968) e Shoham e Taitel (1984) ->  estratificado ondulado                      # [=f_2] Shoham (2005) pag 77 - Taitel & Duckler (1976)   

                #f_i =  f_2 #  Testar Taitel Raiz (só descomentar!) -> colebrok-white para Taitel & Duckler (1976)
                    
            else:   # Shoham (2005) pag 79 a 82 - Xiao et al. (1990)
            # OBS.: prevê um critério para testar se é ondulado que só depende da pressão (sem tensão interfacial, massas específicas e escorregamento) - nada a ver com KH, também difícil de acreditar!
                f_i =  self.f_friccao_interface_xiao1990(hlt,ULt,UGt,DGt,vs_l,vs_g)               
            
            return f_i
        
        def geo_S(hltf : float) -> float :
            # Calcula parâmetros geométricos do padrão estratificado
            # OBS.: ver seção 3.2.1 de Shoham (2005) -> Taitel e Duckler (1976)
            
            # parâmetros geométricos
            auxf = (2*hltf - 1)                         # variável auxiliar
            
            SGt = np.arccos(auxf)                         # perímetro parietal da fase gasosa adimensional
            SLt = np.pi - SGt                           # perímetro parietal da fase líquida adimensional
            SIt = np.sqrt(1 - ((auxf)**2))              # perímetro interfacial entre as fases adimensional
            ALt = 0.25*(SLt + (auxf)*SIt)               # área da fase líquida adimensional
            AGt = 0.25*(SGt - (auxf)*SIt)               # área da fase gasosa adimensional
            ULt = (np.pi/4)/ALt                         # razão entre velocidade in situ e superficial da fase líquida (=1/alfa1) adimensional
            UGt = (np.pi/4)/AGt                         # razão entre velocidade in situ e superficial da fase gasosa (=1/alfa2) adimensional
            DLt = 4*ALt/SLt                             # diâmetro hidráulico equivalente da fase líquida adimensional 
            DGt = 4*AGt/(SGt + SIt)                     # diâmetro hidráulico equivalente da fase gasosa adimensional
    
            return SGt, SLt, SIt, ALt, AGt, ULt, UGt, DLt, DGt
        
        def tensoes_S(hltf : float, ULt : float, DLt : float, UGt : float, DGt : float) -> float:
            
            # fatores de fricção para cada fase
            ## fator de atrito da fase líquida
            f_1 = f_liquido(ULt,DLt) # True é a opção usando colebrok-white  - Taitel & Duckler (1976) [Ouyang & Aziz (1996) não depende da rugosidade - difícil de acreditar!]     
            
            ## fator de atrito da fase gasosa
            Re_2 = (self.rho_g*abs(UGt*vs_g)*(DGt*self.d))/self.mu_g       # número de Reynolds da fase gasosa
            f_2 =  self.fatoratrito(Re_2, self.epd/DGt)       # fator de fricção da fase gasosa - colebrok-white para Taitel & Duckler (1976) e Shoham (2005)
            
            ## fator de atrito na interface 
            f_i = f_interface(f_2,hltf,ULt,UGt,DGt)
            
            # tensões de cisalhamento parietais de cada fase
            Tau_W_1 = f_1*self.rho_l*(ULt*vs_l)*abs(ULt*vs_l)/2              # tensão de cisalhamento parietal da fase líquida
            
            Tau_W_2 = f_2*self.rho_g*(UGt*vs_g)*abs(UGt*vs_g)/2              # tensão de cisalhamento parietal da fase gasosa
        
            # tensão de cisalhamento na interface        
            Tau_i = f_i*self.rho_g*(UGt*vs_g - ULt*vs_l)*abs(UGt*vs_g - ULt*vs_l)/2   # cisalhamento na interface
            
            return Tau_W_1,Tau_W_2,Tau_i
        
        def balanco_momento_S(hltf : float) -> float:
            # OBS.: ver seção 3.2.1 de Shoham (2005) -> Taitel e Duckler (1976)
            
            [SGt, SLt, SIt, ALt, AGt, ULt, UGt, DLt, DGt] = geo_S(hltf) # geometria estratificado

            [Tau_W_1,Tau_W_2,Tau_i] = tensoes_S(hltf,ULt,DLt,UGt,DGt)   # tensões estratificado

            # balanço do momento:        
            bm = Tau_W_2*SGt/(AGt*self.d) - Tau_W_1*SLt/(ALt*self.d) + Tau_i*(SIt/self.d)*( 1/ALt + 1/AGt ) - (self.rho_l - self.rho_g)*self.grav*np.sin(self.alpha) # Shoham (2005) pg. 239 Eq3.28
            
            return bm
        
        # Resolve-se a altura adimensional de equilíbrio -> hlt [0,1] é a raiz do balanço de momento
        # OBS.1: como busca-se a solução em um intervalo fechado hlt [0,1], o método de Brent é o melhor, pois combina biseção com secante, mas função deve ter sinais opostos nos limites do intervalo de busca
        # OBS.2: o balanço de momento possui singularidades em hlt = 0 e = 1, então os limites dos intervalos devem ser bem próximos, porém diferentes 
        # OBS.3: Tomar cuidado com o met da secante, pois o modelo é não linear e leva resultados absurdos se hlt estiver fora do intervalo [0,1]
        # OBS.4: a não congergência da solução já indica que o padrão não é possível! 
        # OBS.5: ver seção 3.2.1 pag 68 de Shoham (2005) -> 3 soluções par ao caso ascendente inclinado (a menor é a estável e deve ser a escolhida), não há multiplicidade para horizontal e descendente (1 raiz somente)
    
        # intervalo total de busca
        margem = 0.00001 # eps
        hlt_min = 0 + margem # margem para evitar as singularidades
        hlt_max = 1 - margem # margem para evitar as singularidades
        passo = 0.01
        
        # função que vai buscar a raiz do balanço de momento em um dado intervalo: met brentq (combina biseção com secante - melhor combinação possível para busca de sol em intervalo definido)
        def acha_raiz(hlt_min,hlt_max,passo):
            
            # se os sinais dos extremos do intervalo forem diferentes, chama o método de brent
            if (np.sign(balanco_momento_S(hlt_min)) != np.sign(balanco_momento_S(hlt_max))):
                
                try: # como pode haver problemas de convergência, deve-se pensar numa excessão!
                    hlt = brentq(balanco_momento_S,hlt_min,hlt_max)
                    
                except:
                    
                    hlt = np.nan
                    # print("MAPA - Aviso: o método de brent não convergiu: hlt - Estratificado")
                    
            else: # caso os sinais dos extremos sejam iguais -> temos de refinar a faixa -> caso com esc contracorrente fica mais "mal comportado", esta busca será mais frequente
                
                while ((np.sign(balanco_momento_S(hlt_min)) == np.sign(balanco_momento_S(hlt_max))) and (hlt_max > hlt_min + passo)): # desconta um passo do limite superior (tende a buscar a menor raíz - estabilidade) com critério de parada caso não haja
                    
                    hlt_max = hlt_max - passo # uma possibilidade de melhorar esta busca seria usar algo q remonte a biseção

                try: # como pode haver problemas de convergência, deve-se pensar numa excessão!
                
                    hlt = brentq(balanco_momento_S,hlt_min,hlt_max)
                    
                except:
                    
                    hlt = np.nan
                    # print("MAPA - Aviso: o método de brent não convergiu: hlt - Estratificado")
        
            return hlt
        #-----------------------
        # Tentativa de tratar multiplicidade - precisa melhorar
        
        # para esc bifásico estratificado pode haver até 3 raízes e a menor é a estável, considerando que lambda é uam aproximação de hlt (modelo homogêneo) e que alfa é menor que lambda para escoreegamentos negativos (ascendente) e maior para positivos (descendentes)
        # a melhor estratégia seria dividir o intervalo de busca na altura hlt equivalente a lambda, porém isso nos daria 2 intervalos para a busca das raizes discriminados pelo escorregamento, como podemos ter 3 raízes, propôe-se dividí-lo arbitrariamente em mais 2 
        # intervalos exatamente no meio enquanto busca-se a melhor solução
        
        # busca hlt (lambda) -> só serve para calcular o hlt relativo a lambda (modelo homogênio sem escorregamento)
        def hlamb(hlt : float):
            [_, _, _, _, _, _, UGt, _, _] = geo_S(hlt)
            f_obj = 1/UGt - (vs_g/(vs_l+vs_g))
            return f_obj
        
        if flag_3: # flag_3 = True trata multiplicidade
            
            # ver seção 3.2.1 pag 68 de Shoham (2005) -> 3 soluções par ao caso ascendente inclinado (a menor é a estável e deve ser a escolhida), não há multiplicidade para horizontal e descendente (1 raiz somente)
            if (self.alpha > 0): # ascendente inclinado -> pode haver 3 raízes

                # inclinado ascendente -> escorregamento positivo, então as raízes são superiores hlt_lamb

                hlt_lamb = brentq(hlamb,hlt_min,hlt_max) # hlt equivalente a lambda
                # print(hlt_lamb)
                
                # # 01 -> busca integral
                # hlt = acha_raiz(hlt_lamb,hlt_max,passo) # busca de raiz no intervalo total, sem fracionar ou se preocupar com multiplicidade de raízes - código antigo, se quiser testar é só descomentar
                # print(hlt)
                
                # # 02 -> amostra intervalo regularmente, mapeia mudanças de sinais, calcula os zeros para cada intervalo com mudança de sinal
                # h_search = np.linspace(hlt_lamb,hlt_max,10)                                                         # fraciona intervalo
                # intervalo = np.where(np.diff(np.sign(np.array([balanco_momento_S(x) for x in h_search]))))[0] + 1   # mapea mudanças de sinal na função objetivo
                # # hlt = np.array([acha_raiz(h_search[i-1],h_search[i],passo) for i in intervalo])                     # busca as raízes entorno das transições dos sinais
                # # hlt = np.nanmin(hlt)
                # # print(hlt)

                # # uma otimização seria finalizar o loop se uma raiz for encontrada -> como começa da menor para a maior, assim acaba-se selecionando a menor sem ter que calcular as outras
                # for i in intervalo:
                #     hlt = acha_raiz(h_search[i-1],h_search[i],passo)
                #     if (~np.isnan(hlt)):
                #         break
                # print(hlt)
                
                # 03 -> amostra intervalo regular e busca as raízes
                h_search = np.linspace(hlt_lamb,hlt_max,3)                                                         # fraciona intervalo
                # hlt = np.array([acha_raiz(h_search[i-1],h_search[i],passo) for i in range(1, len(h_search))])      # busca as raízes entorno das transições dos sinais
                # hlt = np.nanmin(hlt)
                # print(hlt)
                
                # uma otimização seria finalizar o loop se uma raiz for encontrada -> como começa da menor para a maior, assim acaba-se selecionando a menor sem ter que calcular as outras
                for i in range(1, len(h_search)):
                    hlt = acha_raiz(h_search[i-1],h_search[i],passo)
                    if (~np.isnan(hlt)):
                        break
                # print(hlt)
                
                # # exibir quando há multiplicidade
                # if np.count_nonzero(~np.isnan(hlt))>1: 
                #     print(hlt)
                #     print(acha_raiz(hlt_min,hlt_max,passo))              
                #     print(h_search)
                #     print(intervalo)
                
            else: # horizontal e descendente -> somente uma raiz
                
                hlt = acha_raiz(hlt_min,hlt_max,passo) # busca de raiz no intervalo total, sem fracionar ou se preocupar com multiplicidade de raízes - código antigo, se quiser testar é só descomentar
            
        else:

            # Solução padrão sem tratar multiplicidade - pedreirão!
            hlt = acha_raiz(hlt_min,hlt_max,passo) # busca de raiz no intervalo total, sem fracionar ou se preocupar com multiplicidade de raízes - código antigo, se quiser testar é só descomentar
            
        # calcula parâmetros geométricos usando a altura adimensional de equilíbrio
        [SGt, SLt, SIt, ALt, AGt, ULt, UGt, DLt, DGt] = geo_S(hlt)
        
        # calcula as tenões de cisalhamento usando a altura adimensional de equilíbrio
        [Tau_W_1,Tau_W_2,Tau_i] = tensoes_S(hlt,ULt,DLt,UGt,DGt)   # tensões estratificado

        return hlt, SGt, SLt, SIt, ALt, AGt, ULt, UGt, DLt, DGt, Tau_W_1, Tau_W_2, Tau_i

    ### okokok
    def fracao_liquido_slug(self, vs_l : float, vs_g : float) -> float:
        """
        Description
        -------
        Function that checks which intermittent regime it is: elongated bubbles, slug and churn
        
        Inputs
        ----------
        vs_l :      float -> Superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> Superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> Mass density of the liquid phase [kg/m^3]
        rho_g :     float -> Mass density of the gas phase [kg/m^3]
        mu_l :      float -> Viscosity of the liquid phase [Pa.s]
        mu_g :      float -> Viscosity of the gas phase [Pa.s]
        sigma :     float -> Surface tension between the liquid and gas phases [N/m]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        epd :       float -> Relative roughness of the pipe [-]
        alpha :     float -> inclination angle of the pipe [rad]
        g :         float -> Acceleration due to gravity [m/s^2]

        Outputs
        -------
        Rs :        float -> Slug liquid fraction [-]
        """ 
        
        # OBS.: a orientação em Barnea (1987) é que deve-se percorrer no mapa de padrões uma linha com J constante até a fronteira com bolhas dispersas
        
        # calcula parâmetros do padrão bolhas dispersas na golfada de líquido 
        [_, dcd, dcb, _, fm, J] = self.bolhas_dispersas(vs_l,vs_g) # vs_l e vs_g são relativos ao padrão golfadas, deve-se pegar no mapa a linha de J cte até a fronteira com bolhas dispersas, daí usa-se o lambda deste ponto para calcular vs_l e vs_g
        
        #Teste:o diâmetro crítico é o menor, aquele que levará a golfada ao limite de transição - Barnea & Brauner (1985)
        # dc = min(dcd,dcb)
        dc = dcd
        
        # Shoham (2005) pag 110 eq 3.167 -> diâmetro da bolha (dcd) e fator de fricção (fs -> )
        # a fração de vazio do limite de transição de bolhas dispersas é a máxima fração de vazio que a golfada pode acomodar - Barnea & Brauner (1985) 
        alfas = 0.058*(dc*((2*fm*(J**3)/self.d)**(2/5))*((self.rho_l/self.sigma)**(3/5)) - 0.725)**2 # fração de vazio no slug
                
        Rs = 1 - alfas 
        
        return Rs

    ### okokok
    def nivel_anular(self, vs_l : float, vs_g : float) -> tuple:
        """
        Description
        -------
        Function that checks the stability criteria of the annular pattern [Barnea (1986)] (film stability or core blockage by liquid -> intermittent transition)
        
        Inputs
        ----------
        vs_l :      float -> superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> mass density of the liquid phase [kg/m^3]
        rho_g :     float -> mass density of the gas phase [kg/m^3]
        mu_l :      float -> viscosity of the liquid phase [Pa.s]
        mu_g :      float -> viscosity of the gas phase [Pa.s]
        sigma :     float -> surface tension between the liquid and gas phases [N/m]
        d :         float -> equivalent hydraulic diameter of the pipe [m]
        epd :       float -> relative roughness of the pipe [-]
        alpha :     float -> inclination angle of the pipe [rad]
        g :         float -> acceleration due to gravity [m/s^2]

        Outputs
        -------
        alfa_1 :    float -> volumetric fraction of equilibrium liquid -> alpha_1 [0.1]
        X2 :        float -> Lockhart-Martinelli number
        Y :         float -> Lockhart-Martinelli number
        """
        
        # cálculo do alfa_1 de equilíbrio
        
        # gradientes de pressão de fricção - fase sozinha no duto!
        # fatores de fricção para cada fase de FANNING
        Re_1 = (self.rho_l*abs(vs_l)*self.d)/self.mu_l              # número de Reynolds da fase líquida
        f_1 = self.fatoratrito(Re_1, self.epd)                 # fator de atrito da fase líquida - Rugoso
        
        Re_2 = (self.rho_g*abs(vs_g)*self.d)/self.mu_g              # número de Reynolds da fase gasosa
        f_2 = self.fatoratrito(Re_2, self.epd)                 # fator de atrito da fase gasosa - Rugoso
        
        # gradientes de pressão
        dpdx_1S = (2*f_1*self.rho_l*(abs(vs_l)*vs_l))/self.d   # líquido
        dpdx_2S = (2*f_2*self.rho_g*(abs(vs_g)*vs_g))/self.d   # gás
        
        # número de Lockhart-Martinelli
        X2 = dpdx_1S/dpdx_2S                        
        Y = (self.rho_l-self.rho_g)*self.grav*np.sin(self.alpha)/dpdx_2S

        def balanco_momento_A(alfa_1): # Shoham (2005) pag 205 Eq 5.28 -> Barnea (1987)
        
            f = ((1 + 75*alfa_1)/(((1-alfa_1)**(5/2))*alfa_1)) - (1/(alfa_1**3))*X2 - Y # função obj que deseja-se a raiz
            
            return f
        
        # Resolve-se a fração volumétrica de líquido de equilíbrio -> alfa_1 [0,1] é a raiz do balanço de momento para o padrão anular
        # OBS.1: como busca-se a solução em um intervalo fechado alfa_1 [0,1], o método de Brent é o melhor, pois combina biseção com secante, mas função deve ter sinais opostos nos limites do intervalo de busca
        # OBS.2: o balanço de momento possui singularidades em alfa_1 = 0 e = 1, então os limites dos intervalos devem ser bem próximos, porém diferentes de 0 e 1
        # OBS.3: Tomar cuidado com o met. da secante, pois o modelo é não linear e leva resultados absurdos se hlt estiver fora do intervalo [0,1]
        # OBS.4: a não congergência da solução já indica que o padrão não é possível!

        # intervalo de busca
        margem = 0.0001 # eps
        alfa_1_min = 0 + margem
        alfa_1_max = 1 - margem
        passo = 0.01        
                                    
        # se os sinais dos extremos do intervalo forem diferentes, chama o método de brent
        if (np.sign(balanco_momento_A(alfa_1_min)) != np.sign(balanco_momento_A(alfa_1_max))):
            
            try: # como pode haver problemas de convergência, deve-se pensar numa excessão!
                alfa_1 = brentq(balanco_momento_A,alfa_1_min,alfa_1_max)
                
            except:
                alfa_1 = np.nan
                print("Mapa - Aviso: o método de brent não convergiu: alfa_1 -> nivel_anular")
                
        else: # caso os sinais dos extremos sejam iguais
            # desconta um passo do limite superior (tende a buscar a menor raíz - estabilidade) com critério de parada caso não haja
            while ((np.sign(balanco_momento_A(alfa_1_min)) == np.sign(balanco_momento_A(alfa_1_max))) and (alfa_1_max > alfa_1_min + passo)):
                
                alfa_1_max = alfa_1_max - passo # uma possibilidade de melhorar esta busca seria usar algo q remonte a biseção    
                
            try: # como pode haver problemas de convergência, deve-se pensar numa excessão!
                alfa_1 = brentq(balanco_momento_A,alfa_1_min,alfa_1_max)
                
            except:
                alfa_1 = np.nan
                print("Mapa - Aviso: o  método de brent não convergiu: alfa_1 -> nivel_anular")
        
        return alfa_1, X2, Y

    ### okokok
    def anular(self, vs_l : float, vs_g : float) -> bool:
        """
        Description
        -------
        Function that checks the stability criteria of the annular pattern [Barnea (1986)] (film stability or core blockage by liquid -> intermittent transition)
        
        Inputs
        ----------
        vs_l :      float -> Superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> Superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> Mass density of the liquid phase [kg/m^3]
        rho_g :     float -> Mass density of the gas phase [kg/m^3]
        mu_l :      float -> Viscosity of the liquid phase [Pa.s]
        mu_g :      float -> Viscosity of the gas phase [Pa.s]
        sigma :     float -> Surface tension between the liquid and gas phases [N/m]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        epd :       float -> Relative roughness of the pipe [-]
        alpha :     float -> Inclination angle of the pipe [rad]
        g :         float -> Acceleration due to gravity [m/s^2]

        Outputs
        -------
        e_AN :      bool -> Whether the annular pattern is stable
        """
        
        [alfa_1,X2,Y] = self.nivel_anular(vs_l,vs_g)
    
        # Teste 1 e 2: instabilidade do filme de líquido e alfal grande o suficiente para tamponar o core -> Barnea (1986) e Barnea (1987)        
        Rs = 0.48  # mínimo holdup de líquido necessário para formação do slug
        #Rs = fracao_liquido_slug(vs_l,vs_g)
        
        if (Y >= (((2 - 3*alfa_1/2)/(alfa_1**3*(1 - 3*alfa_1/2)))*X2)) or ((alfa_1/Rs) >= 0.5):
            e_AN = False
            
        else:
            e_AN = True
        
        return e_AN

    ### okokok
    def nivel_bolha(self, vs_l : float, vs_g : float) -> bool:
        """
        Description
        -------
        Function that checks if the pattern is bubbles
        
        Inputs
        ----------
        vs_l :      float -> Superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> Superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> Mass density of the liquid phase [kg/m^3]
        rho_g :     float -> Mass density of the gas phase [kg/m^3]
        mu_l :      float -> Viscosity of the liquid phase [Pa.s]
        mu_g :      float -> Viscosity of the gas phase [Pa.s]
        sigma :     float -> Surface tension between the liquid and gas phases [N/m]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        epd :       float -> Relative roughness of the pipe [-]
        alpha :     float -> Inclination angle of the pipe [rad]
        g :         float -> Acceleration due to gravity [m/s^2]

        Outputs
        -------
        e_B :       bool -> Whether the bubble pattern exists and is stable
        """
        
        # Resolve-se a fração de vazio de equilíbrio considerando o escorregamento - modelo de drift flux -> alfa [0,1] é a raiz da função objetivo do modelo de drift
        # OBS.1: como busca-se a solução em um intervalo fechado alfa [0,1], o método de Brent é o melhor, pois combina biseção com secante, mas função deve ter sinais opostos nos limites do intervalo de busca
        # OBS.2: sem singularidades nos extremos alfa [0,1]!
        # OBS.3: Tomar cuidado com o met da secante, pois o modelo é não linear e leva resultados absurdos se hlt estiver fora do intervalo [0,1]
        # OBS.4: a não congergência da solução já indica que o padrão não é possível!
        
        # implementa o modelo de drift para bolhas - Shoham (2005) pag. 245 Eq. 4.42 Hassan e Kabir (1988)
        def drift_bolhas(alfa): # função cuja raiz dá o alfa de equilíbrio (Barnea, 1987)
        
            # permitir escolha de modelos -> possível melhorar com modelos do Ishi (regime de bolhas, etc)
            n = 0.5
            C0 = 1.2
            Vinf = 1.53*(((self.grav*self.sigma*(self.rho_l - self.rho_g))/(self.rho_l**2))**(1/4)) # Harmathy
            
            f = - vs_g + alfa*C0*(vs_l+vs_g) + alfa*((1-alfa)**n)*Vinf*np.sin(self.alpha)
            
            return f
    
        # intervalo de busca
        alfa_min = 0
        alfa_max = 1
                                                        
        # se os sinais dos extremos do intervalo forem diferentes, chama o método de brent
        if (np.sign(drift_bolhas(alfa_min)) != np.sign(drift_bolhas(alfa_max))):
            
            try: # como pode haver problemas de convergência, deve-se pensar numa excessão!
                alfa = brentq(drift_bolhas,alfa_min,alfa_max)
                
            except:
                alfa = np.nan
                print("MAPA - Aviso: o método de brent não convergiu: alfa bolhas -> bolha")
                
        else: # caso os sinais dos extremos sejam iguais 
        
            # o fsolve faz uma busca mais ampla e pode extrapolar o intervalo alfa [0,1], necessário testar as raízes! (Powell hybrid method)
            try: # como pode haver problemas de convergência, deve-se pensar numa excessão!
                alfa_i = 0.5 # estimativa inicial - > meio do intervalo [0,1]
                alfa = fsolve(drift_bolhas, alfa_i)[0] #solução é um array -> sol[0]
            
                if ((alfa < alfa_min) or (alfa > alfa_max)):
                    alfa = np.nan
                    print("MAPA - Erro: alfa fora do intervalo [0,1]: alfa bolhas -> bolha")
                
            except:
                alfa = np.nan
                print("MAPA - Aviso: o método Powell Hybrid (fsolve) não convergiu: alfa bolhas -> bolha")
        
        return alfa
    
    ### okokok
    def bolha(self, vs_l : float, vs_g : float) -> bool: 
        """
        Description
        -------
        Function that checks if the pattern is bubbles
        
        Inputs
        ----------
        vs_l :      float -> Superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> Superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> Mass density of the liquid phase [kg/m^3]
        rho_g :     float -> Mass density of the gas phase [kg/m^3]
        mu_l :      float -> Viscosity of the liquid phase [Pa.s]
        mu_g :      float -> Viscosity of the gas phase [Pa.s]
        sigma :     float -> Surface tension between the liquid and gas phases [N/m]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        epd :       float -> Relative roughness of the pipe [-]
        alpha :     float -> Inclination angle of the pipe [rad]
        g :         float -> Acceleration due to gravity [m/s^2]

        Outputs
        -------
        e_B :       bool -> Whether the bubble pattern exists and is stable
        """
        
        # calcula o diâmetro crítico da tubulação: velocidade da bolha de taylor superior à velocidade das  bolhas (coalescência) -> para dutos de grandes diâmetros - Taitel et al. (1980)
        d_crit = 19*np.sqrt((self.rho_l - self.rho_g)*self.sigma/((self.rho_l**2)*self.grav))
        
        # ângulo crítico
        U0 = 1.53*((self.grav*(self.rho_l - self.rho_g)*self.sigma)/(self.rho_l**2))**(1/4)      # velocidade de ascensão da bolha
        Cl = 0.8                                                        # coeficiente de lift da bolha
        gama = (1.1 + 1.5)/2;                                           # coeficiente de distorção da bolha - a referência dada no artigo é um intervalo, usou-se o médio
        # gama = 1.1;                                                   # coeficiente de distorção da bolha - a referência dada no artigo é um intervalo, usou-se o médio
        
        # calcula-se o diâmetro da bolha
        [_, dcd, dcb, _, _, _] = self.bolhas_dispersas(vs_l,vs_g)
        
        # qual diâmetro é o crítico?
        dc = min(dcd,dcb)
        
        # calcula o ângulo crítico que previna a migração de bolhas para o topo (arrasto > que lift) o que geraria coalescência - Barnea et al. (1985)
        def bubbs(alpha_crit): # função obj cuja raiz dará o alpha_crit
        
            #f = (np.cos(alpha_crit)/(np.sin(alpha_crit)**2)) - (3/4)*np.cos(np.pi/4)*((U0**2)/g)*(Cl*(gama**2)/dc) # tem uma singularidade em alpha_crit = 0
            f = np.cos(alpha_crit) - (np.sin(alpha_crit)**2)*(3/4)*np.cos(np.pi/4)*((U0**2)/self.grav)*(Cl*(gama**2)/dc) # possui os mesmos zeros, mas deve-se testar para que seja diferente de zero
            
            return f
        
        #a função obj é par, então não precisa buscar [-90,90], basta buscar [0+eps,90] (0+eps-> 0 é singularidade)
        alpha_crit_min = 0.0001*np.pi/180
        alpha_crit_max = 90*np.pi/180
        
        # se os sinais dos extremos do intervalo forem diferentes, chama o método de brent
        if (np.sign(bubbs(alpha_crit_min)) != np.sign(bubbs(alpha_crit_max))):
            
            try: # como pode haver problemas de convergência, deve-se pensar numa excessão!
                alpha_crit = brentq(bubbs,alpha_crit_min,alpha_crit_max)
                
            except:
                alpha_crit = np.nan
                print("MAPA - Aviso: o método de brent não convergiu: alpha_crit -> bolha")
                
        else: # caso os sinais dos extremos sejam iguais 
        
            # o fsolve faz uma busca mais ampla e pode extrapolar o intervalo, necessário testar as raízes! (Powell hybrid method)
            try: # como pode haver problemas de convergência, deve-se pensar numa excessão!
                alpha_crit_i = 45*np.pi/180 # estimativa inicial - > meio do intervalo
                alpha_crit = fsolve(bubbs, alpha_crit_i)[0] #solução é um array -> sol[0]
                
                if ((alpha_crit < alpha_crit_min) or (alpha_crit > alpha_crit_max)):
                    alpha_crit = np.nan
                    print("MAPA - Erro: raiz fora do intervalo: alpha_crit -> bolha")
                    
            except:
                alpha_crit = np.nan
                print("MAPA - Aviso: o método Powell Hybrid (fsolve) não convergiu: alpha_crit -> bolha")
        
        # diâmetro maior que o crítico e angulo suficientemente grande para prevenir a migração de bolhas para o topo da parede do duto
        if ((self.d > d_crit) and (abs(self.alpha) > alpha_crit)):

            alfa = self.nivel_bolha(vs_l,vs_g)
            
            if (alfa < 0.25): # não há padrão bolhas se a fração de vazio for superior a 25% - Taitel et al. (1980)
                e_B = True
                
            else:
                e_B = False
        
        else:
            e_B = False
        
        return e_B

    ### okokok
    def intermitentes(self, vs_l : float, vs_g : float) -> tuple:
        """
        Description
        -------
        Function that checks which intermittent regime it is: elongated bubbles, slug and churn
        
        Inputs
        ----------
        vs_l :      float -> Superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> Superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> Mass density of the liquid phase [kg/m^3]
        rho_g :     float -> Mass density of the gas phase [kg/m^3]
        mu_l :      float -> Viscosity of the liquid phase [Pa.s]
        mu_g :      float -> Viscosity of the gas phase [Pa.s]
        sigma :     float -> Surface tension between the liquid and gas phases [N/m]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        epd :       float -> Relative roughness of the pipe [-]
        alpha :     float -> Inclination angle of the pipe [rad]
        g :         float -> Acceleration due to gravity [m/s^2]

        Outputs
        -------
        e_BA :      bool -> Whether the elongated bubbles pattern is stable
        e_SL :      bool -> Whether the slug pattern is stable
        e_CH :      bool -> Whether the churn pattern is stable
        """    
        
        Rs = self.fracao_liquido_slug(vs_l, vs_g)
        
        if (Rs == 1): # se alfas é nulo (ZERO é muito restrito!), temos bolhas alongadas (caso limite do slug, com golfada não aerada) - Barnea & Brauner (1985)
            e_BA = True
            e_SL = False
            e_CH = False
            
        elif (Rs >= 0.48): # se alfas é menor que 52% (limite de fração de vazio na slug em BD), temos slug  
            e_BA = False
            e_SL = True
            e_CH = False
            
        else: # se alfas é maior que 52% (limite de fração de vazio na slug em BD), há coalescência na golfada, que é cortada pelo gás, levando ao churn - Brauner & Barnea (1986)
            e_BA = False
            e_SL = False
            e_CH = True


        return e_BA,e_SL,e_CH

    ### okokok
    def estratificado_ondulado(self, hlt : float, ULt : float, UGt : float, vs_l : float, vs_g : float) -> bool:
        """
        Description
        -------
        Function that checks whether the pattern is wavy layered or not
        
        Inputs
        ----------
        hlt :       float -> Dimensionless liquid height
        ULt :       float -> Ratio between in situ and surface velocity of the liquid phase (=1/alpha1) dimensionless
        UGt :       float -> Ratio between in situ and surface velocity of the gas phase
        vs_l :      float -> Superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> Superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> Mass density of the liquid phase [kg/m^3]
        rho_g :     float -> Mass density of the gas phase [kg/m^3]
        mu_l :      float -> Viscosity of the liquid phase [Pa.s]
        mu_g :      float -> Viscosity of the gas phase [Pa.s]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        alpha :     float -> Inclination angle of the pipe [rad]
        g :         float -> Acceleration due to gravity [m/s^2]

        Outputs
        -------
        e_STW :    bool -> Whether the stratified pattern is wavy
        """

        s = 0.01 # sheltering coefficient - Ref.: Shoham (2005) Eq. 3.53 - Não tem em Barnea (1987)
        
        # (1): "efeito vento" ou ondas induzidas pelo cisalhamento provocado pelo gas - Taitel & Dukler (1976)
        # (2): mesmo sem cisalhamento, no escoamento descendentes podem surgir ondas ("roll waves"), condição depende do num de Froude - Barnea et al.(1982)
        EOP1 = ((UGt*vs_g) >= np.sqrt(4*self.mu_l*(self.rho_l-self.rho_g)*self.grav*np.cos(self.alpha)/(self.rho_l*s*self.rho_g*(ULt*vs_l))))
        EOP2 = ((((ULt*vs_l)/np.sqrt(self.grav*hlt*self.d)) >= 1.5) and (self.alpha<0))
        if ( EOP1 or EOP2):
            e_STW = True
            
        else: # não é padrão estratificado ondulado
            e_STW = False
        
        return e_STW
    
    ### okokok
    def trans_estratificado_aunular(self, hlt : float, ULt : float, DLt : float, vs_l : float) -> bool:
        """
        Description
        -------
        Function that verifies the transition from stratified to annular 
        based on the droplets' trajectories (at high speed, drops detach 
        from the film and adhere to the top, forming an annular concentric film) 
        Barnea et al. (1982)
        
        Inputs
        ----------
        hlt :       float -> Dimensionless liquid height
        ULt :       float -> Ratio between in situ and surface velocity of the liquid phase (=1/alpha1) dimensionless
        DLt :       float -> Equivalent hydraulic diameter of the dimensionless liquid phase
        vs_l :      float -> Superficial velocity or volumetric flow rate of the liquid phase [m/s]
        rho_l :     float -> Mass density of the liquid phase [kg/m^3]
        mu_l :      float -> Viscosity of the liquid phase [Pa.s]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        epd :       float -> Relative roughness of the pipe [-]
        alpha :     float -> Inclination angle of the pipe [rad]
        g :         float -> Acceleration due to gravity [m/s^2]

        Outputs
        -------
        e_ANST :    bool -> If the stratified pattern is unstable and transitions to the annular
        """
        
        # número de Reynolds da fase líquida    
        Re_1 = self.rho_l*(ULt*vs_l)*(DLt*self.d)/self.mu_l
        
        # fator de fricção da mistura de FANNING    
        #f_1 = f_blasius(Re_1)      #tubo liso
        f_1 = self.fatoratrito(Re_1, self.epd/DLt) #tubo rugoso
        
        if ((ULt*vs_l)**2 >= (self.grav*self.d*(1-hlt)*np.cos(self.alpha)/f_1)):
            e_ANST = True
        else:
            e_ANST = False
        
        return e_ANST

    ### okokok
    def trans_estratificado_kelvinhelmholtz(self, vs_l : float, vs_g : float) -> tuple:
        """
        Description
        -------
        Function that checks whether the interfacial wave of the 
        stratified pattern is unstable (helmholtz kelvin instability) 
        growing in amplitude making the stratified pattern impossible 
        (annular or intermittent)
        
        Inputs
        ----------
        vs_l :      float -> Superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> Superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> Mass density of the liquid phase [kg/m^3]
        rho_g :     float -> Mass density of the gas phase [kg/m^3]
        mu_l :      float -> Viscosity of the liquid phase [Pa.s]
        mu_g :      float -> Viscosity of the gas phase [Pa.s]
        sigma :     float -> Surface tension between the liquid and gas phases [N/m]
        d :         float -> Equivalent hydraulic diameter of the pipe [m]
        epd :       float -> Relative roughness of the pipe [-]
        alpha :     float -> Inclination angle of the pipe [rad]
        g :         float -> Acceleration due to gravity [m/s^2]

        Outputs
        -------
        e_NSKH :    bool  -> Whether stratified pattern is unstable via kelvin-helmholtz
        hlt :       float -> Dimensionless liquid height
        SLt :       float -> Parietal perimeter of the dimensionless liquid phase
        SGt :       float -> Parietal perimeter of the dimensionless gas phase
        SIt :       float -> Interfacial perimeter between dimensionless phases
        ALt :       float -> Area of the dimensionless liquid phase
        AGt :       float -> Area of the dimensionless gas phase
        ULt:        float -> Ratio between in situ and surface velocity of the liquid phase (=1/alpha1) dimensionless
        UGt:        float -> Ratio between in situ and surface velocity of the gas phase (=1/alpha2) dimensionless
        DLt :       float -> Equivalent hydraulic diameter of the dimensionless liquid phase
        DGt:        float -> Equivalent hydraulic diameter of the dimensionless gas phase
        """
        
        # Resolve a altura da interface de equilíbrio e parâmetros geométricos (todos adimensionais - nomenclatura barnea (1987))
        [hlt, SGt, SLt, SIt, ALt, AGt, ULt, UGt, DLt, DGt, _, _, _] = self.nivel_estratificado(vs_l,vs_g)
        
        # Criterio para a instabilidade de kelvin helmholtz: Taitel & Dukler (1976)
        Fro = np.sqrt(self.rho_g/(self.rho_l-self.rho_g))*(vs_g)/ \
            (np.sqrt(self.d*self.grav*np.cos(self.alpha)))
        
        Fro = np.sqrt(1.0/(self.rho_l-self.rho_g))*(vs_g)/ \
            (np.sqrt(self.d*self.grav*np.cos(self.alpha)))

        # Fro = 0.5 * (vs_g + vs_l) / \
        #     (np.sqrt(self.d*self.grav*np.cos(self.alpha)))
        
        CNS = (Fro**2) * ((1.0/((1.0-hlt)**2))*((UGt**2*SIt)/AGt))
        
        if CNS >= 1:
            e_NSKH = True
        else:
            e_NSKH = False

        return e_NSKH, hlt, SGt, SLt, SIt, ALt, AGt, ULt, UGt, DLt, DGt
    
    ### okokok
    def Shoham2005_function_point(self, vs_l : float, vs_g : float) -> str:
        """
        Description
        -------
        Function that implements the decision tree for the unified model flow pattern (upward and downward) as proposed by Barnea (1987)

        Inputs
        ----------
        vs_l :      float -> superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :      float -> superficial velocity or volumetric flow rate of the gas phase [m/s]
        rho_l :     float -> mass density of the liquid phase [kg/m^3]
        rho_g :     float -> mass density of the gas phase [kg/m^3]
        mu_l :      float -> viscosity of the liquid phase [Pa.s]
        mu_g :      float -> viscosity of the gas phase [Pa.s]
        sigma :     float -> surface tension between the liquid and gas phases [N/m]
        d :         float -> equivalent hydraulic diameter of the pipe [m]
        epd :       float -> relative roughness of the pipe [-]
        alpha :     float -> inclination angle of the pipe [rad]
        g :         float -> acceleration due to gravity [m/s^2]

        Outputs
        -------
        pattern :   str -> flow pattern ('BD': 1 dispersed bubbles, 'A': 2 annular, 'B': 3 bubbles, 'EB': 4 elongated bubbles, 'SL': 5 slug, 'CH': 6 churn, 'SW': 7 wavy stratified, 'SS': 8 smooth stratified)
        """
        e_bd = self.trans_bolhas_dispersas(vs_l,vs_g)
        if (e_bd == True): # padrão bolhas dispersas
            # self.pattern_map.append("Dispersed Bubbles")
            self.pattern_map.append(self.flow_dic_shoham["Dispersed Bubbles"])
            pattern = "Dispersed Bubbles"
        else:
            # Teste 02: Transição para não estratificado: instabilidade de  Kelvin-Helmholtz - Taitel & Duckler (1976)
            [e_NSKH, hlt, SGt, SLt, SIt, ALt, AGt, ULt, UGt, DLt, DGt] = self.trans_estratificado_kelvinhelmholtz(vs_l,vs_g)
            # Teste 03: Transição do estratificado para anular a partir das trajetórias das gotas (em alta velocidade gotas se descolam do filme e aderem o topo formando um filme concêntrico anuular) Barnea et al. (1982)
            e_ANST = self.trans_estratificado_aunular(hlt, ULt, DLt, vs_l)
            if ((e_NSKH == True) or (e_ANST == True)):
                # Teste 05: Transição anular-intermitente
                e_AN = self.anular(vs_l,vs_g)
                if (e_AN == True):     # Padrão Anular
                    # self.pattern_map.append("Annular")
                    self.pattern_map.append(self.flow_dic_shoham["Annular"])
                    pattern = "Annular"
                else: # padrões tidos como intermitentes
                    #Teste 06: Verifica se é bolha (existência e testa transição p slug)
                    e_B = self.bolha(vs_l,vs_g)
                    if (e_B == True):  # Padrão bolhas
                        # self.pattern_map.append("Bubbles")
                        self.pattern_map.append(self.flow_dic_shoham["Bubbles"])
                        pattern = "Bubbles"
                    else: # padrão intermitente (bolhas alongadas, slug e churn são generacaimente chamados de intermitentes)
                        #Teste 07: Transição Bolha alongada-slug-churn ()
                        [e_BA,e_SL,e_CH] = self.intermitentes(vs_l, vs_g)
                        if (e_BA == True): # padrão bolhas alongadas
                            # self.pattern_map.append("Elongated Bubbles")
                            self.pattern_map.append(self.flow_dic_shoham["Elongated Bubbles"])
                            pattern = "Elongated Bubbles"
                        elif (e_SL == True):  # padrão slug
                            # self.pattern_map.append("Slug")
                            self.pattern_map.append(self.flow_dic_shoham["Slug"])
                            pattern = "Slug"
                        elif (e_CH == True):  # padrão churn
                            # self.pattern_map.append("Churn")
                            # self.pattern_map.append(self.flow_dic_shoham["Churn"])
                            # pattern = "Churn"
                            if (self.alpha*180/np.pi <= 30):
                                self.pattern_map.append(self.flow_dic_shoham["Slug"])
                                pattern = "Slug"
                            else:    
                                self.pattern_map.append(self.flow_dic_shoham["Churn"])
                                pattern = "Churn"
            elif ((e_NSKH == False) and (e_ANST == False)): # estratificado
                # Teste 04: Verifica se é estratificado ondulado
                e_STW = self.estratificado_ondulado(hlt,ULt,UGt,vs_l,vs_g)
                if (e_STW == True):     # padrão estratificado ondulado
                    # self.pattern_map.append("Stratified Wavy")
                    self.pattern_map.append(self.flow_dic_shoham["Stratified Wavy"])
                    pattern = "Stratified Wavy"
                else: # padrão estratificado liso
                    # self.pattern_map.append("Smooth Stratified")
                    self.pattern_map.append(self.flow_dic_shoham["Smooth Stratified"])
                    pattern = "Smooth Stratified"
        return pattern

    ### okokok
    def Shoham2005Map(self) -> list:
        """
        Description
        -------
        Function that implements the decision tree for the unified model flow pattern (upward and downward) as proposed by Shoham (2005)

        Inputs
        ----------
        vs_l :          float -> superficial velocity or volumetric flow rate of the liquid phase [m/s]
        vs_g :          float -> superficial velocity or volumetric flow rate of the gas phase [m/s]

        Outputs
        -------
        pattern_map :   int -> 
        """

        self.pattern_map = []
        ntotal = len(self.vel_g)*len(self.vel_l)
        progress_bar = tqdm(total=ntotal, desc="Shoham2005 Progress")
        for vs_g in self.vel_g:
            for vs_l in self.vel_l:
                progress_bar.update(1)
                self.Shoham2005_function_point(vs_l, vs_g)

        progress_bar.close()

    def PhenomPatternsMap(self):
        """
        Description
        -------
        The function creates a flow pattern map according to the model specified in the parameters

        Inputs
        ----------
        self.fenomenol :    str -> Models 

        Outputs
        -------
        pattern_map :       int -> 
        """

        if (self.fenomenol.upper().replace(" ","") == "BARNEA1986"):
            self.Barnea1986Map()
        elif (self.fenomenol.upper().replace(" ","") == "SHOHAM2005"):
            self.Shoham2005Map()
        else:
            print('Method not implemented!')
            return 0

class PhenomDataDriven(Patterns):

    def __init__(self, parms):
        """
        Description
        -------
        Initializes the PhenomDataDriven object based on the given parameters and prepares
        the data for the RandomForest classification model.

        Inputs
        ----------
        parms : dict -> 
            Dictionary containing the necessary parameters for initialization, including fluid 
            properties, velocities, and pipe characteristics.
        
        Outputs
        -------
        None -> 
            The processed data is stored as attributes for future use in model training and predictions.
        """

        super().__init__(parms)
        
        prev_vel_l = np.ravel(self.V_l).reshape(self.resol**2,1)
        prev_vel_g = np.ravel(self.V_g).reshape(self.resol**2,1)
        prev_mu_l = np.full((self.resol**2, 1), self.mu_l)
        prev_mu_g = np.full((self.resol**2, 1), self.mu_g)
        prev_rho_l = np.full((self.resol**2, 1), self.rho_l)
        prev_rho_g = np.full((self.resol**2, 1), self.rho_g)
        prev_d = np.full((self.resol**2, 1), self.d)
        prev_alpha = np.full((self.resol**2, 1), self.alpha)
        prev_sigma = np.full((self.resol**2, 1), self.sigma)

        self.X_prev = np.concatenate((prev_vel_l, prev_vel_g, prev_mu_l, 
                                      prev_mu_g, prev_rho_l, prev_rho_g, 
                                      prev_d, prev_alpha, prev_sigma),axis=1)
        
        with pkg_resources.path(Data,'FlowTechPatternData.hdf5') as file:
            with h5py.File(file, "r") as f:
                X_pattern_data_aux = f["XPatternData"][:]
                if (self.fenomenol.upper().replace(" ","") == "BARNEA1986"):
                    Y_pattern_data_aux = f["YPatternDataBarneaHyb"][:]
                elif (self.fenomenol.upper().replace(" ","") == "SHOHAM2005"):
                    Y_pattern_data_aux = f["YPatternDataShohamHyb"][:]
                else:
                    print('Method not implemented!')
                    return 0
        
        # Filter data based on pipe inclination angle
        Tol_Alpha = 180.0
        angle = (self.alpha * 180) / np.pi
        results = abs(X_pattern_data_aux[:, 7] - angle) < Tol_Alpha
        self.X_pattern_data = X_pattern_data_aux[results]
        self.Y_pattern_data = Y_pattern_data_aux[results]
        
        # Encode labels
        self.label_encoder = preprocessing.LabelEncoder()
        self.Y_pattern_data = self.label_encoder.fit_transform(self.Y_pattern_data.ravel())

        self.X_train_pattern_data, self.X_test_pattern_data, self.y_train_pattern_data, self.y_test_pattern_data = model_selection.train_test_split(self.X_pattern_data, self.Y_pattern_data, test_size = 0.01, random_state = 0)
        
        self.sc = preprocessing.StandardScaler()
        self.X_train_pattern_data = self.sc.fit_transform(self.X_train_pattern_data)
        self.X_test_pattern_data = self.sc.transform(self.X_test_pattern_data)
        self.X_prev_rf = self.sc.transform(self.X_prev)

    def RandomForestSVMMap(self, n_estimators=200, criterion='gini', class_weight=None, bootstrap=True, random_state=0):
        """
        Description
        -------
        This function generates a flow pattern map using a RandomForest classifier. It trains the 
        model on the experimental data and then predicts the flow pattern for the input parameters.

        Inputs
        ----------
        n_estimators : int -> 
            Number of trees in the RandomForest classifier. Default is 200.

        criterion : str -> 
            Criterion to measure the quality of a split ('gini' or 'entropy'). Default is 'gini'.

        bootstrap : bool -> 
            Whether to use bootstrap samples when building trees. Default is True.

        random_state : int -> 
            Seed used by the random number generator. Default is 0.

        Outputs
        -------
        pattern_map : numpy.ndarray -> 
            A 2D array representing the predicted flow pattern map, reshaped according to the resolution.
        """

        self.pattern_map = []
        clf_pattern_data = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            criterion=criterion, 
            bootstrap=bootstrap, 
            random_state=random_state
            )
        
        clf_pattern_data.fit(self.X_train_pattern_data, self.y_train_pattern_data)
        y_pred = clf_pattern_data.predict(self.X_prev_rf)

        # feature_importances = clf_pattern_data.feature_importances_

        # for i, importance in enumerate(feature_importances):
        #     print(f"Parameter {i + 1}: {importance}")
        
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X_prev_rf, y_pred, test_size=0.01)
        classifier_model = SVC(kernel='linear', C=1)
        classifier_model.fit(X_train, y_train)
        y_pred = classifier_model.predict(self.X_prev_rf)

        self.pattern_map = np.reshape(y_pred,(self.resol,self.resol))

    def GaussianMap(self):
        """
         Function.

        Parameters
        ----------
        """
        self.pattern = []
        clf_pattern_data = GaussianNB()
        clf_pattern_data.fit(self.X_train_pattern_data, self.y_train_pattern_data)
        y_pred = clf_pattern_data.predict(self.X_prev_rf)
        
        # X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X_prev_rf, y_pred, test_size=0.01)
        # classifier_model = SVC(kernel='linear', C=1)
        # classifier_model.fit(X_train, y_train)
        # y_pred = classifier_model.predict(self.X_prev_rf)
        
        self.pattern = np.reshape(y_pred,(self.resol,self.resol))

    def DecisionTreeClassifierMap(self, criterion='gini', random_state=0):
        """
        Description
        -------
        
        Inputs
        ----------
        
        Outputs
        -------
        
        """
        self.pattern_map = []
        clf_pattern_data = DecisionTreeClassifier(
            criterion=criterion, 
            random_state=random_state
            )
        
        clf_pattern_data.fit(self.X_train_pattern_data, self.y_train_pattern_data)
        y_pred = clf_pattern_data.predict(self.X_prev_rf)
        
        self.pattern_map = np.reshape(y_pred,(self.resol,self.resol))

    def RandomForestMap(self, n_estimators=200, criterion='gini', class_weight=None, bootstrap=True, random_state=0):
        """
        Description
        -------
        
        Inputs
        ----------
        
        Outputs
        -------
        
        """
        self.pattern_map = []
        clf_pattern_data = RandomForestClassifier(
            n_estimators=n_estimators, 
            criterion=criterion,
            class_weight=class_weight,
            bootstrap=bootstrap, 
            random_state=random_state
            )
        
        clf_pattern_data.fit(self.X_train_pattern_data, self.y_train_pattern_data)
        y_pred = clf_pattern_data.predict(self.X_prev_rf)
        
        self.pattern_map = np.reshape(y_pred,(self.resol,self.resol))
    
    def KNNMap(self):
        """
        Description
        -------
        
        Inputs
        ----------
        
        Outputs
        -------
        
        """
        
        self.pattern = []
        clf_pattern_data = KNeighborsClassifier(
            n_neighbors=5,
            metric='minkowski', 
            p = 2
            )
        
        clf_pattern_data.fit(self.X_train_pattern_data, self.y_train_pattern_data)
        y_pred = clf_pattern_data.predict(self.X_prev_rf)
        
        # X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X_prev_rf, y_pred, test_size=0.01)
        # classifier_model = SVC(kernel='linear', C=1)
        # classifier_model.fit(X_train, y_train)
        # y_pred = classifier_model.predict(self.X_prev_rf)
        
        self.pattern = np.reshape(y_pred,(self.resol,self.resol))
    
    def LogisticRegressionMap(self):
        """
        Description
        -------
        
        Inputs
        ----------
        
        Outputs
        -------
        
        """

        self.pattern = []
        clf_pattern_data = LogisticRegression(
            n_neighbors=5, 
            metric='minkowski', 
            p = 2
            )
        
        clf_pattern_data.fit(self.X_train_pattern_data, self.y_train_pattern_data)
        y_pred = clf_pattern_data.predict(self.X_prev_rf)
        
        # X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X_prev_rf, y_pred, test_size=0.01)
        # classifier_model = SVC(kernel='linear', C=1)
        # classifier_model.fit(X_train, y_train)
        # y_pred = classifier_model.predict(self.X_prev_rf)
        
        self.pattern = np.reshape(y_pred,(self.resol,self.resol))
    
    def SVMMap(self):
        """
        Description
        -------
        
        Inputs
        ----------
        
        Outputs
        -------
        
        """

        self.pattern = []
        clf_pattern_data = SVC(kernel='linear', random_state=1, C = 2.0)
        clf_pattern_data.fit(self.X_train_pattern_data, self.y_train_pattern_data)
        y_pred = clf_pattern_data.predict(self.X_prev_rf)
        
        # X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X_prev_rf, y_pred, test_size=0.01)
        # classifier_model = SVC(kernel='linear', C=1)
        # classifier_model.fit(X_train, y_train)
        # y_pred = classifier_model.predict(self.X_prev_rf)
        
        self.pattern = np.reshape(y_pred,(self.resol,self.resol))
    
    def MLPCMap(self):
        """
        Description
        -------
        
        Inputs
        ----------
        
        Outputs
        -------
        
        """

        self.pattern = []
        clf_pattern_data = MLPClassifier(
            hidden_layer_sizes=(20,20,20), 
            solver= 'lbfgs',
            max_iter=2000,
            random_state=0
            )
        
        clf_pattern_data.fit(self.X_train_pattern_data, self.y_train_pattern_data)
        y_pred = clf_pattern_data.predict(self.X_prev_rf)

        # X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X_prev_rf, y_pred, test_size=0.01)
        # classifier_model = SVC(kernel='linear', C=1)
        # classifier_model.fit(X_train, y_train)
        # y_pred = classifier_model.predict(self.X_prev_rf)
        
        self.pattern = np.reshape(y_pred,(self.resol,self.resol))

    def PhenomDataDrivenPatternsMap(self):
        if (self.data_driven.upper().replace(" ", "") == 'RANDOMFOREST'):
            self.RandomForestMap()
        elif (self.data_driven.upper().replace(" ", "") == 'MLPC'):
            self.MLPCMap()
        elif (self.data_driven.upper().replace(" ", "") == 'KNN'):
            self.KNNMap()
        elif (self.data_driven.upper().replace(" ", "") == 'SVM'):
            self.SVMMap()
        elif (self.data_driven.upper().replace(" ", "") == 'GAUSSIANNB'):
            self.GaussianMap()
        elif (self.data_driven.upper().replace(" ", "") == 'DECISIONTREE'):
            self.DecisionTreeClassifierMap()
        elif (self.data_driven.upper().replace(" ", "") == 'LogisticRegression'):
            self.LogisticRegressionMap()
        else:
            print('Method not implemented!')
            return 0

    ########################################
    def DecisionTreeOptmizeHyperParms(self):
        """
        Description
        -------
        Inputs
        ----------

        Outputs
        -------

        """
        param_grid = {
              'criterion': ['gini', 'entropy', 'log_loss'],
              'splitter': ['best', 'random'],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              'max_features': [None, 'auto', 'sqrt', 'log2'],
              }
        
        clf_pattern_data = DecisionTreeClassifier()
        print(30*'*+')
        grid_search = GridSearchCV(estimator=clf_pattern_data, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
        
        grid_search.fit(self.X_pattern_data, self.Y_pattern_data)
        
        # Get the best hyperparameters
        best_params = grid_search.best_params_
        best_results = grid_search.best_score_
        print(f"Test Method: {clf_pattern_data}")
        print(f"Best parameters: {best_params}")
        print(f"Best results: {best_results}")

    def RandomForestMapOptmizeHyperParms(self):
        """
        Description
        -------
        This function optimizes hyperparameters for a RandomForest classifier using GridSearchCV (or RandomizedSearchCV) 
        and then generates a flow pattern map based on the optimized model. It also optionally refines predictions using SVC.

        Inputs
        ----------
        random_state : int -> 
            Seed used by the random number generator. Default is 0.

        Outputs
        -------
        pattern_map : numpy.ndarray -> 
            A 2D array representing the predicted flow pattern map, reshaped according to the resolution.
        """
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'bootstrap': [True],
            'random_state': [0, 1],
            'class_weight': [None, 'balanced'],
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        clf_pattern_data = RandomForestClassifier()

        print(30*'*+')
        grid_search = GridSearchCV(estimator=clf_pattern_data, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(self.X_pattern_data, self.Y_pattern_data)
        
        # Get the best hyperparameters
        best_params = grid_search.best_params_
        best_results = grid_search.best_score_
        print(f"Test Method: {clf_pattern_data}")
        print(f"Best parameters: {best_params}")
        print(f"Best results: {best_results}")

    def KNNOptmizeHyperParms(self):
        """
        Description
        -------
        Inputs
        ----------

        Outputs
        -------

        """
        param_grid = {
              'n_neighbors': [3, 5, 10, 20],
              'p': [1, 2]
              }

        clf_pattern_data = KNeighborsClassifier()
        print(30*'*+')
        grid_search = GridSearchCV(estimator=clf_pattern_data, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(self.X_pattern_data, self.Y_pattern_data)
        
        # Get the best hyperparameters
        best_params = grid_search.best_params_
        best_results = grid_search.best_score_
        print(f"Test Method: {clf_pattern_data}")
        print(f"Best parameters: {best_params}")
        print(f"Best results: {best_results}")

    def LogisticRegressionOptmizeHyperParms(self):
        """
        Description
        -------
        Inputs
        ----------

        Outputs
        -------

        """
        param_grid = {
              'tol': [0.0001, 0.00001, 0.000001],
              'C': [1.0, 1.5, 2.0],
              'solver': ['lbfgs', 'sag', 'saga']
              }

        clf_pattern_data = LogisticRegression()
        print(30*'*+')
        grid_search = GridSearchCV(estimator=clf_pattern_data, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(self.X_pattern_data, self.Y_pattern_data)
        
        # Get the best hyperparameters
        best_params = grid_search.best_params_
        best_results = grid_search.best_score_
        print(f"Test Method: {clf_pattern_data}")
        print(f"Best parameters: {best_params}")
        print(f"Best results: {best_results}")

    def SVMOptmizeHyperParms(self):
        """
        Description
        -------
        Inputs
        ----------

        Outputs
        -------

        """
        param_grid = {
              'tol': [0.001, 0.0001, 0.00001],
              'C': [1.0, 1.5, 2.0],
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
              }

        clf_pattern_data = SVC()
        print(30*'*+')
        grid_search = GridSearchCV(estimator=clf_pattern_data, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(self.X_pattern_data, self.Y_pattern_data)
        
        # Get the best hyperparameters
        best_params = grid_search.best_params_
        best_results = grid_search.best_score_
        print(f"Test Method: {clf_pattern_data}")
        print(f"Best parameters: {best_params}")
        print(f"Best results: {best_results}")

    def MLPCOptmizeHyperParms(self):
        """
        Description
        -------
        Inputs
        ----------

        Outputs
        -------

        """
        param_grid = {
              'hidden_layer_sizes' : [(20,20,20), (20,20)],
              'activation': ['relu', 'logistic'],
              'solver': ['adam', 'sgd'],
              'batch_size': [10, 56]
              }

        clf_pattern_data = MLPClassifier()
        print(30*'*+')
        grid_search = GridSearchCV(estimator=clf_pattern_data, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(self.X_pattern_data, self.Y_pattern_data)
        
        # Get the best hyperparameters
        best_params = grid_search.best_params_
        best_results = grid_search.best_score_
        print(f"Test Method: {clf_pattern_data}")
        print(f"Best parameters: {best_params}")
        print(f"Best results: {best_results}")

class PhenomDataDrivenHybrid(Patterns):

    def __init__(self, parms, aux_pattern_hybrid=[]):
        """
        Description
        -------
        Initializes the PhenomDataDrivenHybrid object, integrating both experimental and 
        phenomenological model data for flow pattern prediction.

        Inputs
        ----------
        parms : dict -> 
            Dictionary containing the necessary parameters for initialization, including fluid 
            properties, velocities, and pipe characteristics.

        aux_pattern_hybrid : list, optional -> 
            List containing initial hybrid patterns from the phenomenological model. Default is an empty list.
        
        Outputs
        -------
        None -> 
            The processed data is stored as attributes for future model training and predictions.
        """

        super().__init__(parms)
        self.Phenom = Phenom(parms)
        self.Phenom.__init__(parms)
        self.PhenomDataDriven = PhenomDataDriven(parms)
        self.PhenomDataDriven.__init__(parms)

        # Load hybrid pattern from phenomenological model if not provided
        if (len(aux_pattern_hybrid) == 0):
            self.Phenom.PhenomPatterns()
            aux_pattern_hybrid = self.Phenom.pattern_map
        
        # Load experimental pattern data from HDF5 file
        with pkg_resources.path(Data,'FlowTechPatternData.hdf5') as file:
            with h5py.File(file, "r") as f:
                if (self.fenomenol.upper().replace(" ","") == "BARNEA1986"):
                    self.Y_pattern_hybrid = f["YPatternDataBarneaHyb"][:]
                elif (self.fenomenol.upper().replace(" ","") == "SHOHAM2005"):
                    self.Y_pattern_hybrid = f["YPatternDataShohamHyb"][:]
                else:
                    print('Method not implemented!')
                    return 0
        
        # Prepare auxiliary pattern hybrid
        aux_pattern_hybrid = np.asarray(aux_pattern_hybrid).reshape(self.resol**2, 1)
 
        # Concatenate hybrid data with experimental pattern data
        self.X_pattern_hybrid = deepcopy(self.PhenomDataDriven.X_pattern_data)
        self.X_pattern_hybrid = np.concatenate((self.X_pattern_hybrid, self.Y_pattern_hybrid), axis=1)

        # Encode the labels
        self.label_encoder = LabelEncoder()
        self.Y_pattern_hybrid = self.label_encoder.fit_transform(self.Y_pattern_hybrid.ravel())
        
        # Concatenate predicted values for previous data
        self.X_prev_hybrid = np.concatenate((self.PhenomDataDriven.X_prev, aux_pattern_hybrid), axis=1)

        # Apply OneHotEncoding to categorical data
        self.ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [9])], remainder='passthrough')
        self.X_pattern_hybrid = np.array(self.ct.fit_transform(self.X_pattern_hybrid))
        self.X_prev_hybrid = np.array(self.ct.transform(self.X_prev_hybrid))

        # Split hybrid data into training and test sets
        self.X_train_pattern_hybrid, self.X_test_pattern_hybrid, self.y_train_pattern_hybrid, self.y_test_pattern_hybrid = train_test_split(
            self.X_pattern_hybrid, self.PhenomDataDriven.Y_pattern_data, test_size=0.1, random_state=0)

        # Scale the data
        self.sc = StandardScaler()
        self.X_train_pattern_hybrid = self.sc.fit_transform(self.X_train_pattern_hybrid)
        self.X_test_pattern_hybrid = self.sc.transform(self.X_test_pattern_hybrid)
        self.X_prev_hybrid = self.sc.transform(self.X_prev_hybrid)

    def HybridRandomForestMap(self, n_estimators=80, criterion='gini', class_weight=None, bootstrap=False, random_state=0):
        """
        Description
        -------
        Trains a hybrid RandomForest model that integrates both experimental data and data 
        predicted by the phenomenological model to generate a flow pattern map.

        Inputs
        ----------
        None -> 
            No direct inputs are required, as this function operates on the preprocessed data 
            stored as attributes of the object.
        
        Outputs
        -------
        pattern_map : numpy.ndarray -> 
            A 2D array representing the predicted flow pattern map, reshaped according to the resolution.
        """

        self.pattern_map = []
        
        # Initialize and train RandomForestClassifier
        clf_pattern_hybrid = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            bootstrap=bootstrap,
            random_state=random_state,
            class_weight=class_weight
        )
        clf_pattern_hybrid.fit(self.X_train_pattern_hybrid, self.y_train_pattern_hybrid)
        
        # Predict flow patterns using the trained RandomForest model
        y_pred_hybrid = clf_pattern_hybrid.predict(self.X_prev_hybrid)

        feature_importances = clf_pattern_hybrid.feature_importances_

        # for i, importance in enumerate(feature_importances):
        #     print(f"Parameter (hybrid): {i + 1}: {importance}")

        # Convert predicted labels back to original classes
        self.pattern_map = self.label_encoder.inverse_transform(y_pred_hybrid)

        # Split data for SVC refinement
        X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid = train_test_split(
            self.X_prev_hybrid, y_pred_hybrid, test_size=0.01)

        # Refine predictions using Support Vector Classifier
        classifier_model = SVC(kernel='linear', C=1)
        classifier_model.fit(X_train_hybrid, y_train_hybrid)
        y_pred_hybrid_svc = classifier_model.predict(self.X_prev_hybrid)

         # Reshape predictions into a 2D pattern map
        self.pattern_map = np.reshape(y_pred_hybrid_svc, (self.Phenom.resol, self.Phenom.resol))
        
        return self.pattern_map

    def RandomForestHybridMapOptmizeParms(self):
        """
        Description
        -------
        This function optimizes hyperparameters for a RandomForest classifier using GridSearchCV 
        and generates a flow pattern map based on the optimized model. It also optionally refines predictions using SVC.

        Inputs
        ----------
        None -> 
            No direct inputs, uses the preprocessed data stored as attributes.

        Outputs
        -------
        pattern_map : numpy.ndarray -> 
            A 2D array representing the predicted flow pattern map, reshaped according to the resolution.

        Notes
        -------
        - Performs hyperparameter optimization using GridSearchCV.
        - Then trains a RandomForest model using the best parameters and evaluates model performance using F1 score, precision, recall, and accuracy.
        """
        self.pattern_map = []

        clf_pattern_data = RandomForestClassifier(
            n_estimators=200,
            criterion='gini',
            bootstrap=True,
            random_state=0,
            class_weight='balanced'  # Handle class imbalance
        )

        # Hyperparameter optimization with GridSearchCV
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'bootstrap': [True, False],
            'random_state': [0, 1],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8]
        }

        grid_search = GridSearchCV(estimator=clf_pattern_data, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(self.X_train_pattern_hybrid, self.y_train_pattern_hybrid)

        best_params = grid_search.best_params_
        # print(f"Best parameters: {best_params}")

        clf_pattern_data.set_params(**best_params)
        clf_pattern_data.fit(self.X_train_pattern_hybrid, self.y_train_pattern_hybrid)

        y_pred_rf = clf_pattern_data.predict(self.X_test_pattern_hybrid)

        # Avaliar métricas
        f1 = f1_score(self.y_test_pattern_hybrid, y_pred_rf, average='weighted')
        precision = precision_score(self.y_test_pattern_hybrid, y_pred_rf, average='weighted')
        recall = recall_score(self.y_test_pattern_hybrid, y_pred_rf, average='weighted')
        accuracy = accuracy_score(self.y_test_pattern_hybrid, y_pred_rf)

        # print(f"F1 Score: {f1}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}")

        classifier_model = SVC(kernel='linear', C=1)
        classifier_model.fit(self.X_test_pattern_hybrid, y_pred_rf)
        y_pred_rf_svc = classifier_model.predict(self.X_prev_hybrid)

        self.pattern_map = np.reshape(y_pred_rf_svc, (self.Phenom.resol, self.Phenom.resol))

        return self.pattern_map
   
    def HybridKNNMap(self):
        """
         Function.

        Parameters
        ----------
        """
        self.pattern = []
        clf_pattern_hybrid = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
        clf_pattern_hybrid.fit(self.X_train_pattern_hybrid, self.y_train_pattern_hybrid)
        KNeighborsClassifier()
        y_pred_hybrid = clf_pattern_hybrid.predict(self.X_prev_hybrid)
        
        # X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid = model_selection.train_test_split(self.X_prev_hybrid, y_pred_hybrid, test_size=0.01)
        # classifier_model = SVC(kernel='linear', C=1)
        # classifier_model.fit(X_train_hybrid, y_train_hybrid)
        # y_pred_hybrid = classifier_model.predict(self.X_prev_hybrid)
        
        self.pattern = np.reshape(y_pred_hybrid,(self.Phenom.resol,self.Phenom.resol))

    def HybridSVMMap(self):
        """
        Function.
        Parameters
        ----------
        """
        self.pattern = []
        clf_pattern_hybrid = SVC(kernel='linear', random_state=1, C = 2.0)
        clf_pattern_hybrid.fit(self.X_train_pattern_hybrid, self.y_train_pattern_hybrid)
        SVC(C=2.0, random_state=1)
        y_pred_hybrid = clf_pattern_hybrid.predict(self.X_prev_hybrid)
        
        # X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid = model_selection.train_test_split(self.X_prev_hybrid, y_pred_hybrid, test_size=0.01)
        # classifier_model = SVC(kernel='linear', C=1)
        # classifier_model.fit(X_train_hybrid, y_train_hybrid)
        # y_pred_hybrid = classifier_model.predict(self.X_prev_hybrid)
        
        self.pattern = np.reshape(y_pred_hybrid,(self.Phenom.resol,self.Phenom.resol))

    def HybridMLPCMap(self):
        """
        HybridMLPC Function.

        Parameters
        ----------
        """
        self.pattern = []
        clf_pattern_hybrid = MLPClassifier(hidden_layer_sizes=(20,20,20), solver= 'lbfgs',max_iter=2000,random_state=0)
        clf_pattern_hybrid.fit(self.X_train_pattern_hybrid, self.y_train_pattern_hybrid)
        y_pred_hybrid = clf_pattern_hybrid.predict(self.X_prev_hybrid)
        
        # X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid = model_selection.train_test_split(self.X_prev_hybrid, y_pred_hybrid, test_size=0.01)
        # classifier_model = SVC(kernel='linear', C=1)
        # classifier_model.fit(X_train_hybrid, y_train_hybrid)
        # y_pred_hybrid = classifier_model.predict(self.X_prev_hybrid)
        
        self.pattern = np.reshape(y_pred_hybrid,(self.Phenom.resol,self.Phenom.resol))

    def HybridGaussianMap(self):
        """
        Function.

        Parameters
        ----------
        """
        self.pattern = []
        clf_pattern_hybrid = GaussianNB()
        clf_pattern_hybrid.fit(self.X_train_pattern_hybrid, self.y_train_pattern_hybrid)
        GaussianNB()
        y_pred_hybrid = clf_pattern_hybrid.predict(self.X_prev_hybrid)
        
        # X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid = model_selection.train_test_split(self.X_prev_hybrid, y_pred_hybrid, test_size=0.01)
        # classifier_model = SVC(kernel='linear', C=1)
        # classifier_model.fit(X_train_hybrid, y_train_hybrid)
        # y_pred_hybrid = classifier_model.predict(self.X_prev_hybrid)
        
        self.pattern = np.reshape(y_pred_hybrid,(self.resol,self.resol))
    
    def PhenomDataDrivenHybridPatternsMap(self):

        if (self.data_driven.upper().replace(" ", "") == 'RANDOMFOREST'):
            self.HybridRandomForestMap()
        elif (self.data_driven.upper().replace(" ", "") == 'MLPC'):
            self.HybridMLPCMap()
        elif (self.data_driven.upper().replace(" ", "") == 'KNN'):
            self.HybridKNNMap()
        elif (self.data_driven.upper().replace(" ", "") == 'SVM'):
            self.HybridSVMMap()
        elif (self.data_driven.upper().replace(" ", "") == 'GAUSSIANNB'):
            self.HybridGaussianMap()
        else:
            print('Method not implemented!')
            return 0
        
class PhenomLiqLiq(Patterns):

    def __init__(self, parms):
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
        super().__init__(parms)

        # Definicao das constantes globais

        # self.DENW = float(1070.1)
        # self.DENO = float(831.3)
        # self.VISW = float(0.00076)
        # self.VISO = float(0.00734)
        # self.DI = float(0.08280)

        self.DENW = self.rho_l
        self.DENO = self.rho_g
        self.VISW = self.mu_l
        self.VISO = self.mu_g
        self.DI = self.d
        
        self.EPW = float(0.00001)
        
        self.SUROW = float(0.036)

        self.ISHEL = 0

        self.ANG = self.alpha

    # Funcao EQU
    def EQU(self, VWS, VOS, HLD):
        # IOERR = []
        # # Validacao dos parametros
        # if VWS < 0 or VOS < 0:
        #     IOERR.append('EQU: Illegal value input: VWS or VOS')
        #     return None
        # elif HLD < 0:
        #     IOERR.append('EQU: Illegal value input for HLD')
        #     return None
        
        AA = 2.0 * HLD - 1.0
        AW = 0.25 * self.DI**2 * (math.pi - math.acos(AA) + AA * math.sqrt(1.0 - AA**2))
        A = (math.pi / 4.0) * self.DI**2
        AO = A - AW
        VW = VWS * A / AW
        VO = VOS * A / AO
        SW = self.DI * (math.pi - math.acos(AA))
        SO = math.pi * self.DI - SW
        SI = self.DI * math.sqrt(1.0 - AA**2)

        # Definicao dos diametros hidraulicos com base na fase mais rapida
        if VO >= VW:
            DW = 4.0 * AW / SW
            DOIL = 4.0 * AO / (SI + SO)
        else:
            DW = 4.0 * AW / (SI + SW)
            DOIL = 4.0 * AO / SO

        REW = abs(VW) * DW * self.DENW / self.VISW
        REO = abs(VO) * DOIL * self.DENO / self.VISO
        
        # A fase mais rapida determina o fator de atrito interfacial
        if VO >= VW:
            REI = copy.copy(REO)
            ROI = copy.copy(self.DENO)
        else:
            REI = copy.copy(REW)
            ROI = copy.copy(self.DENW)
        
        # Determinacao do esforco de cisalhamento

        FW = self.FFF(REW, DW)
        # if FW == None:
        #     IOERR.append('Error returned from FFF call')
        #     return None
        
        FO = self.FFF(REO, DOIL)
        # if FO == None:
        #     IOERR.append('Error returned from FFF call')
        #     return None

        if VO >= VW:
            FI = copy.copy(FO)
        else:
            FI = copy.copy(FW)

        TAUW = 0.5 * FW * self.DENW * VW**2
        TAUO = 0.5 * FO * self.DENO * VO**2
        TAUI = 0.5 * FI * ROI * (VO - VW) * abs(VO - VW)

        return (AA, AW, A, AO, VW, VO, SW, SO, SI, DW, DOIL, REW, REO, FW, FI, FO, TAUW, TAUO, TAUI)

    # Funcao FF
    def FF(self, VWS, VOS, HLD):
        # IOERR = []
        # if VWS < 0 or VOS < 0:
        #     IOERR.append('FF: Illegal value input: VWS or VOS')
        #     return None
        results = self.EQU(VWS, VOS, HLD)
        # if results == None:
        #     IOERR.append('FF: Error returned from EQU call')
        #     return None
        # Unpack the results
        (AA, AW, A, AO, VW, VO, SW, SO, SI, DW, DOIL, REW, REO, FW, FI, FO, TAUW, TAUO, TAUI) = copy.copy(results)
        # Calculating the value of FF
        FF_value = -TAUW * SW / AW + TAUO * SO / AO + TAUI * (SI / AW + SI / AO) - (self.DENW - self.DENO) * self.grav * math.sin(self.ANG)

        return FF_value

    # Funcao FFF
    def FFF(self, RE, DHYD):
        # IOERR = []
        # # Validacao dos parametros
        # if RE < 0:
        #     IOERR.append('FFF: Illegal value input for RE')
        #     return None
        # Definicao de AT e BT com base no valor de RE
        if RE <= 1502.0:
            AT = 1.0
            BT = 0.0
        else:
            AT = 0.0
            BT = 1.0
        # Calculo da funcao FFF
        FFF_value = AT*16*RE**(-1)+BT*(-3.6*math.log10((6.9/RE) +(self.EPW/(3.7*DHYD))**1.1))**(-2.0)

        return FFF_value
    
    # Funcao HDF
    def HDF(self, VWS, VOS, HLD1):
        EPS = 0.000001
        # if VWS < 0 or VOS < 0.0:
        #     IOERR.append('HDF : Illegal value input for VWS')
        #     IOERR.append('HDF : Illegal value input for VOS')
        #     return None
        NMAX = 100
        N = 0
        X = HLD1 + 10 * EPS
        DX = (1.0 - X - EPS) / 50.0
        XMAX = 1.0

        while True:
            F1 = self.FF(VWS, VOS, X)
            X = X + DX

            if X >= XMAX:
                break

            HLD = copy.copy(X)
            results = self.EQU(VWS, VOS, HLD)
            # if results is None:
            #     IOERR.append('Error returned from EQU call')
            #     return None
            (AA, AW, A, AO, VW, VO, SW, SO, SI, DW, DOIL, REW, REO, FW, FI, FO, TAUW, TAUO, TAUI) = copy.copy(results)
            
            F = self.FF(VWS, VOS, HLD)

            A11 = -TAUW*SW/AW
            A12 = TAUO*SO/AO
            A13 = TAUI*(SI/AW+SI/AO)
            A14 = -(self.DENW - self.DENO) * self.grav * math.sin(self.ANG)
            
            if abs(DX) < EPS:
                break

            N += 1
            if N == 1:
                continue
            elif N > NMAX:
                print('Subroutine HDF did not converge')
                return None

            SIGN = F * F1
            PP = F * F1
            if abs(SIGN) < 1e-12:
                break
            elif SIGN < 0.0:
                DX = - DX / 2.0
            else:
                continue

        return HLD

    # Funcao principal
    def Trallero1995Map(self):
        # Inicializacao das variaveis e listas
        HLD = 0.0
        VM = 0.0
        CW = 0.0
        R = 0.0
        # if self.DI <= 0.0 or self.DENW < 0.0 or \
        #    self.EPW < 0.0 or self.DENO < 0.0 or \
        #    self.VISW < 0.0 or self.VISO < 0.0 or \
        #    self.ISHEL < 0 or self.ISHEL > 1:
        #     IOERR.append('Illegal value input: DI or DENW or EPW or DENO or VISW or VISO or ISHEL')
        #     for err in IOERR:
        #         print(err)
        #     exit(999)
        # Calculos preliminares
        ALAMD = 100 * self.DI
        AK = 2 * math.pi / ALAMD

        self.pattern_map = []
        for VOS in self.vel_l:
            for VWS in self.vel_g:
                # Calculos das velocidades e fracoes
                VM = VWS + VOS
                CW = VWS / VM
                R = VWS / VOS

                HLD1 = 0.0
                HLDMAX = 0.999

                for I in range(1,3):
                    # Chamada para a funcao HDF
                    HLDN = self.HDF(VWS, VOS, HLD1)

                    # if HLDN == None:
                    #     exit(999)

                    if HLDN >= HLDMAX:
                        break

                    HLD = copy.copy(HLDN)

                    # Chamada para a funcao EQU
                    results = self.EQU(VWS, VOS, HLD)

                    # Processamento dos resultados retornados pela EQU
                    (AA, AW, A, AO, VW, VO, SW, SO, SI, DW, DOIL, REW, REO, FW, FI, FO, TAUW, TAUO, TAUI) = copy.copy(results)

                    RW = AW / A
                    RO = AO / A
                    RWH = VWS / (VWS + VOS)
                    ROH = 1 - RWH

                    # Calculos de gradientes de pressao
                    PGOT = (TAUO * SO + TAUI * SI + self.DENO * AO * self.grav * math.sin(self.ANG)) / AO
                    PGWT = (TAUW * SW - TAUI * SI + self.DENW * AW * self.grav * math.sin(self.ANG)) / AW
                    DENM = RWH * self.DENW + ROH * self.DENO
                    VISM = RWH * self.VISW + ROH * self.VISO
                    REM = (DENM * VM * self.DI) / VISM
                    FM = 0.312 / REM**0.25

                    PGH = ((FM * DENM * VM**2) / (2 * self.DI)) + DENM * self.grav * math.sin(self.ANG)

                    # Derivada da equacao de momentum combinada
                    HLD1 = copy.copy(HLD)

                    DEN = self.DENW / RW + self.DENO / RO

                    E = (-((self.FF(VWS, VOS, HLD + 0.0001/2)
                        - self.FF(VWS, VOS, HLD - 0.0001/2))
                        / 0.0001) * math.pi / (4 * math.sqrt(1 - AA**2))) / DEN

                    B = (((self.FF(VWS + (VWS * 0.001)/2, VOS, HLD)
                        - self.FF(VWS - (VWS * 0.001)/2, VOS, HLD))
                        / (VWS * 0.001))
                        - ((self.FF(VWS, VOS + (VOS * 0.001)/2, HLD)
                            - self.FF(VWS, VOS - (VOS * 0.001)/2, HLD))
                            / (VOS * 0.001))) / (2 * DEN)

                    # Contribuicao de sombreamento (sheltering)
                    if self.ISHEL == 1:
                        CS = 0.0730
                    else:
                        CS = 0.0

                    if VO >= VW:
                        ROI = copy.copy(self.DENO)
                    else:
                        ROI = copy.copy(self.DENW)

                    SHEL = ROI * pow((VO - VW), 2) * CS * A * (1/AW + 1/AO) / DEN
                    # Calculo dos criterios de estabilidade
                    FUSS = pow((E/(2*B) - ((self.DENW * VW / RW + self.DENO * VO / RO) / DEN)), 2)
                    GST = 1 / DEN * (self.DENW - self.DENO) * self.grav * math.cos(self.ANG) * A / SI
                    AUST = self.DENW * self.DENO * pow((VO - VW), 2) / (pow(DEN,2) * RW * RO)
                    SIST = self.SUROW * A * pow(AK, 2) / (SI * DEN)
                    CRFV = FUSS + AUST - GST - SIST + SHEL
                    CRFI = AUST - GST - SIST

                    # Calculo do tamanho maximo de gotas de agua
                    RE = VM * self.DI * self.DENO / self.VISO

                    FD = self.FFF(RE, self.DI)

                    # if FD == None:
                    #     exit(999)

                    EP = 2 * FD * pow(VM, 3) / self.DI
                    SK = (pow((pow((self.VISO / self.DENO), 3) / EP), 0.25))
                    DM = 0.73 * (self.SUROW / self.DENO)**0.6 * EP**(-0.4)
                    if DM / SK < 2:
                        DM = 1.0 * self.SUROW * SK / (self.VISO * (((self.VISO / self.DENO) * EP)**0.25))

                    DM = 592.620 * DM * CW**1.832
                    VOMH = math.sqrt((8/3) * DM * self.grav * math.cos(self.ANG) * (self.DENW / self.DENO - 1) / FD)

                    RE = VM*self.DI*self.DENW/self.VISW 
                    
                    FD = self.FFF(RE, self.DI)

                    # if FD == None:
                    #     exit(999)
                    
                    EP = 2.*FD*VM**3./self.DI                                          
                    DM = 1.50*(0.73*(self.SUROW/self.DENW)**0.6*\
                        EP**(-0.4))/CW**3.5         
                    VWMH = math.sqrt((8./3.)*DM*self.grav*\
                        math.cos(self.ANG)*(1-self.DENO/self.DENW)/FD)
                    
                    EP = 2.0*FO*VO**3./DOIL                                           
                    SK = ((self.VISO/self.DENO)**3./EP)**0.25                                   
                    DM = 0.73*(self.SUROW/self.DENO)**0.6*EP**(-0.4)                            
                    
                    if DM / SK < 2.0:
                        DM = 1.0 * self.SUROW * SK / (self.VISO * (((self.VISO / self.DENO) * EP)**0.25)) 
                    
                    DM = 0.68250*DM                                                   
                    VOOH = math.sqrt((8./3.)*DM*self.grav*math.cos(self.ANG)*(self.DENW/self.DENO-1)/FO )            

                    EP = 2.*FW*VW**3./DW                                    
                    DM = 0.73*(self.SUROW/self.DENW)**0.6*EP**(-0.4)                  
                    DM = 11.3250*DM*CW**2.                               
                    VWWH = math.sqrt( (8./3.)*DM*self.grav*math.cos(self.ANG)*(1-self.DENO/self.DENW)/FW )
                    
                    DM   = 2.*math.sqrt(self.SUROW*(self.VISW/self.DENW)/(25*self.DENW*VW**3.*(FW/2)**1.5))
                    DMP  = 25.*self.DENW*(self.VISW/self.DENW)**2/self.SUROW                           

                    if DM < DMP:
                        DM = copy.copy(DMP)                                  
                    
                    DM   = 1.740*DM/CW**7                                       
                    VWWL = math.sqrt((8.0/3.0)*DM*self.grav*\
                           math.cos(self.ANG)*(1.0-self.DENO/self.DENW)/FW )
                    
                    if CRFI < 0:
                        flo_pat = 'ST&MI'
                        if CRFV < 0:
                            flo_pat = 'ST'
                            if abs(self.ANG-0.0) < 1e-12 and abs(VW - VO) < 0.000010:
                                flo_pat = 'ST&MI'
                        if VO >= VOOH and VW >= VWWH:
                            flo_pat = 'DW_O&DO_W'
                        if VO >= VOMH:
                            flo_pat = 'W_O'
                    else:
                        if VW < VWMH and VW > VO:
                            flo_pat = 'DO_W&W'
                        if VW < VWWL and VW > VO:
                            flo_pat = 'ST&MI'
                        if VO >= VOOH and VW >= VWWH:
                            flo_pat = 'DW_O&DO_W'
                        if VW >= VWMH and VW > VO:
                            flo_pat = 'O_W'
                        if VO >= VOMH and VW < VO:
                            flo_pat = 'W_O'
                    
                self.pattern_map.append(self.flow_dic_trallero[flo_pat])

                # BVOS = math.log10(VOS)
                # BVWS = math.log10(VWS)
                # PGOTL = math.log10(PGOT)
                # PGHL = math.log10(PGH)

        self.pattern = np.reshape(self.pattern_map,(self.resol,self.resol))

    def __init__(self, parms, aux_pattern_hybrid=[], data_weight=1.0, model_weight=1.0):
        """
        Initializes the PhenomDataDrivenHybrid object, which integrates both 
        experimental and phenomenological model data for flow pattern prediction.

        Inputs
        ----------
        parms : dict 
            Dictionary containing the parameters required for the initialization of the Patterns class.
        
        aux_pattern_hybrid : list, optional
            List containing initial hybrid patterns from the phenomenological model. Default is an empty list.
        
        data_weight : float, optional
            Weight assigned to the experimental data for the RandomForest model. Default is 1.0.
        
        model_weight : float, optional
            Weight assigned to the data predicted by the phenomenological model. Default is 1.0.
        
        Outputs
        -------
        None
            The processed data is stored as attributes of the object for further model training and predictions.
        
        Notes
        -------
        - This constructor initializes the PhenomDataDriven and Phenom classes and prepares the hybrid dataset, 
          consisting of both experimental and phenomenological data, for training and prediction.
        - OneHotEncoding is applied to categorical features and data is scaled using StandardScaler.
        """
        super().__init__(parms)
        self.data_weight = data_weight
        self.model_weight = model_weight

        # Initialize Phenom and PhenomDataDriven classes
        self.Phenom = Phenom(parms)
        self.Phenom.__init__(parms)
        self.PhenomDataDriven = PhenomDataDriven(parms)
        self.PhenomDataDriven.__init__(parms)

        if len(aux_pattern_hybrid) == 0:
            self.Phenom.PhenomPatterns()
            aux_pattern_hybrid = self.Phenom.pattern_map
        
        # Load hybrid experimental data
        with pkg_resources.path(Data, 'FlowTechPatternData.hdf5') as file:
            with h5py.File(file, "r") as f:
                if self.fenomenol.upper().replace(" ", "") == "BARNEA1986":
                    self.Y_pattern_hybrid = f["YPatternDataBarneaHyb"][:]
                elif self.fenomenol.upper().replace(" ", "") == "SHOHAM2005":
                    self.Y_pattern_hybrid = f["YPatternDataShohamHyb"][:]
                else:
                    print('Method not implemented!')
                    return

        # Prepare the auxiliary pattern hybrid data for model training
        aux_pattern_hybrid = np.asarray(aux_pattern_hybrid).reshape(self.resol**2, 1)

        # Concatenate experimental and phenomenological data
        self.X_pattern_hybrid = deepcopy(self.PhenomDataDriven.X_pattern_data)
        self.X_pattern_hybrid = np.concatenate((self.X_pattern_hybrid, self.Y_pattern_hybrid), axis=1)

        # Encode labels
        self.label_encoder = preprocessing.LabelEncoder()
        self.Y_pattern_hybrid = self.label_encoder.fit_transform(self.Y_pattern_hybrid.ravel())

        # Concatenate phenomenological model data for prediction
        self.X_prev_hybrid = np.concatenate((self.PhenomDataDriven.X_prev, aux_pattern_hybrid), axis=1)

        # Apply OneHotEncoding for categorical features
        self.ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [9])], remainder='passthrough')
        self.X_pattern_hybrid = np.array(self.ct.fit_transform(self.X_pattern_hybrid))
        self.X_prev_hybrid = np.array(self.ct.transform(self.X_prev_hybrid))

        # Split the hybrid dataset into training and test sets
        self.X_train_pattern_hybrid, self.X_test_pattern_hybrid, self.y_train_pattern_hybrid, self.y_test_pattern_hybrid = model_selection.train_test_split(
            self.X_pattern_hybrid, self.PhenomDataDriven.Y_pattern_data, test_size=0.1, random_state=0)

        # Scale the training and test data
        self.sc = preprocessing.StandardScaler()
        self.X_train_pattern_hybrid = self.sc.fit_transform(self.X_train_pattern_hybrid)
        self.X_test_pattern_hybrid = self.sc.transform(self.X_test_pattern_hybrid)
        self.X_prev_hybrid = self.sc.transform(self.X_prev_hybrid)

    def HybridRandomForestMap(self):
        """
        Trains a hybrid RandomForest model that integrates both experimental data and 
        data predicted by the phenomenological model to generate a flow pattern map.

        Inputs
        ----------
        None
        
        Outputs
        -------
        self.pattern : numpy.ndarray
            A 2D array representing the predicted flow pattern map, with dimensions defined by self.Phenom.resol.
        
        Notes
        -------
        - The function trains a RandomForest classifier using both experimental and phenomenological data, 
          applying different weights to the respective datasets.
        - A Support Vector Classifier (SVC) is optionally used to refine the predictions after the initial RandomForest prediction.
        - The resulting predictions are reshaped into a 2D map, with each entry corresponding to a predicted flow pattern class.
        """

        self.pattern = []

        # Calculate weights for experimental and model data
        weights_train = compute_sample_weight({0: self.data_weight, 1: self.model_weight}, self.y_train_pattern_hybrid)

        # Train RandomForest with sample weights for hybrid data
        clf_pattern_hybrid = RandomForestClassifier(n_estimators=80, criterion='gini', bootstrap=False, random_state=0)
        clf_pattern_hybrid.fit(self.X_train_pattern_hybrid, self.y_train_pattern_hybrid, sample_weight=weights_train)

        # Predict the flow patterns based on the trained model
        y_pred_hybrid = clf_pattern_hybrid.predict(self.X_prev_hybrid)

        # Map the predicted classes back to the original labels
        self.pattern_map = self.label_encoder.inverse_transform(y_pred_hybrid)

        # Optional: refine predictions with a Support Vector Classifier (SVC)
        X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid = model_selection.train_test_split(self.X_prev_hybrid, y_pred_hybrid, test_size=0.01)
        classifier_model = SVC(kernel='linear', C=1)
        classifier_model.fit(X_train_hybrid, y_train_hybrid)

        # Final predictions with SVC
        y_pred_hybrid = classifier_model.predict(self.X_prev_hybrid)

        # Reshape predictions into a 2D pattern map
        self.pattern_map = np.reshape(y_pred_hybrid, (self.Phenom.resol, self.Phenom.resol))

        return self.pattern_map
