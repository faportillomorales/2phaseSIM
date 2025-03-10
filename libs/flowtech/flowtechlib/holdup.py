#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Set 09 20:23:09 2023

@author: LEMI Laboratory
"""
import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import fsolve, bisect
from sklearn import preprocessing, model_selection, preprocessing, ensemble 
from sklearn import neural_network, metrics
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tkinter as tk
from tqdm import tqdm

import importlib.resources as pkg_resources
from . import Data

from .patterns import *
from .phenom import *

class HoldUp(Patterns):

    def __init__(self, parms):
        """
        Function.

        Parameters
        ----------
        """
        super().__init__(parms)

        self.Phenom = Phenom(parms)
        self.Phenom.__init__(parms)

    def PhenomPressureHoldUp(self, vs_l, vs_g): # Xiao 1990
        """
         Function.

        Parameters
        ----------
        """
        alpha_l = 0
        self.dp_dx = 0
        self.fraction_l = 0
        self.pattern = self.Phenom.Barnea1986_function_point(vs_l, vs_g)
        
        if (self.pattern == 'Stratified'): #Stratificado
            self.fraction_l = self.Phenom.A_adm_l/self.Phenom.A_adm
            v_g = self.Phenom.v_adm_g*vs_g
            v_l = self.Phenom.v_adm_l*vs_l
            d_l = self.Phenom.d*4*self.Phenom.A_adm_l/self.Phenom.S_adm_l
            d_g = self.Phenom.d*4*self.Phenom.A_adm_g/(self.Phenom.S_adm_g + self.Phenom.S_adm_i)
            Re_l = self.Phenom.rho_l*v_l*d_l/self.Phenom.mu_l
            Re_g = self.Phenom.rho_g*v_g*d_g/self.Phenom.mu_g
            
            if Re_l > 2000: #caso seja turbulento ou laminar fornece os coeficientes para tensão cisalhamento
               C_l = 0.046
               n = 0.2
            else:
               C_l = 16
               n = 1.0
            if Re_g > 2000: #caso seja turbulento ou laminar fornece os coeficientes para tensão cisalhamento
               C_g = 0.046
               m = 0.2
            else:
               C_g = 16
               m = 1.0
            
            tal_l = C_l*(Re_l)**(-n)*self.Phenom.rho_l*v_l**2/2
            tal_g = C_g*(Re_g)**(-m)*self.Phenom.rho_g*v_g**2/2
            self.dp_dx = -((tal_l*self.Phenom.S_adm_l + tal_g*self.Phenom.S_adm_g)/self.Phenom.A_adm/self.Phenom.d + (self.Phenom.A_adm_l*self.Phenom.rho_l/self.Phenom.A_adm + self.Phenom.A_adm_g*self.Phenom.rho_g/self.Phenom.A_adm)*9.81*np.sin(self.Phenom.alpha))
        
        elif (self.pattern == 'Intermittent'):
            U_s = vs_g + vs_l
            #slug region equations
            fraction_s = 1.0/np.power(1.0+(U_s/8.66), 1.39) #xiao 1990
            if fraction_s < 0.48:
                fraction_s = 0.48
            if self.Phenom.d > 0.0381:
                L_s = np.exp(-26.6 +28.5*(np.log(self.Phenom.d)+3.67)**0.1)
            else:
                L_s = 30*self.Phenom.d
            rho_s = fraction_s*self.Phenom.rho_l+(1-fraction_s)*self.Phenom.rho_g
            mu_s = fraction_s*self.Phenom.mu_l+(1-fraction_s)*self.Phenom.mu_g
            Re_s = rho_s*U_s*self.Phenom.d/mu_s

            if Re_s > 2000:
                C = 1.2
                f_s = 0.046/Re_s**0.2
            else:
                C = 2
                f_s = 16/Re_s
            
            U_b = 1.2*U_s + (1.53*(self.Phenom.sigma*9.81*(self.Phenom.rho_l-self.Phenom.rho_g)/self.Phenom.rho_l**2)**0.25)*np.sin(self.Phenom.alpha)*fraction_s**0.1
            U_t = C*U_s + 0.35*((9.81*self.Phenom.d)**0.5)*np.sin(self.Phenom.alpha) + (0.54*(9.81*self.Phenom.d)**0.5)*np.cos(self.Phenom.alpha)
            tal_s = (f_s*rho_s*U_s**2)/2
            U_l = U_s/fraction_s - U_b*(1-fraction_s)/fraction_s

            h_f = self.Phenom.h_adm*self.Phenom.d
            A_f = 0.25*self.Phenom.d**2*(np.pi-np.arccos(2*h_f/self.Phenom.d-1) + (2*h_f/self.Phenom.d-1)*np.sqrt(1-(2*h_f/self.Phenom.d-1)**2)) 
            A_g = 0.25*self.Phenom.d**2*(np.arccos(2*h_f/self.Phenom.d-1) - (2*h_f/self.Phenom.d-1)*np.sqrt(1-(2*h_f/self.Phenom.d-1)**2))
            S_f = self.Phenom.d*(np.pi-np.arccos(2*h_f/self.Phenom.d-1))
            S_g = self.Phenom.d*np.arccos(2*h_f/self.Phenom.d-1)
            S_i = self.Phenom.d*np.sqrt(1-(2*h_f/self.Phenom.d-1)**2)
            A = (self.Phenom.d**2)*np.pi/4 
            fraction_f = A_f/A
            
            U_f = U_t - (U_t-U_l)*fraction_s/fraction_f
            U_g = (U_s - U_f*fraction_f)/(1-fraction_f)
            D_f = 4*A_f/S_f
            D_g = 4*A_g/(S_g + S_i)
            Re_f = self.Phenom.rho_l*U_f*D_f/self.Phenom.mu_l
            Re_g = self.Phenom.rho_g*U_g*D_g/self.Phenom.mu_g

            if Re_f < 2000:
                f_f = 16/Re_f
            else:
                f_f = 0.046/Re_f**0.2
            if Re_g < 2000:
                f_g = 16/Re_g
            else:
                f_g = 0.046/Re_g**0.2
            f_i = 0.0142

            #f_i = f_g
            #f_i = f_g*10 #fator de atrito que obteve performance melhor
            tal_f = f_f*self.Phenom.rho_l*U_f*np.abs(U_f)/2
            tal_g = f_g*self.Phenom.rho_g*U_g*np.abs(U_g)/2
            tal_i = f_i*self.Phenom.rho_g*(U_g-U_f)*np.abs(U_g-U_f)/2        
            
            #L_u = L_s*(U_l*fraction_s-U_f*fraction_f)/(vs_l-U_f*fraction_f)
            L_u = 1.1*L_s
            L_f = L_u - L_s
            self.fraction_l = (U_t*fraction_s + U_b*(1-fraction_s)-vs_g)/U_t
            rho_u = self.fraction_l*self.Phenom.rho_l + (1-self.fraction_l)*self.Phenom.rho_g
            self.dp_dx = -rho_u*9.81*np.sin(self.Phenom.alpha) - ((1/L_u)*((tal_s*np.pi*self.Phenom.d/A*L_s) + ((tal_f*S_f+tal_g*S_g)/A*L_f)))
            
            a = tal_s*np.pi*self.Phenom.d/A*L_s   
            b = (tal_f*S_f+tal_g*S_g)/A*L_f
                    
        elif (self.pattern == 'Annular'):
            
            def parm_anular(delta):
                #D_c = d - 2*delta
                A_c = np.pi*(self.Phenom.d-2*delta)**2/4
                A = np.pi*self.Phenom.d**2/4
                A_f = A-A_c
                S_l = np.pi*self.Phenom.d 
                S_i = np.pi*(self.Phenom.d-2*delta)
                D_c = 4*A_c/S_i
                D_f = 4*A_f/(S_i+S_l)
                v_f = vs_l*(1-E)*self.Phenom.d**2/(4*delta*(self.Phenom.d-delta))
                v_c = (vs_g+vs_l*E)*self.Phenom.d**2/(self.Phenom.d-2*delta)**2
                Re_c = rho_c*v_c*(D_c)/mu_c
                Re_f = self.Phenom.rho_l*v_f*(D_f)/self.Phenom.mu_l
                if Re_c < 2000:
                    f_c = 16/Re_c
                else:
                    f_c = 0.046/Re_c**0.2
                if Re_f < 2000:
                    f_f = 16/Re_f
                else:
                    f_f = 0.046/Re_f**0.2
                Re_sg = self.Phenom.rho_g*vs_g*self.Phenom.d/self.Phenom.mu_g
                
                factor = 1
                if (factor == 1): # shoham(2006)
                    I_h = 1 + 850*((0.42*Re_f**1.25 + 0.00028*Re_f**2.25)**0.4/Re_sg**0.9*(self.Phenom.mu_l/self.Phenom.mu_g)*(self.Phenom.rho_g/self.Phenom.rho_l)**0.5)
                    I_v = 1 + 300*delta/self.Phenom.d
                    I = I_h*(np.cos(self.Phenom.alpha))**2 + I_v*(np.sin(self.Phenom.alpha))**2
                    f_i = I*f_c 
                if (factor == 2): # xiao (1990)
                    f_i = f_c*(1+2250*delta/self.Phenom.d/(rho_c*(v_c-v_f)**2*self.Phenom.d*delta/self.Phenom.d/self.Phenom.sigma)) 
                if (factor == 3): # simplified
                    f_i = f_c
                
                tal_l = f_f*self.Phenom.rho_l*v_f*np.abs(v_f)/2
                tal_i = f_i*rho_c*(v_c-v_f)*np.abs(v_c-v_f)/2
                return tal_l, tal_i, A_f, A_c, A, S_i, S_l
    
            def f_delta (delta):    
                tal_l, tal_i, A_f, A_c, A, S_i, S_l = parm_anular(delta)
                erro = tal_l*S_l/A_f - tal_i*S_i*(1/A_f + 1/A_c) + (self.Phenom.rho_l - rho_c)*9.81*np.sin(self.Phenom.alpha)
                return erro
            
            entranhamento = 4
            if (entranhamento == 1): # Oliemans et al (1986)
                def f_c(E):
                    return ((10**-2.52 * self.Phenom.rho_l**1.08 * self.Phenom.rho_g**0.18 * self.Phenom.mu_l**0.27 * self.Phenom.mu_g**0.28 * self.Phenom.sigma**-1.8 * self.Phenom.d**1.72 * vs_l**0.7 * vs_g**1.44 * 9.81**0.46)*(1-E) - E)
                E = bisect(f_c, 0.000001, 0.999999)

            if (entranhamento == 2):  #sawant et al (2008)
                Re_l = self.Phenom.rho_l*vs_l*self.Phenom.d/self.Phenom.mu_l                     
                Re_lmin = 250*np.log(Re_l)-1265
                We = (self.Phenom.rho_g*vs_g**2*self.Phenom.d/self.Phenom.sigma)*((self.Phenom.rho_l-self.Phenom.rho_g)/self.Phenom.rho_g)**(1/3)
                E = (1-Re_lmin/Re_l)*np.tanh(0.000231*Re_l**-0.35*We**1.25)
            if (entranhamento == 3):  #sawant et al (2009)
                Re_l = self.Phenom.rho_l*vs_l*self.Phenom.d/self.Phenom.mu_l                     
                Re_lmin = 250*np.log(Re_l)-1265
                Nu = self.Phenom.mu_l/(self.Phenom.rho_l**2*self.Phenom.sigma**3/((self.Phenom.rho_l-self.Phenom.rho_g)*9.81))**0.25
                Re_lmin2 = 13*Nu**-0.5 + 0.3*(Re_l-13*Nu**-0.5)**0.95
                We = (self.Phenom.rho_g*vs_g**2*self.Phenom.d/self.Phenom.sigma)*((self.Phenom.rho_l-self.Phenom.rho_g)/self.Phenom.rho_g)**(1/3)
                We2 = We*((self.Phenom.rho_l-self.Phenom.rho_g)/self.Phenom.rho_g)**(-1/12)
                E = (1-Re_lmin2/Re_l)*np.tanh(0.000231*Re_l**-0.35*We2**1.25)
            if (entranhamento == 4):  #No entrainment
                E=0
            
            alpha_c = vs_g/(vs_g+vs_l*E)
            rho_c = alpha_c*self.Phenom.rho_g + (1-alpha_c)*self.Phenom.rho_l
            mu_c = alpha_c*self.Phenom.mu_g + (1-alpha_c)*self.Phenom.mu_l
            delta = bisect(f_delta, 0.000001, self.Phenom.d/2.0000001)
            tal_l, tal_i, A_f, A_c, A, S_i, S_l = parm_anular(delta)
            self.fraction_l = 1 - alpha_c*(1-2*delta/self.Phenom.d)**2  
            self.dp_dx = -tal_l*S_l/A - (A_f/A*self.Phenom.rho_l+A_c/A*rho_c)*9.81*np.sin(self.Phenom.alpha)

        elif (self.pattern == 'Dispersed'):
            v_m = vs_l + vs_g
            self.fraction_l = vs_l/(v_m)
            rho_m = self.Phenom.rho_l*self.fraction_l + self.Phenom.rho_g*(1.0 - self.fraction_l)
            mu_m = self.Phenom.mu_l*self.fraction_l + self.Phenom.mu_g*(1.0 - self.fraction_l)
            Re_m = rho_m*self.Phenom.d*v_m**2/mu_m
            if Re_m > 2000: #caso seja turbulento ou laminar fornece os coeficientes para tensão cisalhamento
                C_m = 0.046
                a = 0.2
            else:
                C_m = 16
                a = 1.0
            f_m = C_m*Re_m**(-a)
            self.dp_dx = -(2*f_m*rho_m*v_m**2/self.Phenom.d + rho_m*9.81*np.sin(self.Phenom.alpha))
            
        return self.dp_dx, self.fraction_l
        
    def HoldUpPatterns(self):
        pass

class HoldUpDataDriven(Patterns):

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
        
        self.HoldUpDD = HoldUp(parms)
        self.HoldUpDD.__init__(parms)

        with pkg_resources.path(Data,'FlowTechPressureData.hdf5') as file:
            with h5py.File(file, "r") as f:
                self.X_dpdx_data = f["XDpDxData"][:]
                self.Y_dpdx = f["YDpDx"][:]

        with pkg_resources.path(Data,'FlowTechHoldUpData.hdf5') as file:
            with h5py.File(file, "r") as f:
                self.X_hold_data = f["XHoldUpData"][:]
                self.Y_hold = f["YHoldUp"][:]
        
        self.X_train_dpdx_data, self.X_test_dpdx_data, self.y_train_dpdx_data, self.y_test_dpdx_data = model_selection.train_test_split(self.X_dpdx_data, self.Y_dpdx, test_size = 0.02,random_state = 0)
        self.X_train_hold_data, self.X_test_hold_data, self.y_train_hold_data, self.y_test_hold_data = model_selection.train_test_split(self.X_hold_data, self.Y_hold, test_size = 0.02,random_state = 0)
        
        n = len(self.ext_point_vg)
        aux_vg = np.ravel(self.ext_point_vg).reshape(n,1)
        aux_vl = np.ravel(self.ext_point_vl).reshape(n,1)
        aux_mu_l = np.full((n, 1), self.mu_l)
        aux_mu_g = np.full((n, 1), self.mu_g)
        aux_rho_l = np.full((n, 1), self.rho_l)
        aux_rho_g = np.full((n, 1), self.rho_g)
        aux_d = np.full((n, 1), self.d)
        aux_alpha = np.full((n, 1), self.alpha)
        aux_sigma = np.full((n, 1), self.sigma)

        ### ver a velocidade
        self.dp_dx = []
        self.fraction_l = []

        for i, vg in enumerate(self.ext_point_vg):
            self.HoldUpDD.PhenomPressureHoldUp(self.ext_point_vg[i],self.ext_point_vl[i])
            self.dp_dx.append(-self.HoldUpDD.dp_dx)
            self.fraction_l.append(self.HoldUpDD.fraction_l)

        self.model_dpdx = np.ravel(self.dp_dx).reshape(n,1)
        self.model_hold = np.ravel(self.fraction_l).reshape(n,1)

        self.X_pred_data = np.concatenate((aux_vg,aux_vl,aux_mu_l,aux_mu_g,aux_rho_l,aux_rho_g,aux_d,aux_alpha,aux_sigma),axis=1)
        self.X_pred_hybrid_dpdx = np.concatenate((aux_vg,aux_vl,aux_mu_l,aux_mu_g,aux_rho_l,aux_rho_g,aux_d,aux_alpha,aux_sigma,self.model_dpdx),axis=1)
        self.X_pred_hybrid_hold = np.concatenate((aux_vg,aux_vl,aux_mu_l,aux_mu_g,aux_rho_l,aux_rho_g,aux_d,aux_alpha,aux_sigma,self.model_hold),axis=1)
        self.sc = preprocessing.StandardScaler()

        self.X_train_dpdx_data = self.sc.fit_transform(self.X_train_dpdx_data)
        self.X_test_dpdx_data = self.sc.transform(self.X_test_dpdx_data)
        self.X_pred_data = self.sc.transform(self.X_pred_data)

        self.X_train_hold_data = self.sc.fit_transform(self.X_train_hold_data)
        self.X_test_hold_data = self.sc.transform(self.X_test_hold_data)

        self.y_train_dpdx_data = np.reshape(self.y_train_dpdx_data,(self.y_train_dpdx_data.shape[0],))
        self.y_train_hold_data = np.reshape(self.y_train_hold_data,(self.y_train_hold_data.shape[0],))

    def GradientBoosting(self):
        """
        Function.
        Parameters
        ----------
        """
        self.clf_dpdx_data = GradientBoostingRegressor(n_estimators=150,learning_rate=0.5,max_depth=4,subsample=0.9,random_state = 0)
        self.clf_hold_data = GradientBoostingRegressor(n_estimators=80,learning_rate=0.1,max_depth=6,subsample=0.5,random_state = 0)
        self.clf_dpdx_data.fit(self.X_train_dpdx_data, self.y_train_dpdx_data)
        self.clf_hold_data.fit(self.X_train_hold_data, self.y_train_hold_data)
        self.dpdx = self.clf_dpdx_data.predict(self.X_pred_data)
        self.holdup = self.clf_hold_data.predict(self.X_pred_data)

        return self.dpdx, self.holdup
    
    def MLP(self):
        """
        Function.
        Parameters
        ----------
        """
        clf_dpdx_data = MLPRegressor(hidden_layer_sizes=(10,30,30,10),activation='relu',max_iter=20000,solver ='lbfgs',random_state = 13)
        clf_hold_data = MLPRegressor(hidden_layer_sizes=(10,30,30,10),activation='relu',max_iter=20000,solver ='lbfgs',random_state = 13)
        clf_dpdx_data.fit(self.X_train_dpdx_data, self.y_train_dpdx_data)
        clf_hold_data.fit(self.X_train_dpdx_data, self.y_train_hold_data)
        self.dpdx = self.clf_dpdx_data.predict(self.X_pred_data)
        self.holdup = self.clf_hold_data.predict(self.X_pred_data)
        return self.dpdx, self.holdup
    
    def HoldUpDataDrivenPatterns(self):
        pass

class HoldUpDataDrivenHybrid(Patterns):

    def __init__(self, parms, aux_holdup_hybrid=[]):
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
        self.HoldUpDDH = HoldUpDataDriven(parms)
        self.HoldUpDDH.__init__(parms)

        with pkg_resources.path(Data,'FlowTechPressureData.hdf5') as file:
            with h5py.File(file, "r") as f:
                self.X_dpdx_hybrid = f["XDpDxDataHybrid"][:]

        with pkg_resources.path(Data,'FlowTechHoldUpData.hdf5') as file:
            with h5py.File(file, "r") as f:
                self.X_hold_hybrid = f["XHoldUpDataHybrid"][:]
        
        # Hybrid
        self.X_train_dpdx_hybrid, self.X_test_dpdx_hybrid, self.y_train_dpdx_hybrid, self.y_test_dpdx_hybrid = model_selection.train_test_split(self.X_dpdx_hybrid, self.HoldUpDDH.Y_dpdx, test_size = 0.02,random_state = 0)
        self.X_train_hold_hybrid, self.X_test_hold_hybrid, self.y_train_hold_hybrid, self.y_test_hold_hybrid = model_selection.train_test_split(self.X_hold_hybrid, self.HoldUpDDH.Y_hold, test_size = 0.02,random_state = 0)

        self.X_train_dpdx_hybrid = self.HoldUpDDH.sc.fit_transform(self.X_train_dpdx_hybrid)
        self.X_test_dpdx_hybrid = self.HoldUpDDH.sc.transform(self.X_test_dpdx_hybrid)
        self.X_pred_hybrid_dpdx = self.HoldUpDDH.sc.transform(self.HoldUpDDH.X_pred_hybrid_dpdx)

        self.X_train_hold_hybrid = self.HoldUpDDH.sc.fit_transform(self.X_train_hold_hybrid)
        self.X_test_hold_hybrid = self.HoldUpDDH.sc.transform(self.X_test_hold_hybrid)
        self.X_pred_hybrid_hold = self.HoldUpDDH.sc.transform(self.HoldUpDDH.X_pred_hybrid_hold)

        self.y_train_dpdx_hybrid = np.reshape(self.y_train_dpdx_hybrid,(self.y_train_dpdx_hybrid.shape[0],))
        self.y_train_hold_hybrid = np.reshape(self.y_train_hold_hybrid,(self.y_train_hold_hybrid.shape[0],))

    def GradientBoostingHybrid(self):
        """
        Function.
        Parameters
        ----------
        """
        self.clf_dpdx_hybrid = GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,max_depth=6,subsample=0.5,random_state = 0)
        self.clf_hold_hybrid = GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,max_depth=4,subsample=0.9,random_state = 0)
        self.clf_dpdx_hybrid.fit(self.X_train_dpdx_hybrid, self.y_train_dpdx_hybrid)
        self.clf_hold_hybrid.fit(self.X_train_hold_hybrid, self.y_train_hold_hybrid)
        self.dpdx = self.clf_dpdx_hybrid.predict(self.HoldUpDDH.X_pred_hybrid_dpdx)
        self.holdup = self.clf_hold_hybrid.predict(self.HoldUpDDH.X_pred_hybrid_hold)
        return self.dpdx, self.holdup

    def MLPHybrid(self):
        """
        Function.
        Parameters
        ----------
        """
        self.clf_dpdx_hybrid = MLPRegressor(hidden_layer_sizes=(10,30,30,10),activation='relu',max_iter=20000,solver ='lbfgs',random_state = 13)
        self.clf_hold_hybrid = MLPRegressor(hidden_layer_sizes=(10,30,30,10),activation='relu',max_iter=20000,solver ='lbfgs',random_state = 13)
        self.clf_dpdx_hybrid.fit(self.X_train_dpdx_hybrid, self.y_train_dpdx_hybrid)  
        self.clf_hold_hybrid.fit(self.X_train_hold_hybrid, self.y_train_hold_hybrid)
        self.dpdx = self.clf_dpdx_hybrid.predict(self.HoldUpDDH.X_pred_hybrid_dpdx)
        self.holdup = self.clf_hold_hybrid.predict(self.HoldUpDDH.X_pred_hybrid_hold)
        return self.dpdx, self.holdup

    def HoldUpDataDrivenHybridPatterns(self):
        pass