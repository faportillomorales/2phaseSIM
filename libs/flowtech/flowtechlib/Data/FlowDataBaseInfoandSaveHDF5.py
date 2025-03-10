#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Set 09 20:23:09 2023

@author: LEMI Laboratory
"""

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, preprocessing
from sklearn import neural_network, metrics
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.gridspec import GridSpec
from collections import Counter

import plotly.express as px
import h5py
import sys

from flowtechlib import dicflowpattern as dfp

class FlowData:

    def __init__(self):
        
        self.database = {
            'Pattern': pd.read_excel("Flow_Database.xlsx", sheet_name = 'Flow_Pattern'),
            'Pattern_Carlos': pd.read_excel("Flow_Database_Carlos.xlsx", sheet_name = 'Flow_Pattern'),
            'Pattern_NUEM': pd.read_excel("Flow_Database_NUEM.xlsx", sheet_name = 'Flow_Pattern'),
            'Pressure': pd.read_excel("Flow_Database.xlsx", sheet_name = 'Pressure'),
            'Holdup': pd.read_excel("Flow_Database.xlsx", sheet_name = 'Holdup')
                        }
        
        self.flow_dic_shoham = dfp.flow_dic_shoham
        self.flow_dic_barnea = dfp.flow_dic_barnea
        self.aux_flow_barnea = dfp.aux_pattern_barnea
        self.aux_flow_shoham = dfp.aux_pattern_shoham

        self.all_data = False

        self.saveBackUp = False
    
    def SaveFlowPatternDataFrame(self):

        self.data_flow_pattern = self.database['Pattern']
        self.data_flow_pattern_carlos = self.database['Pattern_Carlos']
        self.data_flow_pattern_nuem = self.database['Pattern_NUEM']

        ############### References ###############
        References = self.data_flow_pattern.loc[:,["Paper's Reference [-]"]].values
        References_Carlos = self.data_flow_pattern_carlos.loc[:,["Paper's Reference [-]"]].values
        References_Nuem  = self.data_flow_pattern_nuem.loc[:,["Paper's Reference [-]"]].values

        TitleReferences = self.data_flow_pattern.loc[:,["Paper's Title [-]"]].values
        TitleReferences_Carlos = self.data_flow_pattern_carlos.loc[:,["Paper's Title [-]"]].values
        TitleReferences_Nuem = self.data_flow_pattern_nuem.loc[:,["Paper's Title [-]"]].values

        ############### Fluid 1 ###############
        Typ_Liq = self.data_flow_pattern.loc[:,['Fluid 1 [-]']].values
        Vel_Liq = self.data_flow_pattern.loc[:,['Fluid 1 Superficial Velocity [m/s]']].values
        Vis_Liq = self.data_flow_pattern.loc[:,['Fluid 1 Viscosity [Pa.s]']].values
        Den_Liq = self.data_flow_pattern.loc[:,['Fluid 1 Density [kg/m³]']].values

        Typ_Liq_Carlos = self.data_flow_pattern_carlos.loc[:,['Fluid 1 [-]']].values
        Vel_Liq_Carlos = self.data_flow_pattern_carlos.loc[:,['Fluid 1 Superficial Velocity [m/s]']].values
        Vis_Liq_Carlos = self.data_flow_pattern_carlos.loc[:,['Fluid 1 Viscosity [Pa.s]']].values
        Den_Liq_Carlos = self.data_flow_pattern_carlos.loc[:,['Fluid 1 Density [kg/m³]']].values

        Typ_Liq_Nuem = self.data_flow_pattern_nuem.loc[:,['Fluid 1 [-]']].values
        Vel_Liq_Nuem = self.data_flow_pattern_nuem.loc[:,['Fluid 1 Superficial Velocity [m/s]']].values
        Vis_Liq_Nuem = self.data_flow_pattern_nuem.loc[:,['Fluid 1 Viscosity [Pa.s]']].values
        Den_Liq_Nuem = self.data_flow_pattern_nuem.loc[:,['Fluid 1 Density [kg/m³]']].values

        ############### Fluid 2 ###############
        Typ_Gas = self.data_flow_pattern.loc[:,['Fluid 2 [-]']].values
        Vel_Gas = self.data_flow_pattern.loc[:,['Fluid 2 Superficial Velocity [m/s]']].values
        Vis_Gas = self.data_flow_pattern.loc[:,['Fluid 2 Viscosity [Pa.s]']].values
        Den_Gas = self.data_flow_pattern.loc[:,['Fluid 2 Density [kg/m³]']].values

        Typ_Gas_Carlos = self.data_flow_pattern_carlos.loc[:,['Fluid 2 [-]']].values
        Vel_Gas_Carlos = self.data_flow_pattern_carlos.loc[:,['Fluid 2 Superficial Velocity [m/s]']].values
        Vis_Gas_Carlos = self.data_flow_pattern_carlos.loc[:,['Fluid 2 Viscosity [Pa.s]']].values
        Den_Gas_Carlos = self.data_flow_pattern_carlos.loc[:,['Fluid 2 Density [kg/m³]']].values

        Typ_Gas_Nuem = self.data_flow_pattern_nuem.loc[:,['Fluid 2 [-]']].values
        Vel_Gas_Nuem = self.data_flow_pattern_nuem.loc[:,['Fluid 2 Superficial Velocity [m/s]']].values
        Vis_Gas_Nuem = self.data_flow_pattern_nuem.loc[:,['Fluid 2 Viscosity [Pa.s]']].values
        Den_Gas_Nuem = self.data_flow_pattern_nuem.loc[:,['Fluid 2 Density [kg/m³]']].values

        ############### 
        Diam = self.data_flow_pattern.loc[:,['Internal Diameter [m]']].values
        Incl = self.data_flow_pattern.loc[:,['Duct Inclination [°]']].values
        
        Diam_Carlos = self.data_flow_pattern_carlos.loc[:,['Internal Diameter [m]']].values
        Incl_Carlos = self.data_flow_pattern_carlos.loc[:,['Duct Inclination [°]']].values

        Diam_Nuem = self.data_flow_pattern_nuem.loc[:,['Internal Diameter [m]']].values
        Incl_Nuem = self.data_flow_pattern_nuem.loc[:,['Duct Inclination [°]']].values

        Tens = self.data_flow_pattern.loc[:,['Interfacial Tension [N/m]']].values

        Tens_Carlos = self.data_flow_pattern_carlos.loc[:,['Interfacial Tension [N/m]']].values
        
        Tens_Nuem = self.data_flow_pattern_nuem.loc[:,['Interfacial Tension [N/m]']].values

        ### Pattern
        FlowPattMod = self.data_flow_pattern.loc[:,['Model Prediction [-]']].values
        FlowPatt = self.data_flow_pattern.loc[:,['Flow Pattern [-]']].values
        FlowPattUni = self.data_flow_pattern.loc[:,['Unified Flow Pattern [-]']].values

        FlowPattMod_Carlos = self.data_flow_pattern_carlos.loc[:,['Model Prediction [-]']].values
        FlowPatt_Carlos = self.data_flow_pattern_carlos.loc[:,['Flow Pattern [-]']].values
        FlowPattUni_Carlos = self.data_flow_pattern_carlos.loc[:,['Unified Flow Pattern [-]']].values

        FlowPattMod_Nuem = self.data_flow_pattern_nuem.loc[:,['Model Prediction [-]']].values
        FlowPatt_Nuem = self.data_flow_pattern_nuem.loc[:,['Flow Pattern [-]']].values
        FlowPattUni_Nuem = self.data_flow_pattern_nuem.loc[:,['Unified Flow Pattern [-]']].values

        #Output features
        Y_pattern_data_all_str = self.data_flow_pattern.loc[:,['Flow Pattern [-]']].values
        Y_pattern_data_all_str_Carlos = self.data_flow_pattern_carlos.loc[:,['Flow Pattern [-]']].values
        Y_pattern_data_all_str_Nuem = self.data_flow_pattern_nuem.loc[:,['Flow Pattern [-]']].values

        if self.all_data:
            #######################################
            References = np.concatenate((References, References_Carlos, References_Nuem), axis=0)
            #######################################
            TitleReferences = np.concatenate((TitleReferences, TitleReferences_Carlos, TitleReferences_Nuem), axis=0)
            #######################################
            ############### Fluid 1 ###############
            Typ_Liq = np.concatenate((Typ_Liq, Typ_Liq_Carlos, Typ_Liq_Nuem), axis=0)
            Vel_Liq = np.concatenate((Vel_Liq, Vel_Liq_Carlos, Vel_Liq_Nuem), axis=0)
            Vis_Liq = np.concatenate((Vis_Liq, Vis_Liq_Carlos, Vis_Liq_Nuem), axis=0)
            Den_Liq = np.concatenate((Den_Liq, Den_Liq_Carlos, Den_Liq_Nuem), axis=0)
            #######################################
            ############### Fluid 2 ###############
            Typ_Gas = np.concatenate((Typ_Gas, Typ_Gas_Carlos, Typ_Gas_Nuem), axis=0)
            Vel_Gas = np.concatenate((Vel_Gas, Vel_Gas_Carlos, Vel_Gas_Nuem), axis=0)
            Vis_Gas = np.concatenate((Vis_Gas, Vis_Gas_Carlos, Vis_Gas_Nuem), axis=0)
            Den_Gas = np.concatenate((Den_Gas, Den_Gas_Carlos, Den_Gas_Nuem), axis=0)
            #######################################
            #######################################
            Diam = np.concatenate((Diam, Diam_Carlos, Diam_Nuem), axis=0)
            Incl = np.concatenate((Incl, Incl_Carlos, Incl_Nuem), axis=0)
            #######################################
            #######################################
            Tens = np.concatenate((Tens, Tens_Carlos, Tens_Nuem), axis=0)
            ############### Pattern ###############
            FlowPattMod = np.concatenate((FlowPattMod, FlowPattMod_Carlos, FlowPattMod_Nuem), axis=0)
            FlowPatt = np.concatenate((FlowPatt, FlowPatt_Carlos, FlowPatt_Nuem), axis=0)
            FlowPattUni = np.concatenate((FlowPattUni, FlowPattUni_Carlos, FlowPattUni_Nuem), axis=0)
            Y_pattern_data_all_str = np.concatenate((Y_pattern_data_all_str, Y_pattern_data_all_str_Carlos, Y_pattern_data_all_str_Nuem), axis=0)
            #######################################

        Y_pattern_data_barnea_str = []
        Y_pattern_data_shoham_str = []
        
        for i,flow in enumerate(Y_pattern_data_all_str):
            str_flow = flow.item(0)

            if str_flow in self.aux_flow_barnea.keys():
                Y_pattern_data_barnea_str.append([self.aux_flow_barnea[str_flow]])
            else:
                print('Padrão nao classificado (Barnea): ',str_flow,5*'=','>','Referencia: ', References[i][0],',',' ', TitleReferences[i][0])
                Y_pattern_data_barnea_str.append([str_flow])
            
            if str_flow in self.aux_flow_shoham.keys():
                Y_pattern_data_shoham_str.append([self.aux_flow_shoham[str_flow]])
            else:
                print('Padrão nao classificado (Shoham): ',str_flow,5*'=','>','Referencia: ', References[i][0],',',' ', TitleReferences[i][0])
                Y_pattern_data_shoham_str.append([str_flow])
        
        Y_pattern_data_barnea_str = np.array(Y_pattern_data_barnea_str)
        Y_pattern_data_shoham_str = np.array(Y_pattern_data_shoham_str)

        aux_df_Flow_Pattern = {
            "Paper's Reference [-]" : [item for sublista in References for item in sublista],
            "Fluid 1 [-]" : [item for sublista in Typ_Liq for item in sublista],
            "Fluid 1 Superficial Velocity [m/s]" : [item for sublista in Vel_Liq for item in sublista],
            "Fluid 1 Viscosity [Pa.s]" : [item for sublista in Vis_Liq for item in sublista],
            "Fluid 1 Density [kg/m³]" : [item for sublista in Den_Liq for item in sublista],
            "Fluid 2 [-]" : [item for sublista in Typ_Gas for item in sublista],
            "Fluid 2 Superficial Velocity [m/s]" : [item for sublista in Vel_Gas for item in sublista],
            "Fluid 2 Viscosity [Pa.s]" : [item for sublista in Vis_Gas for item in sublista],
            "Fluid 2 Density [kg/m³]" : [item for sublista in Den_Gas for item in sublista],
            "Internal Diameter [m]" : [item for sublista in Diam for item in sublista],
            "Duct Inclination [°]" : [item for sublista in Incl for item in sublista],
            "Interfacial Tension [N/m]" : [item for sublista in Tens for item in sublista],
            "Model Prediction [-]" : [item for sublista in FlowPattMod for item in sublista],
            "Flow Pattern [-]" : [item for sublista in FlowPatt for item in sublista],
            "Flow Pattern Shoham [-]" : [item for sublista in Y_pattern_data_shoham_str for item in sublista],
            "Unified Flow Pattern [-]" : [item for sublista in FlowPattUni for item in sublista]
        }
        
        self.df_Flow_Pattern = pd.DataFrame(aux_df_Flow_Pattern)
        self.df_Flow_Pattern.to_csv('FlowTechPatternData.csv', index=False)

    def SaveFlowPressureDataFrame(self):

        self.data_flow_pressure = self.database['Pressure']
        ### References
        References = self.data_flow_pressure.loc[:, ["Paper's Reference [-]"]].values
        ### Fluid 1
        Typ_Liq = self.data_flow_pressure.loc[:, ['Fluid 1 [-]']].values
        Vel_Liq = self.data_flow_pressure.loc[:, ['Fluid 1 Superficial Velocity [m/s]']].values
        Vis_Liq = self.data_flow_pressure.loc[:, ['Fluid 1 Viscosity [Pa.s]']].values
        Den_Liq = self.data_flow_pressure.loc[:, ['Fluid 1 Density [kg/m³]']].values
        ### Fluid 2
        Typ_Gas = self.data_flow_pressure.loc[:, ['Fluid 2 [-]']].values
        Vel_Gas = self.data_flow_pressure.loc[:, ['Fluid 2 Superficial Velocity [m/s]']].values
        Vis_Gas = self.data_flow_pressure.loc[:, ['Fluid 2 Viscosity [Pa.s]']].values
        Den_Gas = self.data_flow_pressure.loc[:, ['Fluid 2 Density [kg/m³]']].values
        ### 
        Diam = self.data_flow_pressure.loc[:, ['Internal Diameter [m]']].values
        Incl = self.data_flow_pressure.loc[:, ['Duct Inclination [°]']].values
        ###
        self.Tens = self.data_flow_pressure.loc[:, ['Interfacial Tension [N/m]']].values
        ### Pressure
        self.Pressure = self.data_flow_pressure.loc[:, ['Pressure Gradient [Pa/m]']].values
        self.Pressure_Model = self.data_flow_pressure.loc[:, ['Model Prediction Pressure Gradient [Pa/m]']].values

        aux_df_Flow_Pressure = {
            "Paper's Reference [-]" : [item for sublista in References for item in sublista],
            "Fluid 1 [-]" : [item for sublista in Typ_Liq for item in sublista],
            "Fluid 1 Superficial Velocity [m/s]" : [item for sublista in Vel_Liq for item in sublista],
            "Fluid 1 Viscosity [Pa.s]" : [item for sublista in Vis_Liq for item in sublista],
            "Fluid 1 Density [kg/m³]" : [item for sublista in Den_Liq for item in sublista],
            "Fluid 2 [-]" : [item for sublista in Typ_Gas for item in sublista],
            "Fluid 2 Superficial Velocity [m/s]" : [item for sublista in Vel_Gas for item in sublista],
            "Fluid 2 Viscosity [Pa.s]" : [item for sublista in Vis_Gas for item in sublista],
            "Fluid 2 Density [kg/m³]" : [item for sublista in Den_Gas for item in sublista],
            "Internal Diameter [m]" : [item for sublista in Diam for item in sublista],
            "Duct Inclination [°]" : [item for sublista in Incl for item in sublista],
            "Interfacial Tension [N/m]" : [item for sublista in self.Tens for item in sublista],
            "Pressure Gradient [Pa/m]" : [item for sublista in self.Pressure for item in sublista],
            "Model Prediction Pressure Gradient [Pa/m]" : [item for sublista in self.Pressure_Model for item in sublista]
        }

        self.df_Flow_Pressure = pd.DataFrame(aux_df_Flow_Pressure)
        self.df_Flow_Pressure.to_csv('FlowTechPressureData.csv', index=False)

    def SaveFlowHoldupDataFrame(self):

        self.data_flow_holdup = self.database['Holdup']
        ### References
        References = self.data_flow_holdup.loc[:, ["Paper's Reference [-]"]].values
        ### Fluid 1
        Typ_Liq = self.data_flow_holdup.loc[:, ['Fluid 1 [-]']].values
        Vel_Liq = self.data_flow_holdup.loc[:, ['Fluid 1 Superficial Velocity [m/s]']].values
        Vis_Liq = self.data_flow_holdup.loc[:, ['Fluid 1 Viscosity [Pa.s]']].values
        Den_Liq = self.data_flow_holdup.loc[:, ['Fluid 1 Density [kg/m³]']].values
        ### Fluid 2
        Typ_Gas = self.data_flow_holdup.loc[:, ['Fluid 2 [-]']].values
        Vel_Gas = self.data_flow_holdup.loc[:, ['Fluid 2 Superficial Velocity [m/s]']].values
        Vis_Gas = self.data_flow_holdup.loc[:, ['Fluid 2 Viscosity [Pa.s]']].values
        Den_Gas = self.data_flow_holdup.loc[:, ['Fluid 2 Density [kg/m³]']].values
        ### 
        Diam = self.data_flow_holdup.loc[:, ['Internal Diameter [m]']].values
        Incl = self.data_flow_holdup.loc[:, ['Duct Inclination [°]']].values
        ###
        self.Tens = self.data_flow_holdup.loc[:, ['Interfacial Tension [N/m]']].values
        ### Pressure
        self.Vol_Fra = self.data_flow_holdup.loc[:, ['Liquid Volumetric Fraction [-]']].values
        self.Vol_Fra_Model = self.data_flow_holdup.loc[:, ['Model Holdup Prediction [-]']].values

        aux_df_Flow_Holdup = {
            "Paper's Reference [-]" : [item for sublista in References for item in sublista],
            "Fluid 1 [-]" : [item for sublista in Typ_Liq for item in sublista],
            "Fluid 1 Superficial Velocity [m/s]" : [item for sublista in Vel_Liq for item in sublista],
            "Fluid 1 Viscosity [Pa.s]" : [item for sublista in Vis_Liq for item in sublista],
            "Fluid 1 Density [kg/m³]" : [item for sublista in Den_Liq for item in sublista],
            "Fluid 2 [-]" : [item for sublista in Typ_Gas for item in sublista],
            "Fluid 2 Superficial Velocity [m/s]" : [item for sublista in Vel_Gas for item in sublista],
            "Fluid 2 Viscosity [Pa.s]" : [item for sublista in Vis_Gas for item in sublista],
            "Fluid 2 Density [kg/m³]" : [item for sublista in Den_Gas for item in sublista],
            "Internal Diameter [m]" : [item for sublista in Diam for item in sublista],
            "Duct Inclination [°]" : [item for sublista in Incl for item in sublista],
            "Interfacial Tension [N/m]" : [item for sublista in self.Tens for item in sublista],
            "Liquid Volumetric Fraction [-]" : [item for sublista in self.Vol_Fra for item in sublista],
            "Model Holdup Prediction [-]" : [item for sublista in self.Vol_Fra_Model for item in sublista]
        }
        
        self.df_Flow_Holdup = pd.DataFrame(aux_df_Flow_Holdup)
        self.df_Flow_Holdup.to_csv('FlowTechHoldupData.csv', index=False)
    
    def SaveFlowPatternhdf5(self):

        X_pattern_data = self.df_Flow_Pattern.loc[:, ['Fluid 2 Superficial Velocity [m/s]', \
                                                      'Fluid 1 Superficial Velocity [m/s]', \
                                                      'Fluid 1 Viscosity [Pa.s]', \
                                                      'Fluid 2 Viscosity [Pa.s]', \
                                                      'Fluid 1 Density [kg/m³]', \
                                                      'Fluid 2 Density [kg/m³]', \
                                                      'Internal Diameter [m]', \
                                                      'Duct Inclination [°]', \
                                                      'Interfacial Tension [N/m]']].values
        Y_pattern_data_str = self.df_Flow_Pattern.loc[:, ['Unified Flow Pattern [-]']].values
        Y_pattern_data_all_str = self.df_Flow_Pattern.loc[:, ['Flow Pattern [-]']].values

        Y_pattern_data_barnea_str = []
        Y_pattern_data_shoham_str = []
        
        for i,flow in enumerate(Y_pattern_data_all_str):
            str_flow = flow.item(0)
            if str_flow in self.aux_flow_barnea.keys():
                Y_pattern_data_barnea_str.append([self.aux_flow_barnea[str_flow]])
            else:
                # print('Padrão nao classificado (Barnea): ',str_flow,5*'=','>','Referencia: ', References[i][0],',',' ', TitleReferences[i][0])
                Y_pattern_data_barnea_str.append([str_flow])
            
            if str_flow in self.aux_flow_shoham.keys():
                Y_pattern_data_shoham_str.append([self.aux_flow_shoham[str_flow]])
            else:
                # print('Padrão nao classificado (Shoham): ',str_flow,5*'=','>','Referencia: ', References[i][0],',',' ', TitleReferences[i][0])
                Y_pattern_data_shoham_str.append([str_flow])

        Y_pattern_data_barnea_str = np.array(Y_pattern_data_barnea_str)
        Y_pattern_data_shoham_str = np.array(Y_pattern_data_shoham_str)
        
        Y_pattern_data = []
        Y_pattern_data_barnea_hyb = []
        Y_pattern_data_shoham_hyb = []

        for i,j in enumerate(Y_pattern_data_shoham_str):
            Y_pattern_data.append([self.flow_dic_barnea[Y_pattern_data_str[i][0]]])
            Y_pattern_data_barnea_hyb.append([self.flow_dic_barnea[Y_pattern_data_barnea_str[i][0]]])
            Y_pattern_data_shoham_hyb.append([self.flow_dic_shoham[Y_pattern_data_shoham_str[i][0]]])
        
        Y_pattern_data_barnea_hyb = np.array(Y_pattern_data_barnea_hyb)
        Y_pattern_data_shoham_hyb = np.array(Y_pattern_data_shoham_hyb)

        with h5py.File("FlowTechPatternData.hdf5", "w") as f:
            f.create_dataset("XPatternData", data=X_pattern_data)
            f.create_dataset("YPatternData", data=Y_pattern_data)
            f.create_dataset("YPatternDataBarneaHyb", data=Y_pattern_data_barnea_hyb)
            f.create_dataset("YPatternDataShohamHyb", data=Y_pattern_data_shoham_hyb)

    def SaveFlowPressurehdf5(self):

        dataset_dpdx = self.database['Pressure']
        self.XX_dpdx = dataset_dpdx[dataset_dpdx['Type of Flow [-]'] == 'Gas-Liquid']

        self.X_dpdx_data = self.XX_dpdx.loc[:, ['Fluid 2 Superficial Velocity [m/s]', 'Fluid 1 Superficial Velocity [m/s]','Fluid 1 Viscosity [Pa.s]','Fluid 2 Viscosity [Pa.s]',
                    'Fluid 1 Density [kg/m³]','Fluid 2 Density [kg/m³]','Internal Diameter [m]','Duct Inclination [°]','Interfacial Tension [N/m]']].values

        #Output features
        self.Y_dpdx = self.XX_dpdx.loc[:, ['Pressure Gradient [Pa/m]']].values

        self.X_dpdx_hybrid = self.XX_dpdx.loc[:, ['Fluid 2 Superficial Velocity [m/s]', 'Fluid 1 Superficial Velocity [m/s]','Fluid 1 Viscosity [Pa.s]','Fluid 2 Viscosity [Pa.s]',
                    'Fluid 1 Density [kg/m³]','Fluid 2 Density [kg/m³]','Internal Diameter [m]','Duct Inclination [°]','Interfacial Tension [N/m]','Model Prediction Pressure Gradient [Pa/m]']].values

        with h5py.File("FlowTechPressureData.hdf5", "w") as f:
            f.create_dataset("XDpDxData", data=self.X_dpdx_data)
            f.create_dataset("XDpDxDataHybrid", data=self.X_dpdx_hybrid)
            f.create_dataset("YDpDx", data=self.Y_dpdx)
        
        if self.saveBackUp:
            with h5py.File("FlowTechPressureDataBackUp.hdf5", "w") as f:
                f.create_dataset("XDpDxData", data=self.X_dpdx_data)
                f.create_dataset("XDpDxDataHybrid", data=self.X_dpdx_hybrid)
                f.create_dataset("YDpDx", data=self.Y_dpdx)

    def SaveFlowHolduphdf5(self):

        dataset_holdup = self.database['Holdup']
        self.XX_holdup = dataset_holdup[dataset_holdup['Type of Flow [-]'] == 'Gas-Liquid']
        self.X_hold_data = self.XX_holdup.loc[:, ['Fluid 2 Superficial Velocity [m/s]', 'Fluid 1 Superficial Velocity [m/s]','Fluid 1 Viscosity [Pa.s]','Fluid 2 Viscosity [Pa.s]',
                    'Fluid 1 Density [kg/m³]','Fluid 2 Density [kg/m³]','Internal Diameter [m]','Duct Inclination [°]','Interfacial Tension [N/m]']].values
        
        #Output features
        self.Y_hold = self.XX_holdup.loc[:, ['Liquid Volumetric Fraction [-]']].values

        self.X_hold_hybrid = self.XX_holdup.loc[:, ['Fluid 2 Superficial Velocity [m/s]', 'Fluid 1 Superficial Velocity [m/s]','Fluid 1 Viscosity [Pa.s]','Fluid 2 Viscosity [Pa.s]',
                    'Fluid 1 Density [kg/m³]','Fluid 2 Density [kg/m³]','Internal Diameter [m]','Duct Inclination [°]','Interfacial Tension [N/m]','Model Holdup Prediction [-]']].values
        
        self.X_hold_hybrid2 = self.XX_holdup.loc[:, ['Model Holdup Prediction [-]']].values
        
        with h5py.File("FlowTechHoldUpData.hdf5", "w") as f:
            f.create_dataset("XHoldUpData", data=self.X_hold_data)
            f.create_dataset("XHoldUpDataHybrid", data=self.X_hold_hybrid)
            f.create_dataset("XHoldUpDataHybrid2", data=self.X_hold_hybrid2)
            f.create_dataset("YHoldUp", data=self.Y_hold)
        
        if self.saveBackUp:
            with h5py.File("FlowTechHoldUpDataBackUp.hdf5", "w") as f:
                f.create_dataset("XHoldUpData", data=self.X_hold_data)
                f.create_dataset("XHoldUpDataHybrid", data=self.X_hold_hybrid)
                f.create_dataset("XHoldUpDataHybrid2", data=self.X_hold_hybrid2)
                f.create_dataset("YHoldUp", data=self.Y_hold)

    def ContagemPatterns(self,lista):
        lista_unica = [item for sublista in lista for item in sublista]
        contagem_nomes = {}
        
        for nome in lista_unica:
            if nome not in contagem_nomes:
                contagem_nomes[nome] = 0
            contagem_nomes[nome] += 1

        for nome, repeticoes in contagem_nomes.items():
            if repeticoes > 1:
                print(f"O padrao '{nome}' se repete {repeticoes} vezes.")

    def GetConfirmation(self):
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

        message = "Would you like to add the data from the specified file to the database?"
        Tdb = True
        while Tdb:
            confirmation = input(f"{message} (Y/N): ").upper()
            if confirmation in ["Y", "N"]:
                if confirmation == "Y":
                    message_res = 'You have chosen to add new data to the database!'
                elif confirmation == "N":
                    message_res = 'You have chosen NOT to add new data to the database!'
                Tdb = False
            else:
                print("Invalid input. Please enter 'Y' or 'N'!")
        
        return confirmation, message_res

    def AddNewData(self, database):
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
        confirmation, message_res = self.get_confirmation()
        print(message_res)

        if confirmation == "N":
            return 
        
        with h5py.File(database, "r") as f:
            X_pattern_data_Read = f["XPatternData"][:]
            Y_pattern_data_Read = f["YPatternData"][:]
            Y_pattern_data_hyb_Read = f["YPatternDataHyb"][:]
            Y_pattern_data_shoram_Read = f["YPatternDataShoram"][:]

        print('Tamanho original...', len(Y_pattern_data_Read))

        data = {'Pattern': pd.read_excel(newdata, sheet_name = 'Flow_Pattern')}
        dataset = data['Pattern']

        data_flow_pattern_add = dataset[dataset['Type of Flow [-]'] == 'Gas-Liquid']
        
        #Input features
        X_pattern_data_New = data_flow_pattern_add.loc[:, ['Fluid 2 Superficial Velocity [m/s]', 'Fluid 1 Superficial Velocity [m/s]','Fluid 1 Viscosity [Pa.s]','Fluid 2 Viscosity [Pa.s]',
                    'Fluid 1 Density [kg/m³]','Fluid 2 Density [kg/m³]','Internal Diameter [m]','Duct Inclination [°]','Interfacial Tension [N/m]']].values
        
        #Output features
        Y_pattern_data_str_New = data_flow_pattern_add.loc[:, ['Unified Flow Pattern [-]']].values
        Y_pattern_data_all_str_New = data_flow_pattern_add.loc[:, ['Flow Pattern [-]']].values
        Y_pattern_data_hyb_str_New = data_flow_pattern_add.loc[:, ['Model Prediction [-]']].values

        Y_pattern_data_shoram_str_New = []

        for i,flow in enumerate(Y_pattern_data_all_str_New):
            str_flow = flow.item(0)
            if str_flow in self.aux_pattern_rec.keys():
                Y_pattern_data_shoram_str_New.append([self.aux_pattern_rec[str_flow]])
            else:
                print('Padrão nao classificado: ',str_flow)
                Y_pattern_data_shoram_str_New.append([str_flow])        

        Y_pattern_data_shoram_str_New = np.array(Y_pattern_data_shoram_str_New)

        Y_pattern_data_New = []
        Y_pattern_data_hyb_New = []
        Y_pattern_data_shoram_New = []

        for i,j in enumerate(Y_pattern_data_shoram_str_New):
            Y_pattern_data_New.append([self.flow_dic_barnea[Y_pattern_data_str_New[i][0]]])
            Y_pattern_data_hyb_New.append([self.flow_dic_barnea[Y_pattern_data_hyb_str_New[i][0]]])
            Y_pattern_data_shoram_New.append([self.flow_dic_shoham[j[0]]])
        
        print('Tamanho Add...', len(Y_pattern_data_New))

        Y_pattern_data_shoram_New = np.array(Y_pattern_data_shoram_New)

        X_pattern_data = np.vstack((X_pattern_data_Read, X_pattern_data_New))
        Y_pattern_data = np.vstack((Y_pattern_data_Read, Y_pattern_data_New))
        Y_pattern_data_hyb = np.vstack((Y_pattern_data_hyb_Read, Y_pattern_data_hyb_New))
        Y_pattern_data_shoram = np.vstack((Y_pattern_data_shoram_Read, Y_pattern_data_shoram_New))

        with h5py.File("FlowTechPatternData.hdf5", "w") as f:
            f.create_dataset("XPatternData", data=X_pattern_data)
            f.create_dataset("YPatternData", data=Y_pattern_data)
            f.create_dataset("YPatternDataHyb", data=Y_pattern_data_hyb)
            f.create_dataset("YPatternDataShoram", data=Y_pattern_data_shoram)
        
        print('Tamanho Final...', len(Y_pattern_data))

    def VisualizerData(self):

        print(self.df_Flow_Pattern.head(20))
        # Codificar os padrões de escoamento em valores numéricos
        label_encoder = LabelEncoder()
        self.df_Flow_Pattern['Flow Pattern Encoded'] = label_encoder.fit_transform(self.df_Flow_Pattern['Flow Pattern [-]'])
        
        # Definindo as colunas numéricas
        numeric_columns = self.df_Flow_Pattern.select_dtypes(include=['float64', 'int64']).columns

        # Adicionar a coluna codificada na lista de colunas numéricas
        numeric_columns_with_flow_pattern = list(numeric_columns) + ['Flow Pattern Encoded']

        # Calcular a matriz de correlação incluindo o padrão de escoamento codificado
        correlation_matrix_with_flow = self.df_Flow_Pattern[numeric_columns_with_flow_pattern].corr()

        # Visualizar a matriz de correlação com um heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(correlation_matrix_with_flow, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap Including Flow Pattern')
        plt.tight_layout()
        plt.show()

    def VisualizerDataPlot(self):

        # Supondo que o DataFrame ja foi criado e é `self.df_Flow_Pattern`
        df = self.df_Flow_Pattern

        # Dicionário para armazenar as contagens para cada coluna
        contagens = {}

        # Itera por todas as colunas do DataFrame e calcula as contagens
        for coluna in df.columns:
            contagens[coluna] = df[coluna].value_counts().sort_index()  # Conta e ordena os valores únicos

        # Converte o dicionário de contagens em um DataFrame
        # contagens_df = pd.DataFrame(contagens).fillna(0)  # Preenche NaN com 0 onde não há contagens

        # print("Resumo das Contagens para Cada Coluna do DataFrame:")
        # print(contagens)
        print(contagens['Duct Inclination [°]'].size)
        print(contagens['Duct Inclination [°]'])

        return 0
    
        # Passo 2: Visualizações Gerais

        
        # 1. Distribuição das Variaveis Numéricas (Histograma e Boxplot)
        # Histograma - Distribuição das Velocidades Superficiais dos Fluidos
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(df['Fluid 1 Superficial Velocity [m/s]'], bins=20, kde=True)
        plt.title('Distribuição de Velocidade Superficial do Fluido 1')

        plt.subplot(1, 2, 2)
        sns.histplot(df['Fluid 2 Superficial Velocity [m/s]'], bins=20, kde=True)
        plt.title('Distribuição de Velocidade Superficial do Fluido 2')

        plt.tight_layout()
        plt.show()

        # Boxplot - Dispersão da Viscosidade e Densidade dos Fluidos
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.boxplot(y=df['Fluid 1 Viscosity [Pa.s]'])
        plt.title('Boxplot de Viscosidade do Fluido 1')

        plt.subplot(1, 2, 2)
        sns.boxplot(y=df['Fluid 2 Viscosity [Pa.s]'])
        plt.title('Boxplot de Viscosidade do Fluido 2')

        plt.tight_layout()
        plt.show()

        # 2. Relações entre Variaveis (Graficos de Dispersão)
        # Dispersão entre as velocidades superficiais dos fluidos
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Fluid 1 Superficial Velocity [m/s]', y='Fluid 2 Superficial Velocity [m/s]', data=df)
        plt.title('Relação entre Velocidade Superficial do Fluido 1 e Fluido 2')
        plt.xlabel('Velocidade Superficial do Fluido 1 [m/s]')
        plt.ylabel('Velocidade Superficial do Fluido 2 [m/s]')
        plt.show()

        # Heatmap de Correlação
        plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title('Mapa de Correlação entre Variaveis')
        plt.show()

        # 3. Distribuição dos Padrões de Fluxo (Graficos de Barras)
        # Contagem de Padrões de Fluxo
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Flow Pattern [-]', data=df, order=df['Flow Pattern [-]'].value_counts().index)
        plt.title('Distribuição de Padrões de Fluxo')
        plt.xticks(rotation=45)
        plt.show()

        # Comparação entre Predição do Modelo e Padrão de Fluxo Unificado
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.countplot(x='Model Prediction [-]', data=df, order=df['Model Prediction [-]'].value_counts().index)
        plt.title('Distribuição de Predição do Modelo')
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        sns.countplot(x='Unified Flow Pattern [-]', data=df, order=df['Unified Flow Pattern [-]'].value_counts().index)
        plt.title('Distribuição do Padrão de Fluxo Unificado')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__" :
    fd = FlowData()
    fd.SaveFlowPatternDataFrame()
    fd.SaveFlowPressureDataFrame()
    fd.SaveFlowHoldupDataFrame()
    fd.SaveFlowPatternhdf5()
    fd.SaveFlowPressurehdf5()
    fd.SaveFlowHolduphdf5()

    fd.df_Flow_Pattern.info()

    # fd.df_Flow_Pressure.info()
    # fd.df_Flow_Holdup.info()
    # fd.VisualizerDataPlot()

    