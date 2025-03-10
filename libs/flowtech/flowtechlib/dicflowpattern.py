#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Set 09 20:23:09 2023

@author: LEMI Laboratory
"""

flow_dic_shoham = {
            ### Annular - Shoham
            "Annular" : 0 ,\
            ### Bubbles - Shoham
            "Bubbles" : 1 ,\
            ### Churn - Shoham
            "Churn" : 2 ,\
            ###Dispersed Bubbles - Shoham 
            "Dispersed Bubbles" : 3 ,\
            ### Enlogated Bubbles - Shoham
            "Elongated Bubbles" : 4 ,\
            ### Slug - Shoham
            "Slug" : 5 ,\
            ### Smooth Stratified - Shoham
            "Smooth Stratified" : 6 ,\
            ### Wavy Stratified - Shoham
            "Stratified Wavy" : 7 ,\
            ### Falling Film - Shoham
            "Falling Film" : 8
            }

flow_dic_shoham_short = {
            ### Annular - Shoham
            "Annular" : "AN" ,\
            ### Bubbles - Shoham
            "Bubbles" : "BU" ,\
            ### Churn - Shoham
            "Churn" : "CH" ,\
            ###Dispersed Bubbles - Shoham 
            "Dispersed Bubbles" : "DB" ,\
            ### Enlogated Bubbles - Shoham
            "Elongated Bubbles" : "EB" ,\
            ### Slug - Shoham
            "Slug" : "SL" ,\
            ### Smooth Stratified - Shoham
            "Smooth Stratified" : "SS" ,\
            ### Wavy Stratified - Shoham
            "Stratified Wavy" : "SW" ,\
            ### Falling Film - Shoham
            "Falling Film" : "FF"
            }

col_dict_shoham = {
            0:"orange",\
            1:"red",\
            2:"cornflowerblue",\
            3:"indianred",\
            4:"yellow",\
            5:"blue",\
            6:"green",\
            7:"darkgreen",\
            8:"magenta"
            }

flow_dic_barnea = {
            ### Annular - Barnea
            "Annular" : 0 ,\
            ### Bubbles - Shoham
            "Dispersed" : 1 ,\
            ### Intermittent - Shoham
            "Intermittent" : 2 ,\
            ### - Shoham 
            "Stratified" : 3
            }

flow_dic_barnea_short = {
            ### Annular - Barnea
            "Annular" : "AN" ,\
            ### Bubbles - Shoham
            "Dispersed" : "DI" ,\
            ### Intermittent - Shoham
            "Intermittent" : "IN" ,\
            ### - Shoham 
            "Stratified" : "ST"
            }

col_dict_barnea = {
            0:"orange",\
            1:"red",\
            2:"blue",\
            3:"green"
            }

flow_dic_trallero = {
            ### Dispersion of oil in water over a water layer
            "DO_W&W" : 0 ,\
            ### Stratified flow with mixing at the interface
            "ST&MI" : 1 ,\
            ### Stratified
            "ST" : 2 ,\
            ### Emulsion of oil in water
            "O_W" : 3,\
            ### Dispersion of oil in water and water in oil
            "DW_O&DO_W" : 4,\
            ### Emulsion of water in oil
            "W_O" : 5
            }

col_dict_trallero = {
            0:"orange",\
            1:"red",\
            2:"green",\
            3:"indianred",\
            4:"yellow",\
            5:"blue"
            }

aux_pattern_shoham = {### Smooth Stratified
            "Smooth Stratified" : "Smooth Stratified" ,\
            "Stratified Smooth" : "Smooth Stratified" ,\
            "Stratified" : "Smooth Stratified" ,\
            ### Wavy Stratified
            "Stratified Wavy" : "Stratified Wavy" ,\
            "Stratified Wavy Transition to Annular" : "Stratified Wavy" ,\
            "Stratified wavy with dry pipe walls droplet atomization" : "Stratified Wavy" ,\
            ### Churn 
            "Churn" : "Churn" ,\
            "Churn-Froth" : "Churn" ,\
            "Churn-Slug" : "Churn" ,\
            "Churn-Turbulent" : "Churn" ,\
            ### Enlogated Bubbles
            "Enlogated Bubble" : "Elongated Bubbles" ,\
            "Elongated Bubble" : "Elongated Bubbles" ,\
            ### Slug
            "Slug" : "Slug" ,\
            "Slug Flow" : "Slug" ,\
            "Intermittent" : "Slug" ,\
            "Intermittent Flow" : "Slug" ,\
            "Intermittent flow" : "Slug" ,\
            "Plug" : "Slug" ,\
            "Plug Flow" : "Slug" ,\
            "Pseudo-Slug" : "Slug" ,\
            ### Annular
            "Annular" : "Annular" ,\
            "Wavy Annular" : "Annular" ,\
            "Wavy-Annular" : "Annular" ,\
            "Roll-Waves" : "Annular" ,\
            "Rolling Wavy" : "Annular" ,\
            "Semi-Annular" : "Annular" ,\
            "Stratified Annular" : "Annular" ,\
            ### Bubbles
            "Bubble" : "Bubbles" ,\
            "Bubbly" : "Bubbles" ,\
            "Agitated Bubbly" : "Bubbles" ,\
            "Cap-Bubbly" : "Bubbles" ,\
            "Cap-Bubby" : "Bubbles" ,\
            "Cap bubbly" : "Bubbles" ,\
            "Undisturbed Bubbly" : "Bubbles" ,\
            ###Dispersed Bubbles 
            "Dispersed Bubbles" : "Dispersed Bubbles" ,\
            "Dispersed Bubble" : "Dispersed Bubbles" ,\
            "Dispersed bubbles" : "Dispersed Bubbles" ,\
            "Dispersed" : "Dispersed Bubbles" ,\
            "Dispersed Bubbly" : "Dispersed Bubbles" ,\
            "Dual Continuous" : "Dispersed Bubbles" ,\
            #### Nao Classificados
            "Churn Bubbly" : "Churn" ,\
            "Enlogated Bubble with Dispersed Bubbles" : "Elongated Bubbles" ,\
            "Slug Annular" : "Slug" ,\
            "Slug-Annular" : "Slug" ,\
            "Messy-Slug" : "Slug" ,\
            "Stratified Annular to intermittent" : "Slug" ,\
            "Bubbly-Slug" : "Slug" ,\
            "Stratified Wavy transition to annular" : "Annular" ,\
            "Falling Film" : "Falling Film"
            }

aux_pattern_barnea = {### Stratified
            "Smooth Stratified" : "Stratified" ,\
            "Stratified Smooth" : "Stratified" ,\
            "Stratified" : "Stratified" ,\
            ### Wavy Stratified
            "Stratified Wavy" : "Stratified" ,\
            "Stratified Wavy Transition to Annular" : "Stratified" ,\
            "Stratified wavy with dry pipe walls droplet atomization" : "Stratified" ,\
            ### Churn 
            "Churn" : "Intermittent" ,\
            "Churn-Froth" : "Intermittent" ,\
            "Churn-Slug" : "Intermittent" ,\
            "Churn-Turbulent" : "Intermittent" ,\
            ### Enlogated Bubbles
            "Enlogated Bubble" : "Intermittent" ,\
            "Elongated Bubble" : "Intermittent" ,\
            ### Slug
            "Slug" : "Intermittent" ,\
            "Slug Flow" : "Intermittent" ,\
            "Intermittent" : "Intermittent" ,\
            "Intermittent Flow" : "Intermittent" ,\
            "Intermittent flow" : "Intermittent" ,\
            "Plug" : "Intermittent" ,\
            "Plug Flow" : "Intermittent" ,\
            "Pseudo-Slug" : "Intermittent" ,\
            "Roll-Waves" : "Intermittent" ,\
            ### Annular
            "Annular" : "Annular" ,\
            "Wavy Annular" : "Annular" ,\
            "Wavy-Annular" : "Annular" ,\
            "Rolling Wavy" : "Annular" ,\
            "Semi-Annular" : "Annular" ,\
            "Stratified Annular" : "Annular" ,\
            ### Bubbles
            "Bubble" : "Dispersed" ,\
            "Bubbly" : "Dispersed" ,\
            "Agitated Bubbly" : "Dispersed" ,\
            "Cap-Bubbly" : "Dispersed" ,\
            "Cap-Bubby" : "Dispersed" ,\
            "Cap bubbly" : "Dispersed" ,\
            "Undisturbed Bubbly" : "Dispersed" ,\
            ###Dispersed Bubbles 
            "Dispersed Bubbles" : "Dispersed" ,\
            "Dispersed Bubble" : "Dispersed" ,\
            "Dispersed bubbles" : "Dispersed" ,\
            "Dispersed" : "Dispersed" ,\
            "Dispersed Bubbly" : "Dispersed" ,\
            "Dual Continuous" : "Dispersed" ,\
            #### Nao Classificados
            "Churn Bubbly" : "Intermittent" ,\
            "Enlogated Bubble with Dispersed Bubbles" : "Intermittent" ,\
            "Slug Annular" : "Intermittent" ,\
            "Slug-Annular" : "Intermittent" ,\
            "Messy-Slug" : "Intermittent" ,\
            "Stratified Annular to intermittent" : "Intermittent" ,\
            "Bubbly-Slug" : "Intermittent" ,\
            "Stratified Wavy transition to annular" : "Annular" ,\
            "Falling Film" : "Intermittent"
            }