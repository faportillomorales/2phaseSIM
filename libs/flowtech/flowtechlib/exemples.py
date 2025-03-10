#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Set 09 20:23:09 2023

@author: LEMI Laboratory
"""
ext_point_vg = [0.021189299069938092, 0.778151250383643600, 0.7781512503836436, 5.30102999566398100, 5.3010299956639810, 0.17609125905568124, 5.301029995663981]
ext_point_vl = [0.021189299069938092, 0.021189299069938092, 0.7781512503836436, 0.01703333929878037, 0.7781512503836436, 5.00000000000000000, 0.146128035678238]

exemple_0_Barnea = {
        "fluid1" : "Water" ,
        "vel_min_liq" : "0.001" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.001139" ,
        "dens_liq" : "999" ,
        "fluid2" : "Air" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "10.0" ,
        "visc_gas" : "0.000017965" ,
        "dens_gas" : "1.2257" ,
        "inte_tens" : "0.0735" ,
        "diam" : "0.051" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Barnea 1986" ,
        "resol" : "150",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg
        }

exemple_0_Shoham = {
        "fluid1" : "Water" ,
        "vel_min_liq" : "0.001" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.001139" ,
        "dens_liq" : "999" ,
        "fluid2" : "Air" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "10.0" ,
        "visc_gas" : "0.000017965" ,
        "dens_gas" : "1.2257" ,
        "inte_tens" : "0.0735" ,
        "diam" : "0.051" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Shoham 2005" ,
        "resol" : "150",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg
        }

exemple_1_Barnea = {
        "fluid1" : "Water" ,
        "vel_min_liq" : "0.001" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.001139" ,
        "dens_liq" : "999" ,
        "fluid2" : "Air" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "100.0" ,
        "visc_gas" : "0.000017965" ,
        "dens_gas" : "1.2257" ,
        "inte_tens" : "0.0735" ,
        "diam" : "0.051" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Barnea 1986" ,
        "resol" : "250",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg
        }

exemple_1_Shoham = {
        "fluid1" : "Water" ,
        "vel_min_liq" : "0.001" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.001139" ,
        "dens_liq" : "999" ,
        "fluid2" : "Air" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "100.0" ,
        "visc_gas" : "0.000017965" ,
        "dens_gas" : "1.2257" ,
        "inte_tens" : "0.0735" ,
        "diam" : "0.051" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Shoham 2005" ,
        "resol" : "250",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg
        }

ext_point_vg = [0.0212265204359673, 0.0198566786226686, 0.0196787664835165, 0.0197955691642651, 0.021615997311828, 0.1001376458616010, 0.1008002261072260, 0.099161012345679, 0.0991445024390245, 0.0990221765517241, 0.498335714876033, 0.496499943005181, 0.496477635734072, 0.496609250289687, 0.496019351005484, 0.9932710176390780, 0.9924668196286480, 0.994772302816902, 0.992815677069198, 0.997524404137931, 1.984304748618780, 0.499640374485597, 0.993292483561643, 0.6949696577181210, 0.6946306861499370, 0.69407093, 0.697759017985611, 0.2987608655692730, 0.2994927776261940, 0.298087002747253, 0.296310411023622, 0.397137765027322, 0.3971672641509430, 0.397594107586207, 0.2160374230235780, 0.2102437243767310, 0.237879907202216, 0.5971265433789950, 0.5948242758620690, 0.595787500000001, 0.594798511789182, 1.191770953678480, 1.191780248313090, 1.489649635437880, 1.487802240, 1.50879173650794]
ext_point_vl = [0.0495510653950953, 0.0968294935437589, 0.1956438901098900, 0.4618641469740640, 0.919405997311828, 0.0511905101763908, 0.0974975536130537, 0.196289288065844, 0.5005195365853660, 0.9749956868965520, 0.048105479338843, 0.098194731865285, 0.206284006925208, 0.482315891077636, 0.971553687385741, 0.0506009972862959, 0.0888243541114059, 0.194058601408451, 0.472187597014925, 0.991669812413793, 0.192995495856354, 1.408367633744860, 1.350248631506850, 0.0490397906040269, 0.0891616149936468, 0.19551498, 0.455541237410072, 0.0499105116598079, 0.0998247544338335, 0.188870221153846, 0.496971582677165, 0.053120293715847, 0.0961364312668465, 0.192917700689655, 0.0505362926490984, 0.0982840138504154, 0.197062869806094, 0.0544205692541858, 0.0980871379310345, 0.200943542817680, 0.450652117891817, 0.101196666212534, 0.187537624831309, 0.180065951120163, 0.460220296, 1.45672045079365]

exemple_2_Barnea = {
        "fluid1" : "Oil" ,
        "vel_min_liq" : "0.01" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.045",
        "dens_liq" : "867.1" ,
        "fluid2" : "SF6" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "10.0" ,
        "visc_gas" : "0.0000167528016308555" ,
        "dens_gas" : "103" ,
        "inte_tens" : "0.0217" ,
        "diam" : "0.0508" ,
        "incl" : "5.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Barnea 1986" ,
        "resol" : "200",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg
        }

exemple_2_Shoham = {
        "fluid1" : "Oil" ,
        "vel_min_liq" : "0.01" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.045",
        "dens_liq" : "867.1" ,
        "fluid2" : "SF6" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "10.0" ,
        "visc_gas" : "0.0000167528016308555" ,
        "dens_gas" : "103" ,
        "inte_tens" : "0.0217" ,
        "diam" : "0.0508" ,
        "incl" : "5.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Shoham 2005" ,
        "resol" : "200",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg
        }

ext_point_vg = [0.0538557572815534, 0.0981175395189003, 0.1989996119610570, 0.4813064646324550, 0.9982572406015040, 0.0522415591098748, 0.0987327916666667, 0.1989448511821970, 0.5242102810218980, 0.9788730283400810, 0.0519339389736477, 0.0999962906815021, 0.198752422222222, 0.495655944, 0.985538670846395, 0.049456987500000, 0.096861495132128, 0.193792210013908, 0.496154597609562, 0.989329092369478, 0.0545236488439306, 0.0983824313453536, 0.197490031900139, 0.498504776556777, 0.985272180722891, 1.44798204504505, 0.0501028846641318, 0.100064794502618, 0.198337199722608, 0.479185692028986, 0.970499307086614, 1.44282843359375, 0.0523088168044078, 0.0988872579310345, 0.194743579670330, 0.517229655870445, 0.996810195918367, 1.46967642986425, 0.0501155567164179, 0.0994405061224489, 0.193676333787466, 0.524250479876161, 0.958925605263158, 0.0496795403556772, 0.0977879206349207, 0.198049392857143, 0.498752190972222, 1.03761473333333, 1.43139241573034, 0.196876760330578, 0.481698582995951]
ext_point_vl = [0.0172384299583911, 0.0203820171821306, 0.0172038358831711, 0.0178199833564494, 0.0200080263157895, 0.1000863004172460, 0.1002553208333330, 0.0994189680111266, 0.0974144927007299, 0.0993221538461538, 0.1980869334257980, 0.1983541015299030, 0.199177551388889, 0.194428472, 0.194808855799373, 0.298075034722222, 0.297297614742698, 0.297820656467316, 0.315333099601594, 0.301567570281124, 0.3986548395953760, 0.3973764216366160, 0.398580386962552, 0.400911307692308, 0.401128714859438, 0.39940463963964, 0.4991722015209120, 0.498940223821990, 0.497644105409153, 0.506095481884058, 0.503094952755906, 0.49614380859375, 0.6088033181818180, 0.6092440455172410, 0.609284637362637, 0.595781797570850, 0.598431000000000, 0.59578543438914, 0.697780425373134, 0.70333784489795900, 0.698549355585831, 0.701355275541795, 0.695455605263158, 1.0065417359781000, 1.0067668412698400, 1.004978845238090, 1.000195159722220, 1.00002173333333, 1.00115462172285, 1.198988231404960, 1.192388485829960]

exemple_3_Barnea = {
        "fluid1" : "Oil" ,
        "vel_min_liq" : "0.01" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.045",
        "dens_liq" : "867.1" ,
        "fluid2" : "SF6" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "10.0" ,
        "visc_gas" : "0.0000167528016308555" ,
        "dens_gas" : "103" ,
        "inte_tens" : "0.0217" ,
        "diam" : "0.0508" ,
        "incl" : "10.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Barnea 1986" ,
        "resol" : "200",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg
        }

exemple_3_Shoham = {
        "fluid1" : "Oil" ,
        "vel_min_liq" : "0.01" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.045",
        "dens_liq" : "867.1" ,
        "fluid2" : "SF6" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "10.0" ,
        "visc_gas" : "0.0000167528016308555" ,
        "dens_gas" : "103" ,
        "inte_tens" : "0.0217" ,
        "diam" : "0.0508" ,
        "incl" : "10.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Shoham 2005" ,
        "resol" : "200",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg
        }

ext_point_vl = [0.0186709981343284, 0.0196046666666667, 0.021380600000000, 0.0212228562992126, 0.0203275924528302, 0.0987651682070241, 0.1003526051931560, 0.100552307735832, 0.100006103448276, 0.100337641025641, 0.4964105558958650, 0.4962552338308460, 0.496449811594203, 0.496389960000000, 0.496742581613508, 0.993255816693945, 1.979272436395760, 2.202247748397430]
ext_point_vg = [0.0456056231343283, 0.0858923091247672, 0.185347504166667, 0.4113804724409450, 0.9112631811320760, 0.0436135194085028, 0.0967735502454106, 0.186139508847535, 0.455965644562334, 0.947340639194139, 0.0403238422664625, 0.0802294494195687, 0.123664995859213, 0.406240643076923, 0.962639836772983, 0.505770551554828, 0.730548911660777, 0.474548463141026]

exemple_4_Barnea = {
        "fluid1" : "Oil" ,
        "vel_min_liq" : "0.01" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.045",
        "dens_liq" : "867.1" ,
        "fluid2" : "SF6" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "10.0" ,
        "visc_gas" : "0.0000167528016308555" ,
        "dens_gas" : "103" ,
        "inte_tens" : "0.0217" ,
        "diam" : "0.0508" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Barnea 1986" ,
        "resol" : "200",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg
        }

exemple_4_Shoham = {
        "fluid1" : "Oil" ,
        "vel_min_liq" : "0.01" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.045",
        "dens_liq" : "867.1" ,
        "fluid2" : "SF6" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "10.0" ,
        "visc_gas" : "0.0000167528016308555" ,
        "dens_gas" : "103" ,
        "inte_tens" : "0.0217" ,
        "diam" : "0.0508" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Shoham 2005" ,
        "resol" : "200",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg
        }

exemple_0_LiqLiq = {
        "fluid1" : "Water" ,
        "vel_min_liq" : "0.0001" ,
        "vel_max_liq" : "8.0" ,
        "visc_liq" : "0.00076",
        "dens_liq" : "1070.1" ,

        "fluid2" : "Oil" ,
        "vel_min_gas" : "0.0001" ,
        "vel_max_gas" : "8.0" ,
        "visc_gas" : "0.00734" ,
        "dens_gas" : "831.3" ,

        "inte_tens" : "0.076" ,
        "diam" : "0.08280" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Trallero 1995" ,
        "resol" : "250",
        "ext_point_vl": [1.0],
        "ext_point_vg": [1.0]
        }

exemple_carlos = {
        "fluid1" : "Oil" ,
        "vel_min_liq" : "0.01" ,
        "vel_max_liq" : "3.0" ,
        "visc_liq" : "0.0404",
        "dens_liq" : "867.1" ,
        "fluid2" : "SF6" ,
        "vel_min_gas" : "0.02" ,
        "vel_max_gas" : "2.0" ,
        "visc_gas" : "0.0000170683016308555" ,
        "dens_gas" : "101.3" ,
        "inte_tens" : "0.032" ,
        "diam" : "0.0508" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Barnea 1986" ,
        "resol" : "200",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg
        }

#####################
### Testes HoldUp ###
#####################

ext_point_vg = [0.021189299069938092, 0.778151250383643600, 0.7781512503836436, 5.30102999566398100, 5.3010299956639810, 0.17609125905568124, 5.301029995663981]
ext_point_vl = [0.021189299069938092, 0.021189299069938092, 0.7781512503836436, 0.01703333929878037, 0.7781512503836436, 5.00000000000000000, 0.146128035678238]
ext_point_pres = [15.601302, 16.898255, 22.158158, 15.589447, 15.086777, 17.801735, 18.821229, 28.928172, 25.508107, 56.464231, 31.307555, 36.437796, 26.234234]
ext_point_houp = [ 0.746162,  0.879920,  0.958071,  0.895559,  0.501188,  0.868149,  0.490948,  0.850694,  0.599623,  0.302556,  0.650569,  0.752505,  0.759251]

exemple_1_Barnea_Holdup = {
        "fluid1" : "Oil" ,
        "vel_min_liq" : "1.01" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.0306681099711818",
        "dens_liq" : "850" ,
        "fluid2" : "SF6" ,
        "vel_min_gas" : "1.01" ,
        "vel_max_gas" : "10.0" ,
        "visc_gas" : "0.0000167528016308555" ,
        "dens_gas" : "101" ,
        "inte_tens" : "0.0217" ,
        "diam" : "0.0508" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Barnea 1986" ,
        "resol" : "50",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg,
        "ext_point_pres": ext_point_pres,
        "ext_point_houp": ext_point_houp
        }

exemple_2_Barnea_Holdup = {
        "fluid1" : "Agua" ,
        "vel_min_liq" : "0.01" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.001139",
        "dens_liq" : "999" ,
        "fluid2" : "Ar" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "10.0" ,
        "visc_gas" : "0.000017965" ,
        "dens_gas" : "1.2257" ,
        "inte_tens" : "0.0735" ,
        "diam" : "0.0254" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Barnea 1986" ,
        "resol" : "50",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg,
        "ext_point_pres": ext_point_pres,
        "ext_point_houp": ext_point_houp
        }


exemple_3_Barnea_Holdup = {
        "fluid1" : "Agua" ,
        "vel_min_liq" : "0.01" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.001139",
        "dens_liq" : "999" ,
        "fluid2" : "Nitrogênio" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "10.0" ,
        "visc_gas" : "0.00001787" ,
        "dens_gas" : "23.77" ,
        "inte_tens" : "0.0735" ,
        "diam" : "0.0254" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Barnea 1986" ,
        "resol" : "50",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg,
        "ext_point_pres": ext_point_pres,
        "ext_point_houp": ext_point_houp
        }


exemple_4_Barnea_Holdup = {
        "fluid1" : "Querosene" ,
        "vel_min_liq" : "0.01" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.0013",
        "dens_liq" : "825" ,
        "fluid2" : "Nitrogênio" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "10.0" ,
        "visc_gas" : "0.00001787" ,
        "dens_gas" : "23.77" ,
        "inte_tens" : "0.735" ,
        "diam" : "0.0254" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Barnea 1986" ,
        "resol" : "50",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg,
        "ext_point_pres": ext_point_pres,
        "ext_point_houp": ext_point_houp
        }

exemple_5_Barnea_Holdup = {
        "fluid1" : "Agua" ,
        "vel_min_liq" : "0.01" ,
        "vel_max_liq" : "30.0" ,
        "visc_liq" : "0.001139",
        "dens_liq" : "999" ,
        "fluid2" : "Ar" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "20.0" ,
        "visc_gas" : "0.000017965" ,
        "dens_gas" : "1.2257" ,
        "inte_tens" : "0.0735" ,
        "diam" : "0.0254" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Barnea 1986" ,
        "resol" : "50",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg,
        "ext_point_pres": ext_point_pres,
        "ext_point_houp": ext_point_houp
        }

exemple_6_Barnea_Holdup = {
        "fluid1" : "Agua" ,
        "vel_min_liq" : "0.01" ,
        "vel_max_liq" : "30.0" ,
        "visc_liq" : "0.001139",
        "dens_liq" : "999" ,
        "fluid2" : "Nitrogênio" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "20.0" ,
        "visc_gas" : "0.00001787" ,
        "dens_gas" : "23.77" ,
        "inte_tens" : "0.0735" ,
        "diam" : "0.0254" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Barnea 1986" ,
        "resol" : "50",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg,
        "ext_point_pres": ext_point_pres,
        "ext_point_houp": ext_point_houp
        }

exemple_7_Barnea_Holdup = {
        "fluid1" : "Querosene" ,
        "vel_min_liq" : "0.01" ,
        "vel_max_liq" : "30.0" ,
        "visc_liq" : "0.0013",
        "dens_liq" : "825" ,
        "fluid2" : "Nitrogênio" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "20.0" ,
        "visc_gas" : "0.00001787" ,
        "dens_gas" : "23.77" ,
        "inte_tens" : "0.735" ,
        "diam" : "0.0254" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Barnea 1986" ,
        "resol" : "50",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg,
        "ext_point_pres": ext_point_pres,
        "ext_point_houp": ext_point_houp
        }

exemple_8_Barnea_Holdup = {
        "fluid1" : "Oil" ,
        "vel_min_liq" : "0.01" ,
        "vel_max_liq" : "10.0" ,
        "visc_liq" : "0.0306681099711818",
        "dens_liq" : "850" ,
        "fluid2" : "SF6" ,
        "vel_min_gas" : "0.01" ,
        "vel_max_gas" : "10.0" ,
        "visc_gas" : "0.0000167528016308555" ,
        "dens_gas" : "101" ,
        "inte_tens" : "0.0217" ,
        "diam" : "0.0508" ,
        "incl" : "0.0" ,
        "data_driven" : "Random Forest" ,
        "fenomenol" : "Barnea 1986" ,
        "resol" : "50",
        "ext_point_vl": ext_point_vl,
        "ext_point_vg": ext_point_vg,
        "ext_point_pres": ext_point_pres,
        "ext_point_houp": ext_point_houp
        }