import flowtechlib as ftl
from flowtechlib import exemples

def gera_testes_Barnea(parms,name):
    pat = ftl.Patterns(parms)
    phe = ftl.Phenom(parms)
    V_L = parms["ext_point_vl"][:]
    V_G = parms["ext_point_vg"][:]
    patterns = []
    for i, vs_l in enumerate(V_L):
        for j, vs_g in enumerate(V_G):
            patterns.append(phe.Barnea1986_function_point(vs_l,vs_g))
    
    test_cases = []
    cont = -1
    for i, vs_l in enumerate(V_L):
        for j, vs_g in enumerate(V_G):
            cont += 1
            test_cases.append((vs_l, vs_g, patterns[cont]))

    with open("test_cases_"+name+'.'+'py', 'w') as f:
        f.write("test_cases_" + name + " = [" + str(test_cases[i]) + ",")
        for i in range(1,len(patterns)-1):
            f.write(str(test_cases[i]) + ",")
        f.write(str(test_cases[-1]) + " ] ")

if __name__ == "__main__" :
    gera_testes_Barnea(exemples.exemple_1_Barnea,'exemple_1_Barnea')
    gera_testes_Barnea(exemples.exemple_2_Barnea,'exemple_2_Barnea')
    gera_testes_Barnea(exemples.exemple_3_Barnea,'exemple_3_Barnea')
    gera_testes_Barnea(exemples.exemple_4_Barnea,'exemple_4_Barnea')
