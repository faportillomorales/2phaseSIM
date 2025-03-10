import pytest
import flowtechlib as ftl
from flowtechlib import exemples
import numpy as np
from . import test_cases_exemple_1_Barnea as t1
from . import test_cases_exemple_2_Barnea as t2
from . import test_cases_exemple_3_Barnea as t3
from . import test_cases_exemple_4_Barnea as t4

parms_test_1 = exemples.exemple_1_Barnea
pat_test_1 = ftl.Patterns(parms_test_1)
phe_test_1 = ftl.Phenom(parms_test_1)
def barnea1986_test_1(vs_l, vs_g):
    pattern = phe_test_1.Barnea1986_function_point(vs_l, vs_g)
    return pattern

def shoham2005_test_1(vs_l, vs_g):
    pattern = phe_test_1.Shoham2005_function_point(vs_l, vs_g)
    return pattern

parms_test_2 = exemples.exemple_2_Barnea
pat_test_2 = ftl.Patterns(parms_test_2)
phe_test_2 = ftl.Phenom(parms_test_2)
def barnea1986_test_2(vs_l, vs_g):
    pattern = phe_test_2.Barnea1986_function_point(vs_l, vs_g)
    return pattern

parms_test_3 = exemples.exemple_3_Barnea
pat_test_3 = ftl.Patterns(parms_test_3)
phe_test_3 = ftl.Phenom(parms_test_3)
def barnea1986_test_3(vs_l, vs_g):
    pattern = phe_test_3.Barnea1986_function_point(vs_l, vs_g)
    return pattern

parms_test_4 = exemples.exemple_4_Barnea
pat_test_4 = ftl.Patterns(parms_test_4)
phe_test_4 = ftl.Phenom(parms_test_4)
def barnea1986_test_4(vs_l, vs_g):
    pattern = phe_test_4.Barnea1986_function_point(vs_l, vs_g)
    return pattern

testar_caso = 2
if testar_caso == 1:
    print('Testing case 1: Barnea')
    @pytest.mark.parametrize("vs_l, vs_g, expected", t1.test_cases_exemple_1_Barnea)
    def test_barnea1986(vs_l, vs_g, expected):
        result_pattern = barnea1986_test_1(vs_l,vs_g)
        assert result_pattern == expected

elif testar_caso == 2:
    print('Testing case 2: Barnea')
    @pytest.mark.parametrize("vs_l, vs_g, expected", t2.test_cases_exemple_2_Barnea)
    def test_barnea1986(vs_l, vs_g, expected):
        result_pattern = barnea1986_test_2(vs_l,vs_g)
        assert result_pattern == expected
    
elif testar_caso == 3:
    print('Testing case 3: Barnea')     
    @pytest.mark.parametrize("vs_l, vs_g, expected", t3.test_cases_exemple_3_Barnea)
    def test_barnea1986(vs_l, vs_g, expected):
        result_pattern = barnea1986_test_3(vs_l,vs_g)
        assert result_pattern == expected
        
elif testar_caso == 4:
    print('Testing case 4: Barnea')
    @pytest.mark.parametrize("vs_l, vs_g, expected", t4.test_cases_exemple_4_Barnea)
    def test_barnea1986(vs_l, vs_g, expected):
        result_pattern = barnea1986_test_4(vs_l,vs_g)
        assert result_pattern == expected