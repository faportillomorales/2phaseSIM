from flowtechlib import *
from flowtechlib import exemples

def test(aa):
    print("Testando:",aa)
    parms = exemples.exemple_1_Barnea
    pat = Patterns(parms)
    pat.info()
    phe = Phenom(parms)
    phe.PhenomPatterns()
    print("Data Driven")
    dat = PhenomDataDriven(parms)
    dat.PhenomDataDrivenPatterns()
    print("Data Driven - Hybrid")
    dat_hyb = PhenomDataDrivenHybrid(parms, phe.pattern_str)
    dat_hyb.PhenomDataDrivenHybridPatterns()

if __name__ == "__main__" :
    test(30)