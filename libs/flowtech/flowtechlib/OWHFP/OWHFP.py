import math
import copy

# Definicao das constantes globais
G = 9.81
PI = math.pi

# Funcao FFF
def FFF(EPF, RE, DHYD):
    IOERR = []

    # Validacao dos parametros
    if RE < 0:
        IOERR.append('FFF: Illegal value input for RE')
        return None

    # Definicao de AT e BT com base no valor de RE
    AT = 0.0
    BT = 1.0
    if RE <= 1502.0:
        AT = 1.0
        BT = 0.0

    # Calculo da funcao FFF
    FFF_value = AT*16*RE**(-1)+BT*(-3.6*math.log10((6.9/RE) +(EPF/(3.7*DHYD))**1.1))**(-2.0)

    return FFF_value

# Funcao EQU
def EQU(VWS, VOS, HLD, DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW):
    IOERR = []

    # Validacao dos parametros
    if VWS < 0:
        IOERR.append('EQU: Illegal value input for VWS')
        return None
    elif VOS < 0:
        IOERR.append('EQU: Illegal value input for VOS')
        return None
    elif HLD < 0:
        IOERR.append('EQU: Illegal value input for HLD')
        return None

    AA = 2.0 * HLD - 1.0
    AW = 0.25 * DI**2 * (PI - math.acos(AA) + AA * math.sqrt(1.0 - AA**2))
    A = (PI / 4.0) * DI**2
    AO = A - AW
    VW = VWS * A / AW

    VO = VOS * A / AO
    SW = DI * (PI - math.acos(AA))
    SO = PI * DI - SW
    SI = DI * math.sqrt(1.0 - AA**2)

    # Definicao dos diametros hidraulicos com base na fase mais rapida
    if VO >= VW:
        DW = 4.0 * AW / SW
        DOIL = 4.0 * AO / (SI + SO)
    else:
        DW = 4.0 * AW / (SI + SW)
        DOIL = 4.0 * AO / SO

    REW = abs(VW) * DW * DENW / VISW
    REO = abs(VO) * DOIL * DENO / VISO

    # A fase mais rapida determina o fator de atrito interfacial
    if VO >= VW:
        REI = copy.deepcopy(REO)
        ROI = copy.deepcopy(DENO)
    else:
        REI = copy.deepcopy(REW)
        ROI = copy.deepcopy(DENW)

    # Determinacao do esforco de cisalhamento

    FW = FFF(EPW, REW, DW)

    if FW == None:
        IOERR.append('Error returned from FFF call')
        return None

    FO = FFF(EPW, REO, DOIL)
    if FO == None:
        IOERR.append('Error returned from FFF call')
        return None

    if VO >= VW:
        FI = copy.deepcopy(FO)
    else:
        FI = copy.deepcopy(FW)

    TAUW = 0.5 * FW * DENW * VW**2
    TAUO = 0.5 * FO * DENO * VO**2
    TAUI = 0.5 * FI * ROI * (VO - VW) * abs(VO - VW)

    return (AA, AW, A, AO, VW, VO, SW, SO, SI, DW, DOIL, REW, REO, FW, FI, FO, TAUW, TAUO, TAUI)

# Funcao FF
def FF(VWS, VOS, HLD, DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW):

    IOERR = []
    if VWS < 0:
        IOERR.append('FF: Illegal value input for VWS')
        return None
    elif VOS < 0:
        IOERR.append('FF: Illegal value input for VOS')
        return None

    results = EQU(VWS, VOS, HLD, DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW)

    if results == None:
        IOERR.append('Error returned from EQU call')
        return None

    # Unpack the results
    (AA, AW, A, AO, VW, VO, SW, SO, SI, DW, DOIL, REW, REO, FW, FI, FO, TAUW, TAUO, TAUI) = results

    # Calculating the value of FF
    FF_value = -TAUW * SW / AW + TAUO * SO / AO + TAUI * (SI / AW + SI / AO) - (DENW - DENO) * G * math.sin(ANG)

    return FF_value

# Funcao HDF
def HDF(EPW, VWS, VOS, ANG, DENW, DENO, HLD1, IOERR, KK):

    EPS = 0.000001

    if VWS < 0 or VOS < 0.0:
        IOERR.append('HDF : Illegal value input for VWS')
        IOERR.append('HDF : Illegal value input for VOS')
        return None

    NMAX = 100
    N = 0
    X = HLD1 + 10 * EPS
    DX = (1.0 - X - EPS) / 50.0
    XMAX = 1.0

    while True:
        F1 = FF(VWS, VOS, X, DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW)
        X = X + DX

        if X >= XMAX:
            break

        HLD = copy.deepcopy(X)
        results = EQU(VWS, VOS, HLD, DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW)

        if results is None:
            IOERR.append('Error returned from EQU call')
            return None

        (AA, AW, A, AO, VW, VO, SW, SO, SI, DW, DOIL, REW, REO, FW, FI, FO, TAUW, TAUO, TAUI) = results

        F = FF(VWS, VOS, HLD, DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW)

        if F == None:
            return None

        if abs(DX) < EPS:
            break

        N += 1
        if N == 1:
            continue
        elif N > NMAX:
            IOERR.append('Subroutine HDF did not converge')
            return None

        SIGN = F * F1
        if SIGN == 0.0:
            break
        elif SIGN < 0.0:
            DX = - DX / 2.0
        else:
            continue

    return HLD

# Funcao principal
def main():
    # Definicao das variaveis globais (substituindo o COMMON)
    global DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW, ISHEL

    # Inicializacao das variaveis e listas
    HLD = 0.0
    VM = 0.0
    CW = 0.0
    # R = [0.0] * 200

    # Leitura dos arquivos
    FILEO = 'OILWAT_1.OUT'
    DENO = float(831.3)
    DENW = float(1070.1)
    VISW = float(0.00076)
    VISO = float(0.00734)
    DI = float(0.08280)
    EPW = float(0.00001)
    SUROW = float(0.036)
    ISHEL = 0

    # Validacao das variaveis de entrada
    IOERR = []

    if DI <= 0.0 or DENW < 0.0 or EPW < 0.0 or \
        DENO < 0.0 or VISW < 0.0 or VISO < 0.0 or \
        ISHEL < 0 or ISHEL > 1:

        IOERR.append('Illegal value input for DI')
        IOERR.append('Illegal value input for DENW')
        IOERR.append('Illegal value input for EPW')
        IOERR.append('Illegal value input for DENO')
        IOERR.append('Illegal value input for VISW')
        IOERR.append('Illegal value input for VISO')
        IOERR.append('Illegal value input for ISHEL')
        for err in IOERR:
            print(err)
        exit(999)

    # Calculos preliminares
    ALAMD = 100 * DI
    AK = 2 * PI / ALAMD
    ANGD = 0.0
    ANG = ANGD * PI / 180.0

    # Escrita de variaveis de entrada e opcoes selecionadas no arquivo de saida
    with open(FILEO, 'w') as ioutfile:
        ioutfile.write(f'\n')

    # Limites para as escalas de velocidade superficial logaritmica
    VSOMIN = -2.00
    VSOMAX = 0.80
    VWSLMIN = -2.00
    VWSLMAX = 0.80
    DVOSL = 0.05
    DVWSL = 0.05
    VOSL = VSOMIN - DVOSL

    KK = 0

    FLOW_PATTERN = []

    while True:

        K = 0
        VWSL = VWSLMIN - DVWSL
        VOSL += DVOSL
        VOS = 10 ** VOSL

        if VOSL > (VSOMAX + 0.1 * DVOSL):
            break

        while True:
            VWSL += DVWSL
            VWS = 10 ** VWSL

            if VWSL > (VWSLMAX + 0.1 * DVWSL):
                break
            K += 1
            KK += 1

            # Calculos das velocidades e fracoes
            VM = VWS + VOS
            CW = VWS / VM
            R = VWS / VOS

            HLD1 = 0.0
            HLDMAX = 0.999

            for I in range(1,3):
                # Chamada para a funcao HDF
                HLDN = HDF(EPW, VWS, VOS, ANG, DENW, DENO, HLD1, IOERR, KK)

                if HLDN == None:
                    exit(999)

                if HLDN >= HLDMAX:
                    break

                HLD = copy.deepcopy(HLDN)

                # Chamada para a funcao EQU
                results = EQU(VWS, VOS, HLD, DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW)

                # Processamento dos resultados retornados pela EQU
                (AA, AW, A, AO, VW, VO, SW, SO, SI, DW, DOIL, REW, REO, FW, FI, FO, TAUW, TAUO, TAUI) = copy.deepcopy(results)

                RW = AW / A
                RO = AO / A
                RWH = VWS / (VWS + VOS)
                ROH = 1 - RWH

                # Calculos de gradientes de pressao
                PGOT = (TAUO * SO + TAUI * SI + DENO * AO * G * math.sin(ANG)) / AO
                PGWT = (TAUW * SW - TAUI * SI + DENW * AW * G * math.sin(ANG)) / AW
                DENM = RWH * DENW + ROH * DENO
                VISM = RWH * VISW + ROH * VISO
                REM = (DENM * VM * DI) / VISM
                FM = 0.312 / REM**0.25

                PGH = ((FM * DENM * VM**2) / (2 * DI)) + DENM * G * math.sin(ANG)

                # Derivada da equacao de momentum combinada
                HLD1 = copy.deepcopy(HLD)

                DEN = DENW / RW + DENO / RO

                E = (-((FF(VWS, VOS, HLD + 0.0001/2, DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW)
                      - FF(VWS, VOS, HLD - 0.0001/2, DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW))
                      / 0.0001) * PI / (4 * math.sqrt(1 - AA**2))) / DEN

                B = (((FF(VWS + (VWS * 0.001)/2, VOS, HLD, DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW)
                     - FF(VWS - (VWS * 0.001)/2, VOS, HLD, DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW))
                     / (VWS * 0.001))
                     - ((FF(VWS, VOS + (VOS * 0.001)/2, HLD, DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW)
                        - FF(VWS, VOS - (VOS * 0.001)/2, HLD, DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW))
                        / (VOS * 0.001))) / (2 * DEN)

                # Contribuicao de sombreamento (sheltering)
                if ISHEL == 1:
                    CS = 0.0730
                else:
                    CS = 0.0

                if VO >= VW:
                    ROI = copy.deepcopy(DENO)
                else:
                    ROI = copy.deepcopy(DENW)

                SHEL = ROI * pow((VO - VW), 2) * CS * A * (1/AW + 1/AO) / DEN
                # Calculo dos criterios de estabilidade
                FUSS = pow((E/(2*B) - ((DENW * VW / RW + DENO * VO / RO) / DEN)), 2)
                GST = 1 / DEN * (DENW - DENO) * G * math.cos(ANG) * A / SI
                AUST = DENW * DENO * pow((VO - VW), 2) / (pow(DEN,2) * RW * RO)
                SIST = SUROW * A * pow(AK, 2) / (SI * DEN)
                CRFV = FUSS + AUST - GST - SIST + SHEL
                CRFI = AUST - GST - SIST

                # Calculo do tamanho maximo de gotas de agua
                RE = VM * DI * DENO / VISO

                FD = FFF(EPW, RE, DI)

                if FD == None:
                    exit(999)

                EP = 2 * FD * pow(VM, 3) / DI
                SK = (pow((pow((VISO / DENO), 3) / EP), 0.25))
                DM = 0.73 * (SUROW / DENO)**0.6 * EP**(-0.4)
                if DM / SK < 2:
                    DM = 1.0 * SUROW * SK / (VISO * (((VISO / DENO) * EP)**0.25))

                DM = 592.620 * DM * CW**1.832
                VOMH = math.sqrt((8/3) * DM * G * math.cos(ANG) * (DENW / DENO - 1) / FD)

                RE = VM*DI*DENW/VISW 
                
                FD = FFF(EPW, RE, DI)

                if FD == None:
                    exit(999)
                
                EP = 2.*FD*VM**3./DI                                          
                DM = 1.50*(0.73*(SUROW/DENW)**0.6*EP**(-0.4))/CW**3.5         
                VWMH = math.sqrt((8./3.)*DM*G*math.cos(ANG)*(1-DENO/DENW)/FD)
                
                EP = 2.0*FO*VO**3./DOIL                                           
                SK = ((VISO/DENO)**3./EP)**0.25                                   
                DM = 0.73*(SUROW/DENO)**0.6*EP**(-0.4)                            
                
                if DM / SK < 2.0:
                    DM = 1.0 * SUROW * SK / (VISO * (((VISO / DENO) * EP)**0.25)) 
                
                DM = 0.68250*DM                                                   
                VOOH = math.sqrt((8./3.)*DM*G*math.cos(ANG)*(DENW/DENO-1)/FO )            

                EP = 2.*FW*VW**3./DW                                    
                DM = 0.73*(SUROW/DENW)**0.6*EP**(-0.4)                  
                DM = 11.3250*DM*CW**2.                               
                VWWH = math.sqrt( (8./3.)*DM*G*math.cos(ANG)*(1-DENO/DENW)/FW )
                
                DM   = 2.*math.sqrt(SUROW*(VISW/DENW)/(25*DENW*VW**3.*(FW/2)**1.5))
                DMP  = 25.*DENW*(VISW/DENW)**2/SUROW                           

                if DM < DMP:
                    DM = copy.deepcopy(DMP)                                  
                
                DM   = 1.740*DM/CW**7                                       
                VWWL = math.sqrt( (8./3.)*DM*G*math.cos(ANG)*(1.-DENO/DENW)/FW )
                
                if CRFI < 0:
                    flo_pat = 'ST&MI'
                    if CRFV < 0:
                        flo_pat = 'ST'
                        if abs(ANGD-0.0) < 1e-12 and abs(VW - VO) < 0.000010:
                            flo_pat = 'ST&MI'
                    if VO >= VOOH and VW >= VWWH:
                        flo_pat = 'DW_O&DO_W'
                    if VO >= VOMH:
                        flo_pat = 'W_O'
                    if KK == 3079: print('Aqui')
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

                FLOW_PATTERN.append(flo_pat)
            if KK >= 3192 and KK <= 3195:
                # input("Pressione Enter para continuar...")
                print(KK, FLOW_PATTERN[-1])

            with open(FILEO, 'a') as ioutfile:
                ioutfile.write(f"{KK:5d}  {VOS:12.5f}{VWS:12.5f}  {HLD:12.5f}      {FLOW_PATTERN[-1]:12s}{RW:12.5f}  {RWH:12.5f}  {PGOT:12.5f}  {PGWT:12.5f}  {PGH:12.5f}\n")
                # ioutfile.write(f"{KK:5d} {'    '} {FLOW_PATTERN[-1]:12s}\n")

            BVOS = math.log10(VOS)
            BVWS = math.log10(VWS)
            PGOTL = math.log10(PGOT)
            PGHL = math.log10(PGH)

            # Escrevendo resultados em arquivos de acordo com o padrao de fluxo
            # if FLOW_PATTERN[-1] == 'ST':
            #     with open('ST.DAT', 'a') as ioutfile1:
            #         ioutfile1.write(f"{BVOS:10.4f} {BVWS:10.4f} {PGOTL:10.4f}\n")
            # elif FLOW_PATTERN[-1] == 'ST&MI':
            #     with open('STMI.DAT', 'a') as ioutfile2:
            #         ioutfile2.write(f"{BVOS:10.4f} {BVWS:10.4f} {PGOTL:10.4f}\n")
            # elif FLOW_PATTERN[-1] == 'DO_W&W':
            #     with open('DOWW.DAT', 'a') as ioutfile3:
            #         ioutfile3.write(f"{BVOS:10.4f} {BVWS:10.4f} {PGHL:10.4f}\n")
            # elif FLOW_PATTERN[-1] == 'O_W':
            #     with open('OW.DAT', 'a') as ioutfile4:
            #         ioutfile4.write(f"{BVOS:10.4f} {BVWS:10.4f} {PGHL:10.4f}\n")
            # elif FLOW_PATTERN[-1] == 'DW_O&DO_W':
            #     with open('DWODOW.DAT', 'a') as ioutfile5:
            #         ioutfile5.write(f"{BVOS:10.4f} {BVWS:10.4f} {PGHL:10.4f}\n")
            # elif FLOW_PATTERN[-1] == 'W_O':
            #     with open('WO.DAT', 'a') as ioutfile6:
            #         ioutfile6.write(f"{BVOS:10.4f} {BVWS:10.4f} {PGHL:10.4f}\n")

    # Fechamento dos arquivos de saida e termino do programa
    # print('PROGRAM TERMINATED AND RESULTS ARE IN FILES: ST, STMI, DOWW, OW, DWODOW, WO .DATs AND', FILEO)

if __name__ == "__main__":
    main()