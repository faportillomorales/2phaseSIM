import math
import copy

class Owhfp:

    def __init__(self):
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
        # Definicao das constantes globais
        self.G = 9.81
        self.FILEO = 'OILWAT_1.OUT'
        self.DENO = float(831.3)
        self.DENW = float(1070.1)
        self.VISW = float(0.00076)
        self.VISO = float(0.00734)
        self.DI = float(0.08280)
        self.EPW = float(0.00001)
        self.SUROW = float(0.036)
        self.ISHEL = 0
        ANGD = 0.0
        self.ANG = ANGD * math.pi / 180.0
        self.VSOMIN = -2.00
        self.VSOMAX = 0.80
        self.VWSLMIN = -2.00
        self.VWSLMAX = 0.80
        self.DVOSL = 0.05
        self.DVWSL = 0.05

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
        FF_value = -TAUW * SW / AW + TAUO * SO / AO + TAUI * (SI / AW + SI / AO) - (self.DENW - self.DENO) * self.G * math.sin(self.ANG)

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
            A14 = -(self.DENW - self.DENO) * self.G * math.sin(self.ANG)
            
            if abs(DX) < EPS:
                break

            N += 1
            if N == 1:
                continue
            elif N > NMAX:
                IOERR.append('Subroutine HDF did not converge')
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
    def main(self):
        
        # Inicializacao das variaveis e listas
        HLD = 0.0
        VM = 0.0
        CW = 0.0
        R = 0.0

        # Validacao das variaveis de entrada
        # IOERR = []

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

        # Escrita de variaveis de entrada e opcoes selecionadas no arquivo de saida
        with open(self.FILEO, 'w') as ioutfile:
            ioutfile.write(f'\n')

        # Limites para as escalas de velocidade superficial logaritmica
        self.VOSL = self.VSOMIN - self.DVOSL

        KK = 0
        FLOW_PATTERN = []

        while True:

            K = 0
            VWSL = self.VWSLMIN - self.DVWSL
            self.VOSL += self.DVOSL
            VOS = 10 ** self.VOSL

            if self.VOSL > (self.VSOMAX + 0.1 * self.DVOSL):
                break

            while True:
                VWSL += self.DVWSL
                VWS = 10 ** VWSL

                if VWSL > (self.VWSLMAX + 0.1 * self.DVWSL):
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
                    PGOT = (TAUO * SO + TAUI * SI + self.DENO * AO * self.G * math.sin(self.ANG)) / AO
                    PGWT = (TAUW * SW - TAUI * SI + self.DENW * AW * self.G * math.sin(self.ANG)) / AW
                    DENM = RWH * self.DENW + ROH * self.DENO
                    VISM = RWH * self.VISW + ROH * self.VISO
                    REM = (DENM * VM * self.DI) / VISM
                    FM = 0.312 / REM**0.25

                    PGH = ((FM * DENM * VM**2) / (2 * self.DI)) + DENM * self.G * math.sin(self.ANG)

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
                    GST = 1 / DEN * (self.DENW - self.DENO) * self.G * math.cos(self.ANG) * A / SI
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
                    VOMH = math.sqrt((8/3) * DM * self.G * math.cos(self.ANG) * (self.DENW / self.DENO - 1) / FD)

                    RE = VM*self.DI*self.DENW/self.VISW 
                    
                    FD = self.FFF(RE, self.DI)

                    # if FD == None:
                    #     exit(999)
                    
                    EP = 2.*FD*VM**3./self.DI                                          
                    DM = 1.50*(0.73*(self.SUROW/self.DENW)**0.6*EP**(-0.4))/CW**3.5         
                    VWMH = math.sqrt((8./3.)*DM*self.G*math.cos(self.ANG)*(1-self.DENO/self.DENW)/FD)
                    
                    EP = 2.0*FO*VO**3./DOIL                                           
                    SK = ((self.VISO/self.DENO)**3./EP)**0.25                                   
                    DM = 0.73*(self.SUROW/self.DENO)**0.6*EP**(-0.4)                            
                    
                    if DM / SK < 2.0:
                        DM = 1.0 * self.SUROW * SK / (self.VISO * (((self.VISO / self.DENO) * EP)**0.25)) 
                    
                    DM = 0.68250*DM                                                   
                    VOOH = math.sqrt((8./3.)*DM*self.G*math.cos(self.ANG)*(self.DENW/self.DENO-1)/FO )            

                    EP = 2.*FW*VW**3./DW                                    
                    DM = 0.73*(self.SUROW/self.DENW)**0.6*EP**(-0.4)                  
                    DM = 11.3250*DM*CW**2.                               
                    VWWH = math.sqrt( (8./3.)*DM*self.G*math.cos(self.ANG)*(1-self.DENO/self.DENW)/FW )
                    
                    DM   = 2.*math.sqrt(self.SUROW*(self.VISW/self.DENW)/(25*self.DENW*VW**3.*(FW/2)**1.5))
                    DMP  = 25.*self.DENW*(self.VISW/self.DENW)**2/self.SUROW                           

                    if DM < DMP:
                        DM = copy.copy(DMP)                                  
                    
                    DM   = 1.740*DM/CW**7                                       
                    VWWL = math.sqrt((8.0/3.0)*DM*self.G*math.cos(self.ANG)*(1.0-self.DENO/self.DENW)/FW )
                    
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

                    FLOW_PATTERN.append(flo_pat)
                
                testeKK = KK == 3079 or KK == 3137 or KK == 3138 or KK == 3193 or KK == 3194 or KK == 3195
                
                if testeKK:
                    # input("Pressione Enter para continuar...")
                    print(f'VO =    {VO}      VOMH =     {VOMH}      VW =     {VW}      VWMH =     {VWMH}')
                    print(CRFI < 0,CRFV < 0,abs(self.ANG-0.0) < 1e-12 and abs(VW - VO) < 0.000010,VO >= VOOH and VW >= VWWH,VO >= VOMH)
                    print(VW < VWMH and VW > VO, VW < VWWL and VW > VO, VO >= VOOH and VW >= VWWH, VW >= VWMH and VW > VO, VO >= VOMH and VW < VO)
                    print(KK, FLOW_PATTERN[-1])
                    
                with open(self.FILEO, 'a') as ioutfile:
                    ioutfile.write(f"{KK:5d}  {VOS:12.5f}{VWS:12.5f}  {HLD:12.5f}      {FLOW_PATTERN[-1]:12s}{RW:12.5f}   {RWH:12.5f}   {PGOT:12.5f}   {PGWT:12.5f}   {PGH:12.5f}\n")
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
        # print('PROGRAM TERMINATED AND RESULTS ARE IN FILES: ST, STMI, DOWW, OW, DWODOW, WO .DATs AND', self.FILEO)

if __name__ == "__main__":
    O = Owhfp()
    O.main()
