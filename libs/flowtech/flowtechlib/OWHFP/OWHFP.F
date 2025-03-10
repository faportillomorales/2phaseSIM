C    ****************************************************************
C    DETERMINATION OF OIL WATER FLOW PATTERN IN HORIZONTAL PIPELINES
C    WRITTEN BY: Jose Luis Trallero
C    REVISED BY: Oscar M.H.Rodriguez     LAST REVISION :January, 2001
C               *  * TULSA UNIVERSITY FLUID FLOW PROJECTS   *  *
C    ****************************************************************
C    This program determines oil-water flow pattern using viscous
C    and inviscid Kelvin Helmholtz method. The SI unit system is used.
C
C                            REFERENCES
C                            ----------
C    1.Barnea & Taitel (1994) "Interfacial And Structural Stability of    
C      Separated Flow",Annual Reviews in Multiphase Flow (Edited by       
C      Hestroni, G.), Pergamon, 387-414.                                  
C    2.Trallero, J.L.:"Oil-Water Flow Patterns in Horizontal Pipes" Ph.D. 
C      Dissertation, The University of Tulsa, 1995.                       
C                                                                         
C    *****************************************************************    
C                       INPUT/OUTPUT LOCAL FILE VARIABLES                 
C                       ---------------------------------                 
C    *IOERR    = Output file for error messages when input values,        
C               passed to the subroutine, are out of range.               
C    *IOUTFILE = Output file for calculation results.                     
C     INFILE   = Template input file.                                     
C                                                                         
C                                    SUBPROGRAM CALLED                    
C                                    -----------------                    
C    HDF     = Determination of the equilibrium liquid level.             
C    EQU     = Determination of the equilibrium geometrical parameters.   
C    FF      = Calculation of RHS of combined momentum equation.          
C    FFF     = Calculation of wall friction factor.                       
C                                                                         
C                                VARIABLE DESCRIPTION                     
C                                --------------------                     
C        A       = Total flow area, (m^2)                                 
C        AK      = Wave number, (1/m)                                     
C        ALAMD   = Wave length, (m)                                       
C        AO      = Oil flow area,  (m^2)                                  
C        AUST    = Velocity term of stability criterion                   
C        AW      = Water flow area, (m^2)                                 
C        B       = Dispersion relation coefficient                        
C        ANG     = Inclination angle, (rad.)                              
C        ANGD    = Inclination angle, (deg.)                              
C        C       = Friction factor coefficient                            
C        CRFI    = Inviscid Kelvin-Helmholtz criterion                    
C        CRFV    = Viscous Kelvin-Helmholtz criterion                     
C        CS      = Sheltering coefficient                                 
C        CW      = Injected water fraction, (-)                           
C        DEN     = Mixture density, (kg/m^3)                              
C     ***DENM    = Mixture density, (kg/m^3)    
C       *DENO    = Oil density, (kg/m^3)                                  
C       *DENW    = Water density, (kg/m^3)                      
C       *DI      = Pipe diameter, (m)                                     
C        DM      = Drop diameter, (m)                                     
C        DMP     = Tiniest drop diameter,  (m)                            
C        DOIL    = Oil hydraulic diameter,  (m)                           
C        DVOSL   = Differential oil superficial velocity, (m/s)           
C        DVWSL   = Differential water superficial velocity, (m/s)         
C        DW      = Water hydraulic diameter, (m)                          
C        E       = Dispersion relation coefficient                        
C        EN      = Friction factor correlation exponent                   
C        EP      = Dissipation rate per unit area (m^2/s^3)
C        **EPW   = Pipe Wall Roughness (m)       
C        F       = RHS of the momentum equation, (pa/m)                   
C        FD      = Friction factor, (-)                                   
C        FI      = Interfacial friction factor, (-)                       
C        FILEO   = Name of output file
C     ***FM      = Mixture friction factor (-)                                    
C        FPAT(I) = Oil-water flow pattern names                           
C        FO      = Oil friction factor, (-)                               
C        FUSS    = Viscous term of stability criterion                    
C        FW      = Water friction factor                                  
C        G       = Gravitational acceleration, (9.81 m/s^2)               
C        GST     = Gravity term of stability criterion                    
C        HH      = Pseudo oil level, (m)                                  
C        HLD1,HLD(l)=Equilibrium layer height variables, (-)              
C        HLDMAX  = Maximum equilibrium layer height in stratified         
C                  flow, (-)                                              
C        HLDN    = Equilibrium layer height in stratified flow, (-)       
C        IERR    = Error code (0=OK, 1=input variables out of range,      
C                                    2=Extrapolation occurring)           
C        ISHEL   = Sheltering coefficient index    
C     ***PGH     = Pressure gradient according to homogeneous model, (pa/m)
C     ***PGOT    = Oil Pressure gradient according to two-fluid model, (pa/m)
C     ***PGWT    = Water Pressure gradient according to two-fluid model, (pa/m)                   
C        R       = Water oil ratio                                        
C        RE      = Reynolds number, (-)
C     ***REM     = Reynolds number of the mixture, (-)                                      
C        REO     = Reynolds number of oil flow, (-)                       
C        REW     = Reynolds number of water flow, (-)                     
C        RO      = Ratio of oil flow area to total flow area, (-)
C     ***ROH     = Ratio of oil flow area to total flow area, homogeneous model, (-)
C        RW      = Ratio of water flow area to total flow area, (-)       
C     ***RWH     = Ratio of oil flow area to total flow area, homogeneous model, (-)
C        SHEL    = Sheltering term of stability criterion                 
C        SI      = Interface length, (m)                                  
C       *SUROW   = Oil-water interfacial tension, (N/m)                   
C        SIST    = Surface tension term of stability criterion            
C        SK      = Kolmogorov microscale                                  
C        SO      = Oil wetted perimeter, (m)                              
C        SW      = Water wetted perimeter, (m)                            
C        TAUI    = Interfacial shear stress, (pa)                         
C        TAUO    = Oil shear stress, (pa)                                 
C        TAUW    = Water shear stress, (pa)                               
C     ***VISM    = Mixture viscosity,  (cp)
C       *VISO    = Oil viscosity,    (cp)
C       *VISW    = Water viscosity,  (cp)
C        VM      = Mixture velocity, (m/s)                                
C        VO      = Actual oil velocity, (m/s)                             
C        VOMH    = Transition velocity based on maximum water             
C                  droplet size and mixture velocity, (m/s)               
C        VOOH    = Transition velocity based on maximum water             
C                  droplet size and oil velocity, (m/s)                   
C        VOS     = Superficial oil velocity, (m/s)                        
C        VOSL    = Superficial oil velocity limit, (m/s)                  
C        VSOMAX  = Maximum logarithmic oil superficial velocity,(m/s)     
C        VSOMIN  = Minimum logarithmic oil superficial velocity,(m/s)     
C        VSWMAX  = Maximum logarithmic water superficial velocity,(m/s)   
C        VSWMIN  = Minimum logarithmic water superficial velocity,(m/s)   
C        VW      = Actual water velocity, (m/s)                           
C        VWMH    = Transition velocity based on maximum oil droplet       
C                  size and mixture velocity, (m/s)                       
C        VWS     = Superficial water velocity, (m/s)                      
C        VWSL    = Superficial water velocity limit, (m/s)                
C        VWWH    = Transition velocity based on maximum oil droplet       
C                  size and water velocity, (m/s)                         
C        VWWL    = Transition velocity based on minimum oil droplet       
C                  size and water velocity, (m/s)                         
C        NF(I)   = Oil water flow pattern nomenclature                    
C                  5     ST       Stratified flow                         
C                  4  ST & MI     Stratified flow with mixing at          
C                                 the interface                           
C                  1     DO/W & W Dispersion of oil in water over a       
C                                 water layer                             
C                  10 O/W         Emulsion of oil in water                
C                  11 DW/O & DO/W Dispersion of oil in water and water    
C                                 in oil                                  
C                  12 W/O         Emulsion of water in oil                
C        I,J,K,N = Do loop indices                                        
C        X,DX,   = Dummy variables                                        
C   (*   Indicates input variables
C    **  Variable added by O. Rodriguez, Jan/2001
C    *** Variable added by O. Rodriguez, Jun/2003)
C*******************************************************************   
      PROGRAM MAIN 
      IMPLICIT   DOUBLE PRECISION (A-H,O-Z)

C Oscar's modification
                               
      COMMON     /A/ DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW
      COMMON     /C/ ISHEL
      CHARACTER  FILEO*12,FPAT(12)*12, virgula*12, AA*2
      DIMENSION  HLD(5),NF(5), VM(200), CW(200), R(200)
      DATA G,PI/9.81,3.1415926540/

      AA = '##'
      FPAT(1) ='DO_W&W'
      FPAT(4) ='ST&MI'
      FPAT(5) ='ST'
      FPAT(10)='O_W '
      FPAT(11)='DW_O&DO_W'
      FPAT(12)='W_O'

      IOUTFILE1 = 1
      IOUTFILE2 = 2
      IOUTFILE3 = 3
      IOUTFILE4 = 4
      IOUTFILE5 = 5
      IOUTFILE6 = 6
      IOUTFILE  = 7
      INFILE    = 8
      IOERR     = 9

      OPEN (UNIT = INFILE, FILE = 'OWHFP.DAT',STATUS = 'OLD')
      OPEN (UNIT = IOERR,  FILE = 'ERROR.DAT',STATUS='UNKNOWN')

      READ(INFILE,1)FILEO
      READ(INFILE,2)DENO,DENW

C Oscar's modification

      READ(INFILE,5)VISW,VISO,DI,EPW,SUROW
      READ(INFILE,6)ISHEL

1     FORMAT(5(/),T10,A10) 
2     FORMAT(//,T10,F10.1,/,T10,F10.1) 
5     FORMAT(5(T10,F10.6,/)) 
6     FORMAT(/,T10,I2,/) 

      OPEN (UNIT = IOUTFILE, FILE = FILEO, STATUS = 'UNKNOWN')

      OPEN (UNIT = IOUTFILE1, FILE = 'ST.DAT', STATUS = 'UNKNOWN')
      OPEN (UNIT = IOUTFILE2, FILE = 'STMI.DAT', STATUS = 'UNKNOWN')
      OPEN (UNIT = IOUTFILE3, FILE = 'DOWW.DAT', STATUS = 'UNKNOWN')
      OPEN (UNIT = IOUTFILE4, FILE = 'OW.DAT', STATUS = 'UNKNOWN')
      OPEN (UNIT = IOUTFILE5, FILE = 'DWODOW.DAT', STATUS = 'UNKNOWN')
      OPEN (UNIT = IOUTFILE6, FILE = 'WO.DAT', STATUS = 'UNKNOWN')
C     -------------------------------------
C     Check input variables for valid range.
C     -------------------------------------
      IERR=0
      IF (DI .LE. 0.0) THEN
         WRITE (IOERR,*) ' Illegal value input for DI'
         IERR = 1
      ELSEIF (DENW .LT. 0.0) THEN
         WRITE (IOERR,*) ' Illegal value input for DENW'
         IERR = 1 
      ELSEIF (EPW .LT. 0.0) THEN
         WRITE (IOERR,*) ' Illegal value input for EPW'
         IERR = 1
      ELSEIF (DENO .LT. 0.0) THEN
         WRITE (IOERR,*) ' Illegal value input for DENO'
         IERR = 1
      ELSEIF (VISW .LT. 0.0) THEN
         WRITE (IOERR,*) ' Illegal value input for VISW'
         IERR = 1
      ELSEIF (VISO .LT. 0.0) THEN
         WRITE (IOERR,*) ' Illegal value input for VISO'
         IERR = 1
      ELSEIF (ISHEL .LT. 0 .OR. ISHEL .GT. 1) THEN
         WRITE (IOERR,*) ' Illegal value input for ISHEL'
         IERR = 1
      ENDIF
      IF (IERR .EQ. 1) GOTO 999                                        
C    ------------------------                                          
C    Preliminary calculations                                          
C    ------------------------                                          
      ALAMD = 100 * DI
      AK    = 2*PI/ALAMD
      ANGD  = 0.0
      ANG   = ANGD*PI/180.0
C    ----------------------------------------------                       
C    Print selected option and input flow variables                       
C    ----------------------------------------------                       
C Oscar's modification
c       WRITE(IOUTFILE,9)
c 9     FORMAT(T5, I2, T15, T3)

      WRITE(IOUTFILE,9)ISHEL,DI,ANGD,EPW,DENW,DENO,VISO,VISW,SUROW
9     FORMAT( '#',/,'#',T5,'SELECTED OPTIONS', /,
     +    '#',T5,16('-'),/,
     +    '#',T5,'ISHEL = ',I2,/'#'/'#'/,
     +    '#',T5,'INPUT FLOW VARIABLES',/,
     +    '#',T5,20('-'),/,
     +    '#',T5,'DI    = ',F9.6,' m',/,
     +    '#',T5,'ANGD  = ',F9.6,' deg.',/,
C Oscar's modification^M
     +    '#',T5,'EPW   = ',F9.6,' m.',/,
     +    '#',T5,'DENW  = ',F9.4,' Kg/m^3'/ ,
     +    '#',T5,'DENO  = ',F9.4,' Kg/m^3',/,
     +    '#',T5,'VISO  = ',F9.4,' cp',/,
     +    '#',T5,'VISW  = ',F9.6,' cp',/,
     +    '#',T5,'SUROW = ',F9.6,' dyne/cm',/'#'/'#'/'#',
     +    T15,'INTERFACIAL LINEAR STABILITY ANALYSIS: RIGOROUS SOLUTION'
     + , /'#'/'#', 35X, 'COMPLETE TWO-FLUID MODEL', /'#'/'#'/'#'/,
     +        '#',T3, 'K    VOS     VWS        HLDUP    FLOW PATTERN RW
     +RWH    PGOT      PGWT     PGH', /'#')

C    ------------------------------------------------------             
C    Limits for the logarithmic superficial velocity scales             
C    ------------------------------------------------------             
      VSOMIN  = -2.00                                                   
      VSOMAX  = 0.80                                                    
      VWSLMIN = -2.00                                                   
      VWSLMAX = 0.80                                                    
      DVOSL   = 0.05                                                    
      DVWSL   = 0.05                                                    
      VOSL    = VSOMIN - DVOSL                                          
C    *****************************************************************  
C    The following procedure will be repeated for every VWS, VOS pairs  
C    *****************************************************************  
      KK = 0                                                            
10    K = 0                                                             
      VWSL = VWSLMIN - DVWSL                                            
      VOSL = VOSL + DVOSL                                               
      VOS  = 10**(VOSL)                                                 
      IF ( VOSL .GT. (VSOMAX + 0.1*DVOSL) )  GOTO 90

20    VWSL = VWSL + DVWSL                                               
      VWS  = 10**VWSL                                                   
      IF ( VWSL .GT. (VWSLMAX + .1*DVWSL) )   GOTO 90                   
      K = K + 1                                                         
      KK = KK + 1
C    -----------------------------------------------------------        
C    Calculate mixture velocity VM, injected water fraction CW &        
C    water-oil ratio R                                                  
C    -----------------------------------------------------------        
      VM(K)  = VWS + VOS                                                
      CW(K)  = VWS/VM(K)                                                
      R(K)   = VWS/VOS                                                  

      HLD1   = 0.00                                                     
      HLDMAX = 0.999                                                    
      I      = 0                                                        
30    I = I + 1                                                         
C    -----------------------------------------------                    
C    No more than three roots for hl/di are accepted                    
C    -----------------------------------------------                    
      IF (I .GT. 3) THEN                                                
          WRITE (*,*) ' I > 3'                                          
          STOP                                                          
      ENDIF                                                             
C    --------------------------------------                             
C    Calculate the equilibrium liquid level                             
C    --------------------------------------                             
      CALL HDF(EPW,VWS,VOS,ANG,DENW,DENO,HLD1, HLDN,IOERR,IERR,KK)      
      IF (HLDN .GE. HLDMAX)  GOTO 70
C    -------------------------------------------------------------------
C    Calculate the geometrical variables for a given liquid level, hl/di
C    -------------------------------------------------------------------
      HLD(I) = HLDN                                                     
      CALL EQU (VWS, VOS, HLD(I), HH, AW, A, AO, VW, VO, SW, SO, SI,    
     +   DW, DOIL, REW, REO, FW, FI, FO, TAUW, TAUO, TAUI , IOERR, IERR)
      IF(IERR .EQ. 1) THEN                                              
        WRITE(IOERR,*)'Error returned from EQU call'                    
        GOTO 999                                                        
      ENDIF                    
C     ------------------------------------------------------------------
C     Holdup according to the Two-fluid model  --- O.Rodriguez, Jun/2003              
C    -------------------------------------------------------------------                                                                      
      RW = AW/A
      RO = AO/A
C    -----------------------------------------------------------------
C    Holdup according to the homogeneous model --- Rodriguez, Jun/2003              
C    -----------------------------------------------------------------
      RWH = VWS/(VWS+VOS)
      ROH = 1 - RWH
C    ------------------------------------------------------------
C    Calculate the Pressure gradient based on the Two-Fluid Model
C    O.Rodriguez, Jun/2003              
C    ------------------------------------------------------------
      PGOT = (TAUO*SO+TAUI*SI+DENO*AO*G*DSIN(ANG))/AO
      PGWT = (TAUW*SW-TAUI*SI+DENW*AW*G*DSIN(ANG))/AW
C    --------------------------------------------------------------
C    Calculate the Pressure gradient based on the Homogeneous Model
C    O.Rodriguez, Jun/2003              
C    --------------------------------------------------------------
      DENM = RWH*DENW+ROH*DENO
      VISM = RWH*VISW+ROH*VISO  
      REM  = (DENM*(VWS+VOS)*DI)/VISM
      FM   = 0.312/REM**0.25
      PGH  = ((FM*DENM*(VWS+VOS)**2)/(2*DI))+DENM*G*DSIN(ANG)
C    ---------------------------------------------------------            
C    Derivative of the RHS of the combined momentum equation              
C    ---------------------------------------------------------            
      HLD1 = HLD(I)
      DEN  = DENW/RW + DENO/RO
      E    = (-((FF(VWS,VOS,HLD(I)+0.0001/2.,IOERR,IERR)                
     +  -FF(VWS,VOS,HLD(I)-0.0001/2. , IOERR, IERR))  /0.0001) * PI     
     +  / (4.*DSQRT(1.-HH**2.)))/(DEN)                                  
      IF(IERR .EQ. 1) THEN                                              
        WRITE(IOERR,*)'Error returned from FF call'                     
        GOTO 999                                                        
      ENDIF                                                             

      B    = ((( FF(VWS+(VWS*0.001)/2,VOS,HLD(I) , IOERR, IERR)         
     +  - FF(VWS-(VWS*0.001)/2,VOS,HLD(I) ,IOERR, IERR)) / (VWS*0.001))-
     +  (( FF(VWS,VOS+(VOS*0.001)/2,HLD(I) , IOERR, IERR)- FF(VWS,VOS-  
     +   (VOS*0.001)/2,HLD(I) , IOERR, IERR)) / (VOS*0.001))) / (2.*DEN)
      IF(IERR .EQ. 1) THEN                                            
        WRITE(IOERR,*)'Error returned from FF call'                   
        GOTO 999                                                      
      ENDIF                                                           
C    -------------------------------------                            
C    Calculate the sheltering contribution                            
C    -------------------------------------                            
      IF (ISHEL .EQ. 1) THEN                                          
        CS  = 0.0730                                                  
      ELSE                                                            
        CS  = 0.0                                                     
      ENDIF                                                           
      IF (VO .GE. VW) THEN                                            
        ROI = DENO                                                    
      ELSE                                                            
        ROI = DENW                                                    
      ENDIF                                                           
      SHEL  = ROI*(VO-VW)*(VO-VW)*CS*A*(1/AW + 1/AO) / DEN            
C    -------------------------------------------------------------    
C    Calculate the viscous and inviscid Kelvin-Helmholtz stability    
C    criterion                                                        
C    -------------------------------------------------------------    
      FUSS = (E/(2.*B)-((DENW*VW/RW + DENO*VO/RO) / DEN))**2          
      GST  = 1./DEN*(DENW-DENO)*G*DCOS(ANG)*A/SI                      
      AUST = DENW*DENO*(VO-VW)**2/(DEN**2*RW*RO)                      
      SIST = SUROW*A*AK**2./(SI*DEN)                                  
      CRFV = FUSS + AUST - GST - SIST + SHEL                          
      CRFI =        AUST - GST - SIST                                 
C    ---------------------------------------------------------        
C    Maximum water droplet size based on VM and oil properties        
C    (continuous phase)                                               
C    ---------------------------------------------------------        
      RE = VM(K)*DI*DENO/VISO

C Oscar's Modification
                                             
      FD = FFF(EPW,RE,DI,IOERR, IERR)                                 
      IF(IERR .EQ. 1) THEN                                            
        WRITE(IOERR,*)'Error returned from FFF call'                  
        GOTO 999                                                      
      ENDIF                                                           
      EP = 2.0*FD*VM(K)**3./DI                                        
      SK = ((((VISO/DENO)**3.)/EP)**0.25)                             
      DM = 0.73*(SUROW/DENO)**0.6*EP**(-0.4)                          
      IF (DM/SK .LT. 2) DM  = 1.0*SUROW*SK/(VISO*(                    
     +                        ((VISO/DENO)*EP)**0.25))                
      DM  = 592.620*DM*CW(K)**1.832                                   
      VOMH= DSQRT( (8./3.)*DM*G*DCOS(ANG)*(DENW/DENO-1)/FD)           
C    -------------------------------------------------------------------- 
C    Maximum oil droplet size based on VM: Hinze Model. Note Dmax that is 
C    independent of the oil viscosity(Valid approximation up to 150 cp).  
C    -------------------------------------------------------------------- 
      RE = VM(K)*DI*DENW/VISW                                          

C Oscar's modification

      FD = FFF(EPW,RE,DI,IOERR,IERR)                                   
      IF(IERR .EQ. 1) THEN                                             
        WRITE(IOERR,*)'Error returned from FFF call'                   
        GOTO 999                                                       
      ENDIF                                                            
      EP = 2.*FD*VM(K)**3./DI                                          
      DM = 1.50*(0.73*(SUROW/DENW)**0.6*EP**(-0.4))/CW(K)**3.5         
      VWMH = DSQRT((8./3.)*DM*G*DCOS(ANG)*(1-DENO/DENW)/FD)            
C    -------------------------------------------------------------        
C    Maximum Water droplet size based on VO and the oil properties        
C    (continuous phase)                                                   
C    -------------------------------------------------------------        
      EP = 2.0*FO*VO**3./DOIL                                           
      SK = ((VISO/DENO)**3./EP)**0.25                                   
      DM = 0.73*(SUROW/DENO)**0.6*EP**(-0.4)                            
      IF (DM/SK .LT. 2.)DM=1.0*SUROW*SK/(VISO*(((VISO/DENO)*EP)**0.25)) 
      DM = 0.68250*DM                                                   
      VOOH = DSQRT((8./3.)*DM*G*DCOS(ANG)*(DENW/DENO-1)/FO )            
C    -------------------------------------------------------------------- 
C    Maximum oil droplet size based on VW: Hinze Model. Note that Dmax is 
C    independent of the oil viscosity(Valid approximation up to 150 cp).  
C    -------------------------------------------------------------------- 
      EP = 2.*FW*VW**3./DW                                    
      DM = 0.73*(SUROW/DENW)**0.6*EP**(-0.4)                  
      DM = 11.3250*DM*CW(K)**2.                               
      VWWH = DSQRT( (8./3.)*DM*G*DCOS(ANG)*(1-DENO/DENW)/FW ) 
C    ----------------------------------------------------------------     
C    Minimum oil droplet size based on VW: Levich Model (Wall effect)     
C    and the water properties (continuous phase)                          
C    ----------------------------------------------------------------     
      DM   = 2.*DSQRT(SUROW*(VISW/DENW)/(25*DENW*VW**3.*(FW/2)**1.5))
      DMP  = 25.*DENW*(VISW/DENW)**2/SUROW                           
      IF (DM  .LT.  DMP) DM = DMP                                  
      DM   = 1.740*DM/CW(K)**7                                       
      VWWL = DSQRT( (8./3.)*DM*G*DCOS(ANG)*(1.-DENO/DENW)/FW )     
C    -------------------------------------                         
C    Declaration of oil water flow pattern                         
C    -------------------------------------
      IF (-CRFI .GT. 0)  THEN
        NF(I) = 4
        IF (-CRFV .GT. 0)  THEN
          NF(I) = 5
          IF (ANGD .EQ. 0.0 .AND. DABS(VW-VO) .LT. 0.000010) THEN
            NF(I) = 4
          ENDIF
        ENDIF
        IF (VO .GE. VOOH .AND. VW .GE. VWWH) THEN 
            NF(I) = 11
        ENDIF
        IF (VO .GE. VOMH) THEN
            NF(I) = 12
        ENDIF
      ELSE                                                             
        IF (VW .LT. VWMH .AND. VW .GT. VO) THEN
           NF(I) = 1
        ENDIF
        IF (VW .LT. VWWL .AND. VW .GT. VO) THEN
           NF(I) = 4
        ENDIF
        IF (VO .GE. VOOH .AND. VW .GE. VWWH) THEN
           NF(I) = 11
        ENDIF
        IF (VW .GE. VWMH .AND. VW .GT. VO) THEN
           NF(I) = 10
        ENDIF
        IF (VO .GE. VOMH .AND. VW .LT. VO) THEN
           NF(I) = 12
        ENDIF
      ENDIF

      IF (KK == 3079 .or. KK == 3136 .or. KK == 3137 .or. 
     & KK == 3193 .or. KK == 3194 .or. KK == 3195) THEN
     
            WRITE(*,*) 'VO = ', VO, 'VOMH = ', VOMH, 
     &                 'VW = ', VW, ' VWMH = ', VWMH

            WRITE(*,*) -CRFI .GT. 0, -CRFV .GT. 0, 
     &       (ANGD .EQ. 0.0 .AND. DABS(VW-VO) .LT. 0.000010),
     &       VO .GE. VOOH .AND. VW .GE. VWWH, VO .GE. VOMH
                  
            WRITE(*,*) VW .LT. VWMH .AND. VW .GT. VO,
     &                 VW .LT. VWWL .AND. VW .GT. VO,
     &                 VO .GE. VOOH .AND. VW .GE. VWWH,                 
     &                 VW .GE. VWMH .AND. VW .GT. VO,
     &                 VO .GE. VOMH .AND. VW .LT. VO
     
            WRITE(*,*) KK,  FPAT(NF(I))
      ENDIF
c       WRITE(*,300) KK, FPAT(NF(I)), CRFI, CRFV, VO, VW, VOOH, VWWH

c 300   FORMAT ('CRFI, CRFV, VO, VW, VOOH, VWWH = ', 
c      & I5, A15, ES12.5, ES12.5, ES12.5, ES12.5, ES12.5, ES12.5)

C    ----------------------------------------------------------
C    Repeat the stability calculation for the other hl/di roots
C    ----------------------------------------------------------
      GOTO 30
C    -----------------------------------------
C    Print the results for every VWS, VOS pair
C    -----------------------------------------

70    WRITE (IOUTFILE,71) KK, VOS, VWS, HLD(I-1), FPAT(NF(I-1)),
     +                    RW, RWH, PGOT, PGWT, PGH

71    FORMAT (I5, 2X, F12.5, F12.5, 2X, F12.5, 3X, A15,5(F12.5,3X)) 

c 70    WRITE (IOUTFILE,71) KK, FPAT(NF(I-1))

c 71    FORMAT (I5, 3X, A15)

      BVOS = LOG10(VOS)
      BVWS = LOG10(VWS)
      PGOTL = LOG10(PGOT)
      PGHL = LOG(PGH)

      IF (FPAT(NF(I-1)) .EQ. 'ST') THEN
        WRITE (IOUTFILE1,80)  BVOS, BVWS, PGOTL
80      FORMAT (F10.4, 1X, F10.4, 1X, F10.4)
      ELSE IF (FPAT(NF(I-1)) .EQ. 'ST&MI') THEN
        WRITE (IOUTFILE2,81)  BVOS, BVWS, PGOTL
81      FORMAT (F10.4, 1X, F10.4, 1X, F10.4)
      ELSE IF (FPAT(NF(I-1)) .EQ. 'DO_W&W') THEN
        WRITE (IOUTFILE3,82)  BVOS, BVWS, +PGHL
82      FORMAT (F10.4, 1X, F10.4, 1X, F10.4)
      ELSE IF (FPAT(NF(I-1)) .EQ. 'O_W') THEN
        WRITE (IOUTFILE4,83)  BVOS, BVWS, +PGHL
83      FORMAT (F10.4, 1X, F10.4, 1X, F10.4)
      ELSE IF (FPAT(NF(I-1)) .EQ. 'DW_O&DO_W') THEN
        WRITE (IOUTFILE5,84)  BVOS, BVWS, PGHL
84      FORMAT (F10.4, 1X, F10.4, 1X, F10.4)
      ELSE IF (FPAT(NF(I-1)) .EQ. 'W_O') THEN
C        WRITE (IOUTFILE6,85)  BVOS, BVWS, PGHL
85      FORMAT (F10.4, 1X, F10.4, 1X, F10.4)
      ENDIF
C    *******************************************                          
C    Repeat the procedure for a new VWS,VOS pair                          
C    *******************************************                          
      GOTO 20
  90  CONTINUE
  
100   IF (VOSL .GE. (VSOMAX - 0.1*DVOSL) .AND. VWSL .GE.              
     +   (VWSLMAX - 0.1*DVWSL))  GOTO 110                             
      IF (VWSL.GE.(VWSLMAX+0.1*DVWSL)) GOTO 10                        
                                                                      
110   CONTINUE                                                        
                                                                      
C      WRITE(*,*)'PROGRAM TERMINATED AND RESULTS ARE IN FILES :
C     + ST, STMI, DOWW, OW, DWODOW, WO .DATs AND ', FILEO
999   STOP

      CLOSE(IOUTFILE1)
      CLOSE(IOUTFILE2)
      CLOSE(IOUTFILE3)
      CLOSE(IOUTFILE4)
      CLOSE(IOUTFILE5)
      CLOSE(IOUTFILE6)

      END PROGRAM                                                  
C    ****************************************************************     
C                DETERMINATION OF EQUILIBRIUM LIQUID LEVEL                
C    WRITTEN BY: Jose Luis Trallero                                       
C    REVISED BY: Avni Serdar Kaya     LAST REVISION :JUNE 12, 1997        
C               *  * TULSA UNIVERSITY FLUID FLOW PROJECTS   *  *          
C    ****************************************************************     
C    This subroutine determines the equilibrium liquid level by           
C    solving combined momentum equation. The SI unit system is used.      
C                                                                           
C                            REFERENCES                                   
C                            ----------                                   
C    1. Trallero, J.L.:"Oil-Water Flow Patterns in Horizontal Pipes"      
C       Ph.D. Dissertation, The University of Tulsa, 1995.                
C                                                                         
C    ****************************************************************     
C                                                                         
C               INPUT/OUTPUT LOCAL FILE VARIABLES                         
C               ---------------------------------                         
C    *IOERR = Output file for error messages when input values passed     
C             to the subroutine are out of range or error occurs in       
C             the calculation.                                            
C                                                                         
C                                                                         
C                        SUBPROGRAM CALLED                                 
C                        -----------------                                
C       EQU = This subroutine calculates equilibrium geometrical          
C            parameters.                                                  
C                                                                         
C                      VARIABLE DESCRIPTION                               
C                      --------------------                               
C            A     = Total flow area, (m^2)                               
C            AA    = Pseudo oil level, (m)                                
C            AO    = Oil flow area,  (m^2)                                
C            AUST  = Velocity term of stability criterion                 
C            AW    = Water flow area, (m^2)                               
C            *ANG  = Inclination angle, (rad.)                            
C            *DI   = Pipe diameter,  (m)                                  
C            *DENO = Oil density, (kg/m^3)                                
C            *DENW = Water density, (kg/m^3)                              
C            DX    = Equilibrium liquid level increment                   
C            DOIL  = Oil hydraulic diameter, (m)                          
C            DW    = Water hydraulic diameter, (m)                        
C            EPS   = Convergence criterion                                
C            F     = RHS of the momentum equation, (pa/m)                 
C            FI    = Interfacial friction factor, (-)                     
C            FO    = Oil friction factor, (-)                             
C            FW    = Water friction factor, (-)                           
C            G     = Gravitational acceleration, (m/s^2)                  
C            HLD1,HLD(l)=Equilibrium layer height variables, (m)          
C            IERR  = Error code (0=OK, 1=input variables out of range,    
C                                      2=Extrapolation occurring)         
C            N     = Loop index                                           
C            NMAX  = Maximum number of iterations                         
C            REO   = Reynolds number of oil flow, (-)                     
C            REW   = Reynolds number of water flow, (-)                   
C            SI    = Interface length, (m)                                
C            SIGN  = Sign of F function                                   
C            SO    = Oil perimeter, (m)                                   
C            SW    = Water perimeter, (m)                                 
C            TAUI  = Interfacial shear stress, (pa)                       
C            TAUO  = Oil shear stress, (pa)                               
C            TAUW  = Water shear stress, (pa)                             
C            VO    = Actual oil velocity, (m/s)                           
C            *VOS  = Superficial oil velocity, (m/s)                      
C            VW    = Actual water velocity, (m/s)                         
C            *VWS  = Superficial water velocity, (m/s)                    
C            F1,X,XMAX= Dummy parameters                                  
C   (*Indicates input variables)                                          
C                                                                         
C   ********************************************************************  
C Oscar's modification                                                                         
      SUBROUTINE HDF(EPW,VWS,VOS,ANG,DENW,DENO,HLD1,HLD,IOERR,IERR,KK) 
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)                             
      DATA G,EPS/9.81,0.000001 /                                       
C     -------------------------------------                            
C     Check input variables for valid range.                           
C     -------------------------------------                            
      IF (VWS .LT. 0) THEN                                             
          WRITE(IOERR,*)'HDF : Illegal value input for VWS'            
          IERR=1                                                       
      ELSE IF (VOS .LT. 0.0)THEN                                       
          WRITE(IOERR,*)'HDF : Illegal value input for VOS'            
          IERR=1                                                       
      ENDIF                                                            
      IF (IERR.EQ.1) GOTO 999                                           
                                                                       
      NMAX = 100                                                       
      N    = 0                                                         
      X    = HLD1 + 10.*EPS                                            
      DX   = (1. - X - EPS)/50.                                        
      XMAX = 1.                                                        
      
      GOTO 20                                                          
 10   F1   = F                                                         
      X    = X + DX                                                    
      IF (X .GE. XMAX)  GOTO 50                                        
 20   HLD = X                                                          
C    -------------------------------------------------------------------  
C    Calculate the geometrical variables for a given liquid level, Hl/DI  
C    -------------------------------------------------------------------  
      CALL EQU (VWS, VOS, HLD, AA, AW, A, AO, VW, VO, SW, SO, SI,     
     +  DW, DOIL, REW, REO, FW, FI, FO, TAUW, TAUO, TAUI, IOERR, IERR)
      IF(IERR .EQ. 1) THEN                                            
        WRITE(IOERR,*)'Error returned from EQU call'                  
        GOTO 999                                                      
      ENDIF                                                           
                                                                      
      F = -TAUW*SW/AW + TAUO*SO/AO + TAUI*(SI/AW+SI/AO)               
     +                             -(DENW-DENO)*G*SIN(ANG)            
     
      A11 = -TAUW*SW/AW
      A12 = TAUO*SO/AO
      A13 = TAUI*(SI/AW+SI/AO)
      A14 = -(DENW-DENO)*G*SIN(ANG) 

      IF (DABS(DX) .LT. EPS)  GOTO 40                                 
      N = N + 1                                                       
      IF (N .EQ. 1) THEN                                              
          GOTO 10                                                     
      ELSE IF  (N .GT. NMAX)  THEN                                    
          WRITE(IOERR,*)'Subroutine HDF did not converge'             
          IERR=1                                                      
          GOTO 999                                                    
      ENDIF

      SIGN = F*F1
      PP = F*F1
      IF (SIGN .EQ. 0.0) THEN                                         
        GOTO 40                                                       
      ELSE IF (SIGN .LT. 0.0)   THEN                                  
           DX = -DX/2.                                                
           GOTO 10                                                    
      ELSE IF (SIGN .GT. 0.0)   THEN                                  
           GOTO 10                                                    
      ENDIF                                                           
                                                                      
 40   CONTINUE   
 50   CONTINUE                                                        
                                                                      
999   RETURN                                                          
      END                                                             
C    **************************************************************** 
C            DETERMINATION OF THE GEOMETRICAL PARAMETERS              
C    WRITTEN BY: Jose Luis Trallero                                   
C    REVISED BY: Avni Serdar Kaya     LAST REVISION :JUNE 12, 1997    
C               *  * TULSA UNIVERSITY FLUID FLOW PROJECTS   *  *      
C    **************************************************************** 
C    This subroutine determines geometrical parameters and shear      
C    stresses. The SI unit system is used.                            
C                                                                     
C                            REFERENCES                               
C                            ----------                               
C    1. Trallero, J.L.:"Oil-Water Flow Patterns in Horizontal Pipes"  
C       Ph.D. Dissertation, The University of Tulsa, 1995.            
C                                                                     
C    **************************************************************** 
C                                                                     
C               INPUT/OUTPUT LOCAL FILE VARIABLES                     
C               ---------------------------------                     
C    *IOERR = Output file for error messages when input values passed 
C             to thesubroutine are out of range or error occurs in    
C             the calculation.                                        
C                                                                     
C                        SUBPROGRAM CALLED                            
C                        -----------------                            
C     FFF   = Calculates wall friction factor.                        
C                                                                     
C                       VARIABLE DESCRIPTION                          
C                       --------------------                          
C            A       = Total flow area, (m^2)                         
C            AA      = Pseudo oil level, (m)                          
C            AO      = Oil flow area,  (m^2)                          
C            AW      = Water flow area, (m^2)                         
C            ANG     = Inclination angle, (rad.)                      
C            DI      = Pipe diameter,  (m)                            
C            DENI    = Interfacial density, (kg/m)                    
C            DENO    = Oil density, (kg/m^3)                          
C            DENW    = Water density, (kg/m^3)                        
C            DX      = Equilibrium liquid level increment             
C            DOIL    = Oil hydraulic diameter, (m)                    
C            DW      = Water hydraulic diameter, (m)                  
C            F       = RHS of the combined momentum equation, (pa/m)  
C            FI      = Interfacial friction factor, (-)               
C            FO      = Oil friction factor, (-)                       
C            FW      = Water friction factor, (-)                     
C            G       = Gravitational acceleration, (m/s^2)            
C           *HLD(l)  = Equilibrium layer height variables, (m)            
C            IERR    = Error code 0=OK, 1=input variables out of range,  
C                                        2=Extrapolation occurring        
C            REI     = Interfacial Reynolds number, (-)                   
C            REO     = Reynolds number of oil flow, (-)                   
C            REW     = Reynolds number of water flow, (-)                 
C            SI      = Interface length, (m)                              
C            SO      = Oil perimeter, (m)                                 
C            SW      = Water perimeter, (m)                               
C            TAUI    = Interfacial shear stress, (pa)                     
C            TAUO    = Oil shear stress, (pa)                             
C            TAUW    = Water shear stress, (pa)                           
C            VISO    = Oil viscosity,    (cp)                             
C            VISW    = Water viscosity,  (cp)                             
C            VO      = Actual oil velocity, (m/s)                         
C           *VOS     = Superficial oil velocity, (m/s)                    
C            VW      = Actual water velocity, (m/s)                       
C           *VWS     = Superficial water velocity, (m/s)                  
C            F1,X,XMAX= Dummy parameter                                   
C   (*Indicates input variables)                                          
C  ********************************************************************
      SUBROUTINE EQU (VWS, VOS, HLD, AA, AW, A, AO, VW, VO, SW, SO, SI,
     + DW, DOIL, REW, REO, FW, FI, FO, TAUW, TAUO, TAUI, IOERR, IERR)  
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
C Oscar's modification                                
      COMMON /A/ DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW
C    ----------------------------                                      
C    Validity check of parameters                                      
C    ----------------------------                                      
      IF (VWS .LT. 0)THEN                                              
          WRITE(IOERR,*)'EQU : Illegal value input for VWS'            
          IERR=1                                                       
      ELSE IF (VOS .LT. 0)THEN                                         
          WRITE(IOERR,*)'EQU : Illegal value input for VOS'            
          IERR=1                                                       
      ELSE IF (HLD .LT. 0)THEN                                         
          WRITE(IOERR,*)'EQU : Illegal value input for HLD'            
          IERR=1                                                       
      ENDIF                                                            
      IF(IERR.EQ.1) GOTO 999                                           
      PI = 3.141592654                                                 
                                                                       
      AA = 2.*HLD - 1.                                                 
      AW = 0.25*DI**2 * (PI-DACOS(AA) + AA*DSQRT(1.-AA**2))            
      A  = (PI/4) * DI**2                                              
      AO = A - AW                                                      
      VW = VWS*A/AW                                                    
      VO = VOS*A/AO                                                    
      SW = DI*(PI - DACOS(AA))                                         
      SO = PI*DI - SW                                                  
      SI = DI*DSQRT(1.-AA**2)                                          
C     ----------------------------------------------------------       
C     Define the hydraulic diameters according to faster phase         
C     faster phase: duct flow  & slower phase: open channel flow       
C     ----------------------------------------------------------       
      IF (VO .GE. VW)  THEN                                            
         DW   = 4.*AW/SW                                               
         DOIL = 4.*AO/(SI+SO)                                          
      ELSE                                                             
         DW   = 4.*AW/(SI+SW)                                          
         DOIL = 4.*AO/SO                                               
      ENDIF                                                            
                                                                       
      REW  = DABS(VW)*DW*DENW/VISW                                     
      REO  = DABS(VO)*DOIL*DENO/VISO                                   
C    -------------------------------------------------------           
C    Faster phase determines the interfacial friction factor           
C    -------------------------------------------------------           
      IF (VO .GE. VW)  THEN                                            
         REI = REO                                                     
         ROI = DENO                                                    
      ELSE                                                             
         REI = REW                                                     
         ROI = DENW                                                    
      ENDIF                                                            
C    -------------------------------------------------------
C    Determination of the shear stress - O.Rodriguez 12/2001
C    -------------------------------------------------------  
      FW = FFF(EPW,REW,DW,IOERR,IERR)                         
      IF(IERR .EQ. 1) THEN                                             
        WRITE(IOERR,*)'Error returned from FFF call'                   
        GOTO 999                                                       
      ENDIF                                                            
      
      FO = FFF(EPW,REO,DOIL,IOERR,IERR)                                
      IF(IERR .EQ. 1) THEN                                             
        WRITE(IOERR,*)'Error returned from FFF call'                   
        GOTO 999                                                       
      ENDIF                                                            
      
      IF (VO .GE. VW)  THEN
         FI = FO
      ELSE
         FI = FW
      ENDIF

      TAUW = 0.5*FW*DENW*VW**2
      TAUO = 0.5*FO*DENO*VO**2
      TAUI = 0.5*FI*ROI*(VO-VW)*DABS(VO-VW)
999   RETURN
      END
C    ****************************************************************  
C                CALCULATION OF WALL FRICTION FACTOR 
C                (INCLUDING THE PIPE WALL ROUGHNESS)                   
C    WRITTEN BY: Jose Luis Trallero                                    
C    REVISED BY: Oscar M.H. Rodriguez     LAST REVISION :December, 2001
C               *  * Kramers Laboratorium   *  *          
C    ****************************************************************  
C    This function calculates wall friction factor. The SI unit system 
C    is used.                                                          
C                                                                      
C                            REFERENCES                                
C                            ----------                                
C    1. Trallero, J.L.:"Oil-Water Flow Patterns in Horizontal Pipes"   
C       Ph.D. Dissertation,The University of Tulsa, 1995.              
C                                                                      
C    ****************************************************************  
C                                                                      
C                INPUT/OUTPUT LOCAL FILE VARIABLES                     
C                ---------------------------------                     
C    *IOERR = Output file for error messages when input values passed  
C             to the subroutine are out of range or error occurs in    
C             the calculation.                                         
C                                                                      
C                        VARIABLE DESCRIPTION                          
C                        --------------------                          
C            C       = Coefficient of friction factor correlation      
C            EN      = Exponent of friction factor correlation         
C            FFF     = Wall friction factor, (-)                       
C            IERR    = Error code (0=OK, 1=input variables out of range,  
C                                        2=Extrapolation occurring)       
C            *RE     = Reynolds number, (-)                               
C   (*Indicates input variables)                                          
C                                                                         
C********************************************************************     
      FUNCTION FFF(EPF, RE, DHYD, IOERR, IERR)
      IMPLICIT DOUBLE PRECISION  (A-H, O-Z)
C    ----------------------------
C    Validity check of parameters
C    ----------------------------
      IF (RE .LT. 0) THEN
          WRITE(IOERR,*)'FFF : Illegal value input for RE'
          IERR=1
      ENDIF
      IF(IERR.EQ.1) GOTO 999

      IF (RE .LE. 1502.) THEN
        AT = 1.0d0
        BT = 0.0d0
      ELSE
         AT = 0.0d0
         BT = 1.0d0
      ENDIF

      FFF = AT*16*RE**(-1)+BT*(-3.6*LOG10( (6.9/RE) +
     & +(EPF/(3.7*DHYD))**1.1))**(-2.0) 

999   RETURN

      END
C    ****************************************************************     
C        CALCULATION OF THE RHS OF COMBINED MOMENTUM EQUATION             
C    WRITTEN BY: Jose Luis Trallero                                       
C    REVISED BY: Avni Serdar Kaya     LAST REVISION :JUNE 12, 1997        
C               *  * TULSA UNIVERSITY FLUID FLOW PROJECTS   *  *          
C    ****************************************************************     
C                                                                           
C    This subroutine calculates RHS of the combined momentum equation.    
C    The SI unit system is used.                                          
C                                                                           
C                            REFERENCES                                   
C                            ----------                                   
C    1. Trallero, J.L.:"Oil-Water Flow Patterns in Horizontal Pipes"      
C       Ph.D. Dissertation, The University of Tulsa, 1995.                
C                                                                         
C    ****************************************************************     
C                                                                         
C                    INPUT/OUTPUT LOCAL FILE VARIABLES                    
C                    ---------------------------------                    
C    *IOERR = Output file for error messages when input values passed     
C             to the subroutine are out of range or error occurs in       
C             the calculation.                                            
C                                                                         
C                        VARIABLE DESCRIPTION                             
C                        --------------------                             
C            FF      = RHS of the combined momentum equation, (pa/m)      
C            G       = Gravitational acceleration, (m/s^2)                
C           *HLD     = Equilibrium layer height variables, (m)            
C            IERR    = Error code (0=OK, 1=input variables out of range,  
C                                        2=Extrapolation occurring)       
C           *VOS     = Superficial oil velocity, (m/s)                    
C           *VWS     = Superficial water velocity, (m/s)                  
C   (*Indicates input variables)                                          
C********************************************************************
      FUNCTION FF(VWS,VOS,HLD , IOERR, IERR)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      COMMON /A/ DI, ANG, EPW, DENW, DENO, VISW, VISO, SUROW
C    ----------------------------
C    Validity check of parameters
C    ----------------------------
      IF (VWS .LT. 0) THEN
          WRITE(IOERR,*)'FF : Illegal value input for VWS'
          IERR=1
      ELSE IF (VOS .LT. 0) THEN
          WRITE(IOERR,*)'FF : Illegal value input for VOS'
          IERR=1
      ENDIF
        IF(IERR.EQ.1) GOTO 999
C    ------------------------------------------------------------------   
C    Calculate the geometrical variables for a given liquid level, hl/di  
C    ------------------------------------------------------------------   
      G = 9.81
      CALL EQU (VWS, VOS, HLD, AA, AW, A, AO, VW, VO, SW, SO, SI,
     & DW, DOIL, REW, REO, FW, FI, FO, TAUW, TAUO, TAUI , IOERR, IERR)
      IF(IERR .EQ. 1) THEN
          WRITE(IOERR,*)'Error returned from EQU call'
          GOTO 999
        ENDIF

      FF = -TAUW*SW/AW + TAUO*SO/AO + TAUI*(SI/AW+SI/AO)
     &                              - (DENW-DENO)*G*DSIN(ANG)

999   RETURN
      END
