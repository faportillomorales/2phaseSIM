# makefile for gvts
OBJ1 = OWHFP.o 

main : $(OBJ1)
	gfortran -pg $(OBJ1) -o prg

clean: 
	rm *.o *.out prg DOWW.DAT DWODOW.DAT ERROR.DAT ST.DAT WO.DAT OW.DAT STMI.DAT OILWAT.OUT

run: 
	./prg 

.f.o:	
	gfortran -O0 -g -Wall -fbounds-check -c $<

#.f.o:	
#	gfortran -pg -c -O3 -funroll-all-loops $<
