BIN_DIR := $(abspath ../../test/bin)

ARCH=\
  -gencode arch=compute_86,code=sm_86

OPTIONS=-O3 -use_fast_math -lcurand -Xcompiler -Wall,-fPIC -lcublas -lcufft -prec-div=false -ftz=true -lineinfo # -rdc=true -Xcompiler -fopenmp --default-stream per-thread  #-maxrregcount 37

all: dct2_naive dct2_lee dct2_fft2 idct2_fft2 idct_idxst idxst_idct
lib: dct2_lib idct2_lib idct_idxst_lib idxst_idct_lib

dct2_naive: dct2_naive.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

dct2_naive_atom: dct2_naive_atom.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

dct2_lee: dct2_lee.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

dct2_fft2: dct2_fft2.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

idct2_fft2: idct2_fft2.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

idct_idxst: idct_idxst.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

idxst_idct: idxst_idct.cu
	mkdir -p $(BIN_DIR)
	nvcc -w $(ARCH) $(OPTIONS) -o $(BIN_DIR)/$@ $<

dct2_lib: dct2_fft2.cu
	nvcc -c $(ARCH) $(OPTIONS) -o $@.o $<
	ar rcs libdct2.a $@.o
	rm $@.o

idct2_lib: idct2_fft2.cu
	nvcc -c $(ARCH) $(OPTIONS) -o $@.o $<
	ar rcs libidct2.a $@.o
	rm $@.o

idct_idxst_lib: idct_idxst.cu
	nvcc -c $(ARCH) $(OPTIONS) -o $@.o $<
	ar rcs libidct_idxst.a $@.o
	rm $@.o

idxst_idct_lib: idxst_idct.cu
	nvcc -c $(ARCH) $(OPTIONS) -o $@.o $<
	ar rcs libidxst_idct.a $@.o
	rm $@.o

clean:
	rm -rf $(BIN_DIR)
	rm -f libdct2.a libidct2.a libidct_idxst.a libidxst_idct.a
