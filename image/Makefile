CFLAGS= -O3 -Wextra 
NVCC=nvcc
NVCC_FLAGS= 
DEPS=


all: imageblur

imageblur: imageblur.cu util.h ppm.cu ppm.h
	$(NVCC) -o imageblur imageblur.cu ppm.cu $(NVCC_FLAGS) $(addprefix -Xcompiler ,$(CFLAGS))

clean:
	rm -rf imageblur output-mem.ppm output-tex1d.ppm output-tex2d.ppm

test: imageblur
	./imageblur 0 output-blur.ppm
	./imageblur 1 output-greyscale.ppm
	./imageblur 2 output-vignette.ppm
	./imageblur 3 output-sharpen.ppm
