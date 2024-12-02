Image Processing Filters with CUDA

This program applies various image processing filters to an input image using both CPU and GPU implementations. The available filters are:

-Blur

-Grayscale

-Vignette

-Sharpen


Compiling the Program:
To compile the program, open your terminal, navigate to the directory containing the code, and run:


nvcc imageblur.cu ppm.cpp -o imageblur


Running the Program:


1.Applying the Blur Filter
To apply the blur filter and save the result as output_blur.ppm:

./imageblur 0 output_blur.ppm


2.Applying the Grayscale Conversion- 
To convert the image to grayscale and save it as output_grayscale.ppm:


./imageblur 1 output_grayscale.ppm


3.Applying the Vignette Filter- 
To apply the vignette filter and save the output as output_vignette.ppm:


./imageblur 2 output_vignette.ppm


4.Applying the Sharpen Filter-
To apply the sharpen filter and save the result as output_sharpen.ppm:


./imageblur 3 output_sharpen.ppm
