# Image Processing Filters with CUDA

This program allows you to apply various image processing filters to an input image using both CPU and GPU implementations. The available filters are:

- **Blur:** Softens the image by reducing sharp edges.
- **Grayscale:** Converts the image to shades of gray.
- **Vignette:** Applies a vignette effect by darkening the edges of the image.
- **Sharpen:** Enhances edges to make the image clearer.

---

## Compiling the Program

To compile the program, follow these steps:

1. Open your terminal.
2. Navigate to the directory containing the source code.
3. Run the following command:

```bash
nvcc imageblur.cu ppm.cpp -o imageblur
```

## Running the Program

After compiling the program, you can apply different filters to your image by running specific commands.

### 1. **Applying the Blur Filter**
To apply the blur filter and save the result as `output_blur.ppm`, use the following command:

```bash
./imageblur 0 output_blur.ppm
```

### 2. **Applying the Grayscale Filter**
To convert the image to grayscale and save it as `output_grayscale.ppm`, use this command:
```bash
./imageblur 1 output_greyscale.ppm
```
### 3. **Applying the Vignette Filter**
To apply the vignette effect and save the result as `output_vignette.ppm`, run the following command:

```bash
./imageblur 2 output_vignette.ppm
```

### 4. **Applying the Sharpen Filter**
To apply the sharpen filter and save the result as `output_sharpen.ppm`, use this command:

```bash
./imageblur 3 output_sharpen.ppm
```


