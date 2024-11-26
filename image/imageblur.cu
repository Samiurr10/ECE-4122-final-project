// Include necessary headers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "util.h"
#include "ppm.h"

#define IMAGE_DIM 2048
#define SAMPLE_SIZE 10
#define SAMPLE_DIM (SAMPLE_SIZE * 2 + 1)
#define NUMBER_OF_SAMPLES (SAMPLE_DIM * SAMPLE_DIM)

/**************************************/
/* Grayscale Conversion Kernel (GPU)  */
/**************************************/
__global__ void image_grayscale(const uchar4 *image, uchar4 *image_output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= IMAGE_DIM || y >= IMAGE_DIM)
        return;

    int index = x + y * IMAGE_DIM;

    uchar4 pixel = image[index];

    // Gray = 0.299 * R + 0.587 * G + 0.114 * B
    unsigned char gray = (unsigned char)((77 * pixel.x + 150 * pixel.y + 29 * pixel.z) >> 8);

    image_output[index].x = gray;
    image_output[index].y = gray;
    image_output[index].z = gray;
    image_output[index].w = 255;
}

/**************************************/
/* Grayscale Conversion (CPU)         */
/**************************************/
void image_grayscale_cpu(const uchar4 *image, uchar4 *image_output)
{
    for (int y = 0; y < IMAGE_DIM; y++)
    {
        for (int x = 0; x < IMAGE_DIM; x++)
        {
            int index = x + y * IMAGE_DIM;

            uchar4 pixel = image[index];

            unsigned char gray = (unsigned char)((77 * pixel.x + 150 * pixel.y + 29 * pixel.z) >> 8);

            image_output[index].x = gray;
            image_output[index].y = gray;
            image_output[index].z = gray;
            image_output[index].w = 255;
        }
    }
}

/**************************************/
/* Blur Kernel (GPU)                  */
/**************************************/
__global__ void image_blur(uchar4 *image, uchar4 *image_output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    x = (x + IMAGE_DIM) % IMAGE_DIM;
    y = (y + IMAGE_DIM) % IMAGE_DIM;

    int index = x + y * IMAGE_DIM;

    int sum_r = 0;
    int sum_g = 0;
    int sum_b = 0;
    int count = 0;

    for (int dy = -SAMPLE_SIZE; dy <= SAMPLE_SIZE; dy++)
    {
        for (int dx = -SAMPLE_SIZE; dx <= SAMPLE_SIZE; dx++)
        {
            int neighbor_x = (x + dx + IMAGE_DIM) % IMAGE_DIM;
            int neighbor_y = (y + dy + IMAGE_DIM) % IMAGE_DIM;
            int neighbor_index = neighbor_x + neighbor_y * IMAGE_DIM;

            uchar4 pixel = image[neighbor_index];

            sum_r += pixel.x;
            sum_g += pixel.y;
            sum_b += pixel.z;
            count++;
        }
    }

    unsigned char avg_r = sum_r / count;
    unsigned char avg_g = sum_g / count;
    unsigned char avg_b = sum_b / count;

    image_output[index].x = avg_r;
    image_output[index].y = avg_g;
    image_output[index].z = avg_b;
    image_output[index].w = 255;
}

/**************************************/
/* Blur Function (CPU)                */
/**************************************/
void image_blur_cpu(const uchar4 *image, uchar4 *image_output)
{
    for (int y = 0; y < IMAGE_DIM; y++)
    {
        for (int x = 0; x < IMAGE_DIM; x++)
        {
            int sum_r = 0;
            int sum_g = 0;
            int sum_b = 0;
            int count = 0;

            for (int dy = -SAMPLE_SIZE; dy <= SAMPLE_SIZE; dy++)
            {
                for (int dx = -SAMPLE_SIZE; dx <= SAMPLE_SIZE; dx++)
                {
                    int neighbor_x = (x + dx + IMAGE_DIM) % IMAGE_DIM;
                    int neighbor_y = (y + dy + IMAGE_DIM) % IMAGE_DIM;
                    int neighbor_index = neighbor_x + neighbor_y * IMAGE_DIM;

                    uchar4 pixel = image[neighbor_index];

                    sum_r += pixel.x;
                    sum_g += pixel.y;
                    sum_b += pixel.z;
                    count++;
                }
            }

            unsigned char avg_r = sum_r / count;
            unsigned char avg_g = sum_g / count;
            unsigned char avg_b = sum_b / count;

            int index = x + y * IMAGE_DIM;

            image_output[index].x = avg_r;
            image_output[index].y = avg_g;
            image_output[index].z = avg_b;
            image_output[index].w = 255;
        }
    }
}

/**************************************/
/* Vignette Kernel (GPU)              */
/**************************************/
__global__ void image_vignette(const uchar4 *image, uchar4 *image_output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= IMAGE_DIM || y >= IMAGE_DIM)
        return;

    int index = x + y * IMAGE_DIM;

    float center_x = IMAGE_DIM / 2.0f;
    float center_y = IMAGE_DIM / 2.0f;

    float dx = x - center_x;
    float dy = y - center_y;

    float distance = sqrtf(dx * dx + dy * dy);

    float max_distance = sqrtf(center_x * center_x + center_y * center_y);

    float v = 2.0f; // Exponent to control the strength of the vignette
    float scaling = 1.0f - powf(distance / max_distance, v);

    if (scaling < 0.0f)
        scaling = 0.0f;

    uchar4 pixel = image[index];

    uchar4 output_pixel;
    output_pixel.x = (unsigned char)(pixel.x * scaling);
    output_pixel.y = (unsigned char)(pixel.y * scaling);
    output_pixel.z = (unsigned char)(pixel.z * scaling);
    output_pixel.w = 255;

    image_output[index] = output_pixel;
}

/**************************************/
/* Vignette Function (CPU)            */
/**************************************/
void image_vignette_cpu(const uchar4 *image, uchar4 *image_output)
{
    float center_x = IMAGE_DIM / 2.0f;
    float center_y = IMAGE_DIM / 2.0f;

    float max_distance = sqrtf(center_x * center_x + center_y * center_y);
    float v = 2.0f; // Exponent to control the strength of the vignette

    for (int y = 0; y < IMAGE_DIM; y++)
    {
        for (int x = 0; x < IMAGE_DIM; x++)
        {
            int index = x + y * IMAGE_DIM;

            float dx = x - center_x;
            float dy = y - center_y;

            float distance = sqrtf(dx * dx + dy * dy);

            float scaling = 1.0f - powf(distance / max_distance, v);

            if (scaling < 0.0f)
                scaling = 0.0f;

            uchar4 pixel = image[index];

            uchar4 output_pixel;
            output_pixel.x = (unsigned char)(pixel.x * scaling);
            output_pixel.y = (unsigned char)(pixel.y * scaling);
            output_pixel.z = (unsigned char)(pixel.z * scaling);
            output_pixel.w = 255;

            image_output[index] = output_pixel;
        }
    }
}

/**************************************/
/* Sharpen Kernel (GPU)               */
/**************************************/
__global__ void image_sharpen(const uchar4 *image, uchar4 *image_output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= IMAGE_DIM || y >= IMAGE_DIM)
        return;

    int index = x + y * IMAGE_DIM;

    int sum_r = 0;
    int sum_g = 0;
    int sum_b = 0;

    int kernel[3][3] = {
        { 0, -1,  0 },
        { -1, 5, -1 },
        { 0, -1,  0 }
    };

    for (int ky = -1; ky <= 1; ky++)
    {
        for (int kx = -1; kx <= 1; kx++)
        {
            int neighbor_x = x + kx;
            int neighbor_y = y + ky;

            neighbor_x = min(max(neighbor_x, 0), IMAGE_DIM - 1);
            neighbor_y = min(max(neighbor_y, 0), IMAGE_DIM - 1);

            int neighbor_index = neighbor_x + neighbor_y * IMAGE_DIM;

            uchar4 pixel = image[neighbor_index];

            int kernel_value = kernel[ky + 1][kx + 1];

            sum_r += pixel.x * kernel_value;
            sum_g += pixel.y * kernel_value;
            sum_b += pixel.z * kernel_value;
        }
    }

    sum_r = min(max(sum_r, 0), 255);
    sum_g = min(max(sum_g, 0), 255);
    sum_b = min(max(sum_b, 0), 255);

    image_output[index].x = (unsigned char)sum_r;
    image_output[index].y = (unsigned char)sum_g;
    image_output[index].z = (unsigned char)sum_b;
    image_output[index].w = 255;
}

/**************************************/
/* Sharpen Function (CPU)             */
/**************************************/
void image_sharpen_cpu(const uchar4 *image, uchar4 *image_output)
{
    int kernel[3][3] = {
        { 0, -1,  0 },
        { -1, 5, -1 },
        { 0, -1,  0 }
    };

    for (int y = 0; y < IMAGE_DIM; y++)
    {
        for (int x = 0; x < IMAGE_DIM; x++)
        {
            int sum_r = 0;
            int sum_g = 0;
            int sum_b = 0;

            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    int neighbor_x = x + kx;
                    int neighbor_y = y + ky;

                    neighbor_x = min(max(neighbor_x, 0), IMAGE_DIM - 1);
                    neighbor_y = min(max(neighbor_y, 0), IMAGE_DIM - 1);

                    int neighbor_index = neighbor_x + neighbor_y * IMAGE_DIM;

                    uchar4 pixel = image[neighbor_index];

                    int kernel_value = kernel[ky + 1][kx + 1];

                    sum_r += pixel.x * kernel_value;
                    sum_g += pixel.y * kernel_value;
                    sum_b += pixel.z * kernel_value;
                }
            }

            sum_r = min(max(sum_r, 0), 255);
            sum_g = min(max(sum_g, 0), 255);
            sum_b = min(max(sum_b, 0), 255);

            int index = x + y * IMAGE_DIM;

            image_output[index].x = (unsigned char)sum_r;
            image_output[index].y = (unsigned char)sum_g;
            image_output[index].z = (unsigned char)sum_b;
            image_output[index].w = 255;
        }
    }
}

int main(int argc, char **argv)
{
    unsigned int image_size;
    uchar4 *d_image, *d_image_output;
    uchar4 *h_image;
    uchar4 *h_image_cpu_output;
    cudaEvent_t start, stop;

    image_size = IMAGE_DIM * IMAGE_DIM * sizeof(uchar4);

    if (argc != 3)
    {
        printf("Syntax: %s mode outputfilename.ppm\n\twhere mode is 0 (blur), 1 (grayscale), 2 (vignette), or 3 (sharpen)\n", argv[0]);
        exit(1);
    }
    int mode = atoi(argv[1]);
    const char *filename = argv[2];

    // Create timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on the GPU for the input and output images
    CHECK_ERROR(cudaMalloc((void **)&d_image, image_size));
    CHECK_ERROR(cudaMalloc((void **)&d_image_output, image_size));

    // Allocate and load host image
    h_image = (uchar4 *)malloc(image_size);
    if (h_image == NULL)
    {
        printf("Malloc failed for h_image\n");
        exit(1);
    }
    h_image_cpu_output = (uchar4 *)malloc(image_size);
    if (h_image_cpu_output == NULL)
    {
        printf("Malloc failed for h_image_cpu_output\n");
        exit(1);
    }
    input_image_file("input.ppm", h_image, IMAGE_DIM);

    // Copy image to device memory
    CHECK_ERROR(cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice));

    // CUDA grid and block dimensions
    dim3 blocksPerGrid((IMAGE_DIM + 15) / 16, (IMAGE_DIM + 15) / 16);
    dim3 threadsPerBlock(16, 16);

    struct timeval start_cpu, end_cpu;
    double cpu_time;
    float gpu_time_ms;

    switch (mode)
    {
    /*************************/
    /* Blur using GPU memory */
    /*************************/
    case 0:
    {
        // CPU implementation
        gettimeofday(&start_cpu, NULL);
        image_blur_cpu(h_image, h_image_cpu_output);
        gettimeofday(&end_cpu, NULL);
        cpu_time = (end_cpu.tv_sec - start_cpu.tv_sec) * 1000.0 +
                   (end_cpu.tv_usec - start_cpu.tv_usec) / 1000.0;

        // GPU implementation
        cudaEventRecord(start, 0);
        image_blur<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_image_output);
        check_launch("kernel blur");
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time_ms, start, stop);

        printf("Blur filter applied.\n");
        printf("CPU time: %f ms\n", cpu_time);
        printf("GPU time: %f ms\n", gpu_time_ms);
    }
    break;

    /*************************/
    /* Grayscale Conversion  */
    /*************************/
    case 1:
    {
        // CPU implementation
        gettimeofday(&start_cpu, NULL);
        image_grayscale_cpu(h_image, h_image_cpu_output);
        gettimeofday(&end_cpu, NULL);
        cpu_time = (end_cpu.tv_sec - start_cpu.tv_sec) * 1000.0 +
                   (end_cpu.tv_usec - start_cpu.tv_usec) / 1000.0;

        // GPU implementation
        cudaEventRecord(start, 0);
        image_grayscale<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_image_output);
        check_launch("kernel grayscale");
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time_ms, start, stop);

        printf("Grayscale conversion applied.\n");
        printf("CPU time: %f ms\n", cpu_time);
        printf("GPU time: %f ms\n", gpu_time_ms);
    }
    break;

    /*************************/
    /* Vignette Filter       */
    /*************************/
    case 2:
    {
        // CPU implementation
        gettimeofday(&start_cpu, NULL);
        image_vignette_cpu(h_image, h_image_cpu_output);
        gettimeofday(&end_cpu, NULL);
        cpu_time = (end_cpu.tv_sec - start_cpu.tv_sec) * 1000.0 +
                   (end_cpu.tv_usec - start_cpu.tv_usec) / 1000.0;

        // GPU implementation
        cudaEventRecord(start, 0);
        image_vignette<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_image_output);
        check_launch("kernel vignette");
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time_ms, start, stop);

        printf("Vignette filter applied.\n");
        printf("CPU time: %f ms\n", cpu_time);
        printf("GPU time: %f ms\n", gpu_time_ms);
    }
    break;

    /*************************/
    /* Sharpen Filter        */
    /*************************/
    case 3:
    {
        // CPU implementation
        gettimeofday(&start_cpu, NULL);
        image_sharpen_cpu(h_image, h_image_cpu_output);
        gettimeofday(&end_cpu, NULL);
        cpu_time = (end_cpu.tv_sec - start_cpu.tv_sec) * 1000.0 +
                   (end_cpu.tv_usec - start_cpu.tv_usec) / 1000.0;

        // GPU implementation
        cudaEventRecord(start, 0);
        image_sharpen<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_image_output);
        check_launch("kernel sharpen");
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time_ms, start, stop);

        printf("Sharpen filter applied.\n");
        printf("CPU time: %f ms\n", cpu_time);
        printf("GPU time: %f ms\n", gpu_time_ms);
    }
    break;

    default:
        printf("Unknown mode %d\n", mode);
        exit(1);
        break;
    }

    // Copy the image back from the GPU for output to file
    CHECK_ERROR(cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost));

    // Output image
    output_image_file(filename, h_image, IMAGE_DIM);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_image);
    cudaFree(d_image_output);
    free(h_image);
    free(h_image_cpu_output);

    return 0;
}
