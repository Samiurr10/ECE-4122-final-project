// Include necessary headers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <filesystem>
#include <map>


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

struct ButtonInfo {
    sf::RectangleShape shape;
    sf::Text text;
};
//nvcc -ccbin /usr/bin/g++-13 imageblur.cu ppm.cu -lsfml-graphics -lsfml-window -lsfml-system -lsfml-system -o imageblur_gui -Xlinker --no-as-needed

int main(int argc, char* argv[]) {
    if (argc != 3 || strcmp(argv[1], "-T") != 0) {
        printf("Usage: %s -T input.ppm\n", argv[0]);
        return 1;
    }
    const char* input_filename = argv[2];

    sf::RenderWindow window(sf::VideoMode(1024, 600), "Image Processor");
    
    // Setup input/output image display
    sf::Texture inputTexture, outputTexture;
    sf::Sprite inputSprite, outputSprite;
    inputSprite.setPosition(200, 10);  // Moved right to make room for buttons
    outputSprite.setPosition(600, 10);

    // CUDA setup
    unsigned int image_size = IMAGE_DIM * IMAGE_DIM * sizeof(uchar4);
    uchar4 *d_image, *d_image_output;
    uchar4 *h_image;
    uchar4 *h_image_cpu_output;
    struct timeval start_cpu, end_cpu;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMalloc((void **)&d_image, image_size);
    cudaMalloc((void **)&d_image_output, image_size);
    h_image = (uchar4 *)malloc(image_size);
    h_image_cpu_output = (uchar4 *)malloc(image_size);
    double cpu_time;
    float gpu_time_ms;

    // Load font
    sf::Font font;
    if (!font.loadFromFile("/usr/share/fonts/TTF/DejaVuSans.ttf")) {
        printf("Error loading font\n");
        return -1;
    }

    // Setup buttons with text
    std::map<int, ButtonInfo> buttons;
    std::map<int, std::string> buttonLabels = {
        {0, "Load Image"},
        {1, "Blur"},
        {2, "Grayscale"},
        {3, "Vignette"},
        {4, "Sharpen"}
    };

    // Create buttons with their text
    for(auto& [id, label] : buttonLabels) {
        ButtonInfo info;
        info.shape.setSize(sf::Vector2f(150, 40));
        info.shape.setPosition(20, 20 + id * 60);
        info.shape.setFillColor(sf::Color(100, 100, 100));

        info.text.setFont(font);
        info.text.setString(label);
        info.text.setCharacterSize(20);
        info.text.setFillColor(sf::Color::White);
        
        // Center text in button
        sf::FloatRect textBounds = info.text.getLocalBounds();
        info.text.setPosition(
            20 + (150 - textBounds.width) / 2,
            20 + id * 60 + (40 - textBounds.height) / 2
        );

        buttons[id] = info;
    }

    // Main loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }

            if (event.type == sf::Event::MouseButtonPressed) {
                for(auto& [id, info] : buttons) {
                    if (info.shape.getGlobalBounds().contains(event.mouseButton.x, event.mouseButton.y)) {
                        if (id == 0) { 
                            if (inputTexture.loadFromFile(input_filename)) {
                                inputSprite.setTexture(inputTexture, true);
                                float scaleX = 380.0f / inputTexture.getSize().x;
                                float scaleY = 380.0f / inputTexture.getSize().y;
                                float scale = std::min(scaleX, scaleY);
                                inputSprite.setScale(scale, scale);
                            }
                        } else {
                            // Process image
                            input_image_file(input_filename, h_image, IMAGE_DIM);
                            cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
                            dim3 blocksPerGrid((IMAGE_DIM + 15) / 16, (IMAGE_DIM + 15) / 16);
                            dim3 threadsPerBlock(16, 16);
                            switch(id - 1) { 
                                case 0: // Blur
                                    gettimeofday(&start_cpu, NULL);
                                    image_blur_cpu(h_image, h_image_cpu_output);
                                    gettimeofday(&end_cpu, NULL);
                                    cpu_time = (end_cpu.tv_sec - start_cpu.tv_sec) * 1000.0 + (end_cpu.tv_usec - start_cpu.tv_usec) / 1000.0;

                                    cudaEventRecord(start, 0);
                                    image_blur<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_image_output);
                                    cudaEventRecord(stop, 0);
                                    cudaEventSynchronize(stop);
                                    cudaEventElapsedTime(&gpu_time_ms, start, stop);

                                    printf("CPU time: %f ms\n",cpu_time);
                                    printf("GPU time %f ms\n", gpu_time_ms);
                                    break;
                                case 1: // Grayscale

                                    gettimeofday(&start_cpu, NULL);
                                    image_blur_cpu(h_image, h_image_cpu_output);
                                    gettimeofday(&end_cpu, NULL);
                                    cpu_time = (end_cpu.tv_sec - start_cpu.tv_sec) * 1000.0 + (end_cpu.tv_usec - start_cpu.tv_usec) / 1000.0;

                                    cudaEventRecord(start, 0);
                                    image_grayscale<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_image_output);
                                    cudaEventRecord(stop, 0);
                                    cudaEventSynchronize(stop);
                                    cudaEventElapsedTime(&gpu_time_ms, start, stop);
                                    printf("CPU time: %f ms\n",cpu_time);
                                    printf("GPU time %f ms\n", gpu_time_ms);
                                    break;
                                case 2: // Vignette

                                    gettimeofday(&start_cpu, NULL);
                                    image_blur_cpu(h_image, h_image_cpu_output);
                                    gettimeofday(&end_cpu, NULL);
                                    cpu_time = (end_cpu.tv_sec - start_cpu.tv_sec) * 1000.0 + (end_cpu.tv_usec - start_cpu.tv_usec) / 1000.0;

                                    cudaEventRecord(start, 0);
                                    image_vignette<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_image_output);
                                    cudaEventRecord(stop, 0);
                                    cudaEventSynchronize(stop);
                                    cudaEventElapsedTime(&gpu_time_ms, start, stop);
                                    printf("CPU time: %f ms\n",cpu_time);
                                    printf("GPU time %f ms\n", gpu_time_ms);
                                    break;
                                case 3: // Sharpen

                                    gettimeofday(&start_cpu, NULL);
                                    image_blur_cpu(h_image, h_image_cpu_output);
                                    gettimeofday(&end_cpu, NULL);
                                    cpu_time = (end_cpu.tv_sec - start_cpu.tv_sec) * 1000.0 + (end_cpu.tv_usec - start_cpu.tv_usec) / 1000.0;

                                    cudaEventRecord(start, 0);
                                    image_sharpen<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_image_output);
                                    cudaEventRecord(stop, 0);
                                    cudaEventSynchronize(stop);
                                    cudaEventElapsedTime(&gpu_time_ms, start, stop);
                                    printf("CPU time: %f ms\n",cpu_time);
                                    printf("GPU time %f ms\n", gpu_time_ms);
                                    break;
                            }

                            cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost);
                            output_image_file("output.ppm", h_image, IMAGE_DIM);
                            outputTexture.loadFromFile("output.ppm");
                            outputSprite.setTexture(outputTexture, true);
                            // Scale output image
                            float scaleX = 380.0f / outputTexture.getSize().x;
                            float scaleY = 380.0f / outputTexture.getSize().y;
                            float scale = std::min(scaleX, scaleY);
                            outputSprite.setScale(scale, scale);
                        }
                    }
                }
            }
        }

        // Render
        window.clear(sf::Color(50, 50, 50));
        
        // Draw buttons and their text
        for(auto& [id, info] : buttons) {
            window.draw(info.shape);
            window.draw(info.text);
        }

        // Draw images if they exist
        if (inputTexture.getSize().x > 0) {
            window.draw(inputSprite);
        }
        if (outputTexture.getSize().x > 0) {
            window.draw(outputSprite);
        }

        window.display();
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_image);
    cudaFree(d_image_output);
    free(h_image);
    free(h_image_cpu_output);

    return 0;
}