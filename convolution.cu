#include <iostream>
#include <cstdlib>
#include "network.h"

int main(int argc, char *argv[]) {

    size_t n = atoi(argv[1]);
    int mode = atoi(argv[2]);

    // setting
    int input_length = 4;
    int input_channel = 1;
    int kernel_length = 2;
    int output_channel = 1;

    cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float ms;

    img *input, *output;
    ker *kernel;

    cudaMallocManaged((void **)&input, sizeof(img));
    cudaMallocManaged((void **)&kernel, sizeof(ker));
    cudaMallocManaged((void **)&output, sizeof(img));

    int padding = 0;
    int stride = 1;

    input->length = input_length;
    input->channel = input_channel;

    kernel->length = kernel_length;
    kernel->bias = 0;

    output->length = ((input->length + 2 * padding) - kernel->length)/stride + 1;
    output->channel = output_channel;

    kernel->channel_out = output->channel;
    kernel->channel_in = input->channel;

    cudaMallocManaged((void **)&input->content,
            n * input->length * input->length * input->channel * sizeof(float));
    cudaMallocManaged((void **)&output->content,
            n * output->length * output->length * output->channel *sizeof(float));

    cudaMallocManaged((void **)&kernel->content,
            kernel->length * kernel->length * kernel->channel_out * kernel->channel_in * sizeof(float));
    cudaMallocManaged((void **)&kernel->bias, kernel->channel_out*sizeof(float));

    // modify input image
    for (int i = 0; i<n * input->length * input->length * input->channel; i++) {
        *(input->content + i) = i;
    }

    // modify kernel
    for (int i = 0; i<kernel->length*kernel->length*kernel->channel_in*kernel->channel_out; i++) {
        *(kernel->content + i) = i;
    }

    for (int i = 0; i<kernel->channel_out; i++) {
        *(kernel->bias + i) = 0;
    }

    cudaEventRecord(start);
    if (mode == 0) {
        convolution(input, output, kernel, padding, stride, n);
    } else {
        accelerated_convolution(input, output, kernel, padding, stride, n);
    }
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);

    printf("%.2f\n", ms);

    cudaFree(input->content);
    cudaFree(kernel->content);
    cudaFree(kernel->bias);
    cudaFree(output->content);

	cudaFree(input);
	cudaFree(kernel);
	cudaFree(output);
}