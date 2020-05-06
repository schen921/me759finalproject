#include <iostream>
#include <cstdlib>
#include <math.h>
#include "network.h"


// device function
// zero padding before convolution
__device__ void zero_padding(
        float *input_content,
        float* input_tmp,
        int input_channel,
        int input_length,
        int padding) {

    int a, b, c, index_in, index_out;

    int tx = threadIdx.x;
	int ty = threadIdx.y;

	int padding_length = input_length + 2 * padding;

    for (c = 0; c < input_channel; c++) {
	    for (b = 0; b < padding_length ; b += blockDim.y) {
	        for (a = 0; a < padding_length; a += blockDim.x) {

	            if (a + tx < padding_length && b + ty < padding_length) {

	                index_in = a + tx + (b + ty) * padding_length
	                        + c * padding_length * padding_length;

	                if (a + tx >= padding && a + tx <input_length + padding
	                    && b + ty >= padding && b + ty <input_length + padding) {

	                    index_out = a + tx -padding + (b + ty - padding) * input_length
	                            + c *input_length * input_length;
	                    *(input_tmp + index_in) = *(input_content + index_out);

	                } else {

	                    *(input_tmp + index_in) = 0;

	                }
	            }
	        }
	    }
	}
    __syncthreads();
}


// kernel function for accelerated convolution
__global__ void accelerated_convolution_kernel(
        img *input,
        img *output,
        ker *kernel,
        int padding,
        int stride) {

    int tx = threadIdx.x;
	int ty = threadIdx.y;

	int input_length = input ->length;
	int input_channel = input ->channel;
	float* input_content = &*(input ->content +
	        input_length * input_length * input_channel * blockIdx.x);

	int kernel_length = kernel->length;
	int kernel_channel_in = kernel->channel_in;
	int kernel_channel_out = kernel->channel_out;
	float* kernel_content = kernel->content;

	int output_length = output ->length;
	int output_channel = output ->channel;
	float* output_content = &*(output ->content +
	        output_length * output_length * output_channel *  blockIdx.x);

	int padding_length = input_length + 2 * padding;

	int index;
	int a, b, c_in, c_out, i;

	int input_arr_length = padding_length * padding_length * input_channel;
    int kernel_arr_length = kernel_length * kernel_length * kernel_channel_in * kernel_channel_out;
    int kernel_fold_length = kernel_arr_length/kernel_channel_out;
    int patches = output_length * output_length;

	extern __shared__ float tmp[];

	float *input_tmp = tmp;
	float *kernel_tmp = (float *) &*(tmp + input_arr_length);
	float *kernel_bias = (float *) &*(kernel_tmp + kernel_arr_length);
	float *input_unfold = (float *) &*(kernel_bias + kernel_channel_out);

	float sum;

	// zero padding
	zero_padding(input_content, input_tmp, input_channel, input_length, padding);

	// fill kernel in shared memory
	for (i = 0; i < kernel_arr_length; i += blockDim.x * blockDim.y) {

	    index = i + tx + ty * blockDim.x;

	    if (index < kernel_arr_length)
	        *(kernel_tmp + index) = *(kernel_content + index);
	}

	// file bias in shared memory
	if (tx + ty * blockDim.x < kernel_channel_out)
	    *(kernel_bias + tx + ty * blockDim.x) = *(kernel->bias + tx + ty * blockDim.x);

	// fill unfolded input in shared memory
	index = tx + ty * blockDim.x;
	i = 0;
	for (c_in = 0; c_in < kernel_channel_in; c_in++) {
	    for (b = 0; b < kernel_length; b++) {
	        for (a = 0; a < kernel_length; a++) {
	            *(input_unfold + i * patches + index)
	                = *(input_tmp + tx*stride + a + (ty*stride + b) * padding_length
	                    + c_in * padding_length * padding_length);
	            i++;
	        }
	    }
	}

	__syncthreads();

	// matrix multiplication
	for (c_out = 0; c_out < kernel_channel_out; c_out++) {

	    sum = 0;
	    for (i = 0; i < kernel_fold_length; i++) {
	        sum += *(kernel_tmp + i + c_out * kernel_fold_length)
	                * *(input_unfold + i * patches + index);
	    }

	    *(output_content + index
	        + c_out * blockDim.x * blockDim.x) = sum + *(kernel_bias + c_out);

	}
}


// kernel function for convolution
__global__ void convolution_kernel(
        img *input,
        img *output,
        ker *kernel,
        int padding,
        int stride) {

    int tx = threadIdx.x;
	int ty = threadIdx.y;

	int input_length = input ->length;
	int input_channel = input ->channel;
	float* input_content = &*(input ->content +
	        input_length * input_length * input_channel * blockIdx.x);

	int kernel_length = kernel->length;
	int kernel_channel_in = kernel->channel_in;
	int kernel_channel_out = kernel->channel_out;
	float* kernel_content = kernel->content;

	int output_length = output ->length;
	int output_channel = output ->channel;
	float* output_content = &*(output ->content +
	        output_length * output_length * output_channel * blockIdx.x);

	int padding_length = input_length + 2 * padding;

	int index;
	int a, b, c_in, c_out;

	int input_arr_length = padding_length * padding_length * input_channel;
    int kernel_arr_length = kernel_length * kernel_length * kernel_channel_in * kernel_channel_out;

	extern __shared__ float tmp[];

	float *input_tmp = tmp;
	float *kernel_tmp = (float *) &*(tmp + input_arr_length);
	float *kernel_bias = (float *) &*(kernel_tmp + kernel_arr_length);

	float sum;

	// zero padding
	zero_padding(input_content, input_tmp, input_channel, input_length, padding);

	// fill kernel in shared memory
	for (int i = 0; i < kernel_arr_length; i += blockDim.x * blockDim.y) {

	    index = i + tx + ty * blockDim.x;

	    if (index < kernel_arr_length)
	        *(kernel_tmp + index) = *(kernel_content + index);
	}

	// fill bias in shared memory
	if (tx + ty * blockDim.x < kernel_channel_out)
	    *(kernel_bias + tx + ty * blockDim.x) = *(kernel->bias + tx + ty * blockDim.x);

	__syncthreads();

    // convolution
	for (c_out = 0; c_out < kernel_channel_out; c_out++) {

	    sum = 0;
	    for (c_in = 0; c_in < kernel_channel_in; c_in++) {
	        for (b = 0; b < kernel_length; b++) {
	            for (a = 0; a < kernel_length; a++) {
	                sum += *(input_tmp + tx*stride + a + (ty*stride + b) * padding_length
	                            + c_in * padding_length * padding_length)
	                        * *(kernel_tmp + a + b * kernel_length
	                            + c_in * kernel_length * kernel_length
	                            + c_out * kernel_channel_in * kernel_length * kernel_length);
	            }
	        }
	    }
	    *(output_content + tx + ty*blockDim.x +
	        c_out * blockDim.x * blockDim.x) = sum + *(kernel_bias + c_out);
	}

}


// kernel function for compute the gradient of convolution kernel
__global__ void convolution_gradient_kernel(
        img *input,
        img *gradient,
        ker *kernel_gradient,
        int padding) {

    int tx = threadIdx.x;
	int ty = threadIdx.y;

	int input_length = input ->length;
	int input_channel = input ->channel;
	float* input_content = &*(input ->content +
	        input_length * input_length * input_channel * blockIdx.x);

	int kernel_length = kernel_gradient->length;
	int kernel_channel_in = kernel_gradient->channel_in;
	int kernel_channel_out = kernel_gradient->channel_out;
	float* kernel_gradient_content = &*(kernel_gradient->content
	        + kernel_length * kernel_length
	        * kernel_channel_in * kernel_channel_out * blockIdx.x);
	float* kernel_gradient_bias = &*(kernel_gradient->bias
	        + kernel_channel_out * blockIdx.x);

	int gradient_length = gradient->length;
	int gradient_channel = gradient ->channel;
	float* gradient_content = &*(gradient->content +
	        gradient_length * gradient_length * gradient_channel * blockIdx.x);

	int index;
	int a, b, c_in, c_out, i;

	int padding_length = input_length + 2 * padding;
	int input_arr_length = padding_length * padding_length * input_channel;
    int gradient_arr_length = gradient_length * gradient_length * gradient_channel;

	extern __shared__ float tmp[];

	float *input_tmp = tmp;
	float *gradient_tmp = (float *) &*(tmp + input_arr_length);

	float sum;

	// zero padding
	zero_padding(input_content, input_tmp, input_channel, input_length, padding);

	// fill gradient in shared memory
	for (i = 0; i < gradient_arr_length; i += blockDim.x * blockDim.y) {

	    index = i+ tx + ty * blockDim.x;

	    if (index < gradient_arr_length)
	        *(gradient_tmp + index) = *(gradient_content + index);
	}

	__syncthreads();

	// compute gradient of kernel
	for (c_out = 0; c_out < kernel_channel_out; c_out++) {
	    for (c_in = 0; c_in < kernel_channel_in; c_in++) {

	        sum = 0;
	        for (b = 0; b < gradient_length; b++) {
	            for (a = 0; a < gradient_length; a++) {
	                sum += *(input_tmp + tx + a + (ty + b) * padding_length
	                            + c_in * padding_length * padding_length)
	                        * *(gradient_tmp + a + b * gradient_length
	                            + c_out * gradient_length * gradient_length);
	            }
	        }

	        *(kernel_gradient_content + tx + ty*blockDim.x
	            + c_in * blockDim.x * blockDim.x
	            + c_out * kernel_channel_in * blockDim.x * blockDim.x)
	            = sum;
	    }
	}

	// compute gradient of bias
	index = tx + ty * blockDim.x;
	for (c_out = 0; c_out < kernel_channel_out; c_out += blockDim.x * blockDim.y) {

	    if (c_out + index < kernel_channel_out) {
	        sum = 0;
	        for (b = 0; b < gradient_length; b++) {
                for (a = 0; a < gradient_length; a++) {
                    sum += *(gradient_tmp + a + b * gradient_length
                              + (c_out + index) * gradient_length * gradient_length);
                }
            }

	        *(kernel_gradient_bias + c_out +index) = sum;
	    }
	}
}


// kernel function for compute the gradient of convolution input
__global__ void convolution_gradient_input(
        ker *kernel,
        img *gradient,
        img *input_gradient,
        int padding) {

    int tx = threadIdx.x;
	int ty = threadIdx.y;

	int gradient_length = gradient->length;
	int gradient_channel = gradient ->channel;
	float* gradient_content = &*(gradient->content +
	        gradient_length * gradient_length * gradient_channel * blockIdx.x);

	int kernel_length = kernel->length;
	int kernel_channel_in = kernel->channel_in;
	int kernel_channel_out = kernel->channel_out;
	float* kernel_content = kernel->content;

	int input_length = input_gradient->length;
	int input_channel = input_gradient ->channel;
	float* input_content = &*(input_gradient->content
	        + input_length * input_length
	        * input_channel * blockIdx.x);

	padding = kernel_length - padding - 1;

	int index;
	int a, b, c_in, c_out, i;

	int padding_length = gradient_length + 2 * padding;
    int gradient_arr_length = padding_length * padding_length * gradient_channel;
	int kernel_arr_length = kernel_length * kernel_length * kernel_channel_in * kernel_channel_out;

	extern __shared__ float tmp[];

	float *gradient_tmp = tmp;
	float *kernel_tmp = (float *) &*(tmp + gradient_arr_length);

	float sum;

	// zero padding
	zero_padding(gradient_content, gradient_tmp, gradient_channel, gradient_length, padding);

	// fill kernel in shared memory
	for (i = 0; i < kernel_arr_length; i += blockDim.x * blockDim.y) {

	    index = i + tx + ty * blockDim.x;

	    if (index < kernel_arr_length)
            *(kernel_tmp + index) = *(kernel_content + index);
	}

	__syncthreads();

	// compute gradient of kernel
	// kernel used in reversed order
	for (c_in = 0; c_in < kernel_channel_in; c_in++) {

	    sum = 0;
	    for (c_out = 0; c_out < kernel_channel_out; c_out++) {
	        for (b = 0; b < kernel_length; b++) {
	            for (a = 0; a < kernel_length; a++) {
	                sum += *(gradient_tmp + tx + a + (ty + b) * padding_length
	                            + c_out * padding_length * padding_length)
	                        * *(kernel_tmp
	                            + kernel_length - a - 1
	                            + (kernel_length - b - 1) * kernel_length
	                            + c_in * kernel_length * kernel_length
	                            + c_out * kernel_channel_in * kernel_length * kernel_length);
	            }
	        }
	    }

	    *(input_content + tx + ty*blockDim.x +
	        c_in * blockDim.x * blockDim.x) = sum;
	}
}


// kernel function for update the gradient of convolution kernel
// accross the batch
__global__ void convolution_gradient_update(
        ker *kernel,
        ker *kernel_gradient,
        float rate,
        int num) {

    int kernel_length = kernel->length;
	int kernel_channel_in = kernel->channel_in;
	int kernel_channel_out = kernel->channel_out;
	float* kernel_content = kernel->content;
	float *kernel_gradient_content = kernel_gradient->content;

	int kernel_arr_length = kernel_length * kernel_length
	        * kernel_channel_in * kernel_channel_out;

	float sum;

	for (int c_out = 0; c_out < kernel_channel_out; c_out++) {
	    sum = 0;
	    for (int i = 0; i < num; i++) {
	        sum += *(kernel_gradient_content + threadIdx.x
	                + c_out * kernel_arr_length / kernel_channel_out
	                + i * kernel_arr_length);
	    }

	    *(kernel_content + threadIdx.x
	                + c_out * kernel_arr_length / kernel_channel_out) -= rate * sum;
	}

	if (threadIdx.x < kernel_channel_out) {
	    sum = 0;
	    for (int i = 0; i < num; i++) {
	        sum += *(kernel_gradient->bias + threadIdx.x
	                + i * kernel_channel_out);
	    }

	    *(kernel->bias + threadIdx.x) -= rate * sum;
	}
}


// kernel function for max_pooling
__global__ void max_pooling_kernel(
        img *input,
        img *output,
        int *index,
        int stride) {

    int tx = threadIdx.x;
	int ty = threadIdx.y;

	int input_length = input ->length;
	int input_channel = input ->channel;
	float* input_content = &*(input ->content +
	        input_length * input_length * input_channel * blockIdx.x);

	int output_length = output ->length;
	int output_channel = output ->channel;
	float* output_content = &*(output ->content +
	        output_length * output_length * output_channel * blockIdx.x);

    float max, n;
    int max_index;

    for (int c = 0; c < input_channel; c++) {

        max = 0;
        max_index = 0;

        for (int b = 0; b < stride; b++) {
            for (int a = 0; a < stride; a++) {
                n = *(input_content + a + tx * stride
                        + (b + ty * stride) * input_length
                        + c * input_channel);
                if (n > max) {
                    max = n;
                    max_index = a + b * stride;
                }
            }
        }
        *(output_content + tx + ty * output_length + c * input_channel) = max;
        *(index + tx + ty * output_length + c * input_channel) = max_index;
    }

}


// kernel function for compute the gradient of max_pooling input
__global__ void max_pooling_gradient_kernel(
        img *input,
        img *output,
        int *index,
        int stride) {

    int tx = threadIdx.x;
	int ty = threadIdx.y;

	int input_length = input ->length;
	int input_channel = input ->channel;
	float* input_content = &*(input ->content +
	        input_length * input_length * input_channel * blockIdx.x);

	int output_length = output ->length;
	int output_channel = output ->channel;
	float* output_content = &*(output ->content +
	        output_length * output_length * output_channel * blockIdx.x);

    float max;
    int max_index;

    for (int c = 0; c < input_channel; c++) {

        max = *(output_content + tx + ty * output_length + c * input_channel);
        max_index = *(index + tx + ty * output_length + c * input_channel);

        for (int b = 0; b < stride; b++) {
            for (int a = 0; a < stride; a++) {
                if (a + b * stride == max_index) {
                    *(input_content + a + tx * stride + (b + ty * stride) * input_length) = max;
                } else {
                    *(input_content + a + tx * stride + (b + ty * stride) * input_length) = 0;
                }
            }
        }
    }
}


// kernel function for compute the output fully connected layer
// before activation function
__global__ void dense_kernel(
        int in_num,
        int out_num,
        float *input,
        float *weight,
        float *output,
        float bias) {

    int i;
    float sum = 0;

    extern __shared__ float input_tmp[];

    // fill input in shared memory
	for (i = 0; i < in_num; i += blockDim.x) {

        int index = i + threadIdx.x;

        if (index < in_num) {
            *(input_tmp + index) = *(input + index + in_num * blockIdx.x);
        }
    }

	__syncthreads();

	for (i = 0; i < in_num; i++) {
        sum += *(input_tmp + i) * *(weight + i * out_num + threadIdx.x);
    }

	*(output + threadIdx.x + out_num * blockIdx.x ) = sum + bias;
}


// kernel function for compute the gradient of fully connected layer weight
__global__ void dense_gradient_kernel(
        int in_num,
        int out_num,
        float *input,
        float *weight_gradient,
        float *bias_gradient,
        float *output_gradient,
        int num) {

    int i;

    extern __shared__ float tmp[];

    float *input_tmp = tmp;
    float *output_tmp = (float *) &*(tmp + in_num);

    float *weight_gradient_tmp = (float *) &*(weight_gradient
            + in_num * out_num * blockIdx.x);

    // fill input in shared memory
	for (i = 0; i < in_num; i += blockDim.x) {

        int index = i + threadIdx.x;

        if (index < in_num) {
            *(input_tmp + index) = *(input + index + in_num * blockIdx.x );
        }
    }

    // fill output gradient in shared memory
    *(output_tmp + threadIdx.x) = *(output_gradient + threadIdx.x
            + out_num * blockIdx.x);

	__syncthreads();

	// compute gradient of kernel
	for (i = 0; i < in_num; i++) {
        *(weight_gradient_tmp + i * out_num + threadIdx.x)
            = *(input_tmp + i) * *(output_tmp + threadIdx.x);
    }

	float sum = 0;
	for (i = 0; i < num; i++) {
	    sum += *(output_gradient + threadIdx.x + i * out_num);
	}
	*(bias_gradient + threadIdx.x) = sum;
}


// kernel function for compute the gradient of fully connected layer input
__global__ void dense_gradient_input(
        int in_num,
        int out_num,
        float *input_gradient,
        float *weight,
        float *output_gradient) {

    int i;
    float sum = 0;

    extern __shared__ float output_tmp[];

    // fill input in shared memory
	for (i = 0; i < out_num; i += blockDim.x) {

        int index = i + threadIdx.x;

        if (index < out_num) {
            *(output_tmp + index) = *(output_gradient + index + out_num * blockIdx.x);
        }
    }

	__syncthreads();

	for (i = 0; i < out_num; i++) {
        sum += *(output_tmp + i) * *(weight + i + threadIdx.x * out_num);
    }

	*(input_gradient + threadIdx.x + in_num * blockIdx.x) = sum;
}


// kernel function for update the gradient of fully connected layer weight
// accross the batch
__global__ void dense_gradient_update (
        int in_num,
        int out_num,
        float *output_gradient,
        float *weight,
        float *weight_gradient,
        float bias,
        float *bias_gradient,
        float rate,
        int num) {

    float sum;

    for (int i = 0; i < out_num; i++) {

        sum = 0;
        for (int j = 0; j < num; j++) {
            sum += *(weight_gradient + i + threadIdx.x * out_num
                    + j * out_num * in_num);
        }

        *(weight + i + threadIdx.x * out_num) -= rate * sum;
    }

    if (threadIdx.x == 0) {

        sum = 0;
        for (int i = 0; i < out_num; i++) {
            sum += *(bias_gradient + i);
        }

        bias -= rate * sum;
    }

}


// kernel function of the output of leaky ReLU
__global__ void leaky_relu_kernel(
        float *input,
        int size,
        int channel) {

	float* input_content = &*(input + size * blockIdx.x);

	for (int c = 0; c < channel; c++) {
	    if (*(input_content + threadIdx.x) < 0)
	        *(input_content + threadIdx.x) *= 0.01;
	}
}


// kernel function of the gradient of leaky ReLU
__global__ void leaky_relu_gradient_kernel(
        float *input,
        float *gradient,
        int size,
        int channel) {

	float* input_content = &*(input + size * blockIdx.x);
	float* gradient_content = &*(gradient + size * blockIdx.x);

	for (int c = 0; c < channel; c++) {
	    if (*(input_content + threadIdx.x) < 0)
	        *(gradient_content + threadIdx.x) *= 0.01;
	}
}


// kernel function of the output of softmax
__global__ void softmax_kernel(
        int in_num,
        float *input,
        float *output) {

    int i;

    extern __shared__ float tmp[];

    int max = 0;
    int sum = 0;

    for (i = 0; i < in_num; i++) {
        *(tmp + i + threadIdx.x * in_num) = *(input + i + threadIdx.x * in_num);
        if (*(tmp + i + threadIdx.x * in_num) > max)
            max = *(tmp + i + threadIdx.x * in_num);
    }

    for (i = 0; i < in_num; i++) {
        *(tmp + i + threadIdx.x * in_num) = exp(*(tmp + i + threadIdx.x * in_num)  - max);
    }

    for (i = 0; i < in_num; i++) {
        sum += *(tmp + i + threadIdx.x * in_num);
    }

    for (i = 0; i < in_num; i++) {
        *(output + i + threadIdx.x * in_num) = *(tmp + i + threadIdx.x * in_num) / sum;
    }
}


// kernel function of the gradient of softmax
__global__ void softmax_kernel_gradient(
        int in_num,
        float *output,
        float *output_gradient,
        float *label) {

    *(output_gradient + threadIdx.x + blockIdx.x * in_num) =
            *(output + threadIdx.x + blockIdx.x * in_num)
            - *(label+ threadIdx.x + blockIdx.x * in_num);

}


// host funtion
// compute corss entropy loss
__host__ float softmax_corss_entropy(
        int in_num,
        float *output,
        float *label,
        int num) {

    int i;
    float n;
    float sum = 0;

    for (i = 0; i < in_num * num; i++) {

        n = *(output + i);
        if (n == 0) n = 0.000000000000001;

        sum += - *(label + i) * log(n);
    }

    return sum;
}


// host function
// compute the count of which prerdiction label and actual label match
__host__ int predict(
        int in_num,
        float *output,
        float *label,
        int num) {

    int count = 0;

    int max_index;
    float max;

    for (int i = 0; i < num; i++) {

        max = 0;

        for (int j = 0; j < in_num; j++) {
            if (*(output + j + i * in_num) > max) {
                max = *(output + j + i * in_num);
                max_index = j;
            }
        }

        if (abs(*(label + max_index + i * in_num) - 1) < 0.00001)
            count ++;
    }

    return count;
}


// host function of forward training process
__host__ void forward(
        void *layer,
        int num) {

    inp *input_layer = (inp *) layer;
    conv *conv_layer_1 = (conv *) input_layer->next;
    maxpool *pool_layer_1 = (maxpool *) conv_layer_1->next;
    conv *conv_layer_2 = (conv *) pool_layer_1->next;
    maxpool *pool_layer_2 = (maxpool *) conv_layer_2->next;
    flat *flat_layer = (flat *) pool_layer_2->next;
    den *dense_layer_1 = (den *) flat_layer->next;
    den *dense_layer_2 = (den *) dense_layer_1->next;
    den *dense_layer_3 = (den *) dense_layer_2->next;
    outp *output_layer = (outp *) dense_layer_3->next;

    dim3 conv_1(conv_layer_1->output_gradient->length,
            conv_layer_1->output_gradient->length);

    dim3 conv_2(conv_layer_2->output_gradient->length,
            conv_layer_2->output_gradient->length);

    dim3 pool_1(pool_layer_1->output_gradient->length,
            pool_layer_1->output_gradient->length);

    dim3 pool_2(pool_layer_2->output_gradient->length,
            pool_layer_2->output_gradient->length);

    int num_thread_1 = pool_layer_1->output_gradient->length *
            pool_layer_1->output_gradient->length;

    int num_thread_2 = pool_layer_2->output_gradient->length *
            pool_layer_2->output_gradient->length;

    convolution_kernel<<<num, conv_1, conv_layer_1->forward_size * sizeof(float)>>> (
            conv_layer_1->input,
            pool_layer_1->input,
            conv_layer_1->kernel,
            conv_layer_1->padding,
            conv_layer_1->stride);
    cudaDeviceSynchronize();

    max_pooling_kernel<<<num, pool_1>>> (
            pool_layer_1->input,
            conv_layer_2->input,
            pool_layer_1->index,
            pool_layer_1->stride);
    cudaDeviceSynchronize();

    leaky_relu_kernel<<<num, num_thread_1>>>(
            conv_layer_2->input->content,
            conv_layer_2->input->length
            * conv_layer_2->input->length
            * conv_layer_2->input->channel,
            conv_layer_2->input->channel);
    cudaDeviceSynchronize();

    convolution_kernel<<<num, conv_2, conv_layer_2->forward_size * sizeof(float)>>> (
            conv_layer_2->input,
            pool_layer_2->input,
            conv_layer_2->kernel,
            conv_layer_2->padding,
            conv_layer_2->stride);
    cudaDeviceSynchronize();

    max_pooling_kernel<<<num, pool_2>>> (
            pool_layer_2->input,
            flat_layer->input,
            pool_layer_2->index,
            pool_layer_2->stride);
    cudaDeviceSynchronize();

    leaky_relu_kernel<<<num, num_thread_2>>>(
            flat_layer->input->content,
            flat_layer->input->length
            * flat_layer->input->length
            * flat_layer->input->channel,
            flat_layer->input->channel);
    cudaDeviceSynchronize();

    dense_kernel<<<num, dense_layer_1->out_num,
        dense_layer_1->in_num * sizeof(float) >>>(
                dense_layer_1->in_num,
                dense_layer_1->out_num,
                dense_layer_1->input,
                dense_layer_1->weight,
                dense_layer_2->input,
                dense_layer_1->bias);
    cudaDeviceSynchronize();

    leaky_relu_kernel<<<num, dense_layer_1->out_num>>>(
            dense_layer_2->input,
            dense_layer_1->out_num,
            1);
    cudaDeviceSynchronize();

    dense_kernel<<<num, dense_layer_2->out_num,
        dense_layer_2->in_num * sizeof(float) >>>(
                dense_layer_2->in_num,
                dense_layer_2->out_num,
                dense_layer_2->input,
                dense_layer_2->weight,
                dense_layer_3->input,
                dense_layer_2->bias);
    cudaDeviceSynchronize();

    leaky_relu_kernel<<<num, dense_layer_2->out_num>>>(
            dense_layer_3->input,
            dense_layer_2->out_num,
            1);
    cudaDeviceSynchronize();

    dense_kernel<<<num, dense_layer_3->out_num,
        dense_layer_3->in_num * sizeof(float) >>>(
                dense_layer_3->in_num,
                dense_layer_3->out_num,
                dense_layer_3->input,
                dense_layer_3->weight,
                output_layer->input,
                dense_layer_3->bias);
    cudaDeviceSynchronize();

    softmax_kernel<<<1, num, output_layer->in_num * num * sizeof(float)>>>(
            output_layer->in_num,
            output_layer->input,
            output_layer->input);
    cudaDeviceSynchronize();
}


// host function of backward training process
__host__ void backward(
        void *layer,
        float rate,
        int num) {

    outp *output_layer = (outp *) layer;
    den *dense_layer_3 = (den *) output_layer->previous;
    den *dense_layer_2 = (den *) dense_layer_3->previous;
    den *dense_layer_1 = (den *) dense_layer_2->previous;
    flat *flat_layer = (flat *) dense_layer_1->previous;
    maxpool *pool_layer_2 = (maxpool *) flat_layer->previous;
    conv *conv_layer_2 = (conv *) pool_layer_2->previous;
    maxpool *pool_layer_1 = (maxpool *) conv_layer_2->previous;
    conv *conv_layer_1 = (conv *) pool_layer_1->previous;

    dim3 conv_kernel_1(conv_layer_1->kernel->length,
            conv_layer_1->kernel->length);

    dim3 conv_input_1(conv_layer_1->input->length,
            conv_layer_1->input->length);

    dim3 conv_kernel_2(conv_layer_2->kernel->length,
            conv_layer_2->kernel->length);

    dim3 conv_input_2(conv_layer_2->input->length,
            conv_layer_2->input->length);

    dim3 pool_1(pool_layer_1->output_gradient->length,
            pool_layer_1->output_gradient->length);

    dim3 pool_2(pool_layer_2->output_gradient->length,
            pool_layer_2->output_gradient->length);

    int num_thread_1 = pool_layer_1->output_gradient->length *
            pool_layer_1->output_gradient->length;

    int num_thread_2 = pool_layer_2->output_gradient->length *
            pool_layer_2->output_gradient->length;

    softmax_kernel_gradient<<<num, output_layer->in_num>>>(
            output_layer->in_num,
            output_layer->input,
            dense_layer_3->output_gradient,
            output_layer->label);
    cudaDeviceSynchronize();


    dense_gradient_kernel<<<num, dense_layer_2->out_num,
        (dense_layer_2->in_num + dense_layer_2->out_num) * sizeof(float)>>>(
                dense_layer_3->in_num,
                dense_layer_3->out_num,
                dense_layer_3->input,
                dense_layer_3->weight_gradient,
                dense_layer_3->bias_gradient,
                dense_layer_3->output_gradient,
                num);
    cudaDeviceSynchronize();

    dense_gradient_input<<<num, dense_layer_2->in_num,
        dense_layer_2->out_num * sizeof(float)>>>(
                dense_layer_3->in_num,
                dense_layer_3->out_num,
                dense_layer_2->output_gradient,
                dense_layer_3->weight,
                dense_layer_3->output_gradient);
    cudaDeviceSynchronize();

    dense_gradient_update<<<1, dense_layer_3->in_num>>>(
            dense_layer_3->in_num,
            dense_layer_3->out_num,
            dense_layer_3->output_gradient,
            dense_layer_3->weight,
            dense_layer_3->weight_gradient,
            dense_layer_3->bias,
            dense_layer_3->bias_gradient,
            rate,
            num);
    cudaDeviceSynchronize();

    leaky_relu_gradient_kernel<<<num, dense_layer_2->out_num>>>(
            dense_layer_3->input,
            dense_layer_2->output_gradient,
            dense_layer_2->out_num,
            1);
    cudaDeviceSynchronize();

    dense_gradient_kernel<<<num, dense_layer_2->out_num,
        (dense_layer_2->in_num + dense_layer_2->out_num) * sizeof(float)>>>(
                dense_layer_2->in_num,
                dense_layer_2->out_num,
                dense_layer_2->input,
                dense_layer_2->weight_gradient,
                dense_layer_2->bias_gradient,
                dense_layer_2->output_gradient,
                num);
    cudaDeviceSynchronize();

    dense_gradient_input<<<num, dense_layer_2->in_num,
        dense_layer_2->out_num * sizeof(float)>>>(
                dense_layer_2->in_num,
                dense_layer_2->out_num,
                dense_layer_1->output_gradient,
                dense_layer_2->weight,
                dense_layer_2->output_gradient);
    cudaDeviceSynchronize();

    dense_gradient_update<<<1, dense_layer_2->in_num>>>(
            dense_layer_2->in_num,
            dense_layer_2->out_num,
            dense_layer_2->output_gradient,
            dense_layer_2->weight,
            dense_layer_2->weight_gradient,
            dense_layer_2->bias,
            dense_layer_2->bias_gradient,
            rate,
            num);
    cudaDeviceSynchronize();

    leaky_relu_gradient_kernel<<<num, dense_layer_1->out_num>>>(
            dense_layer_2->input,
            dense_layer_1->output_gradient,
            dense_layer_1->out_num,
            1);
    cudaDeviceSynchronize();

    dense_gradient_kernel<<<num, dense_layer_1->out_num,
        (dense_layer_1->in_num + dense_layer_1->out_num) * sizeof(float)>>>(
                dense_layer_1->in_num,
                dense_layer_1->out_num,
                dense_layer_1->input,
                dense_layer_1->weight_gradient,
                dense_layer_1->bias_gradient,
                dense_layer_1->output_gradient,
                num);
    cudaDeviceSynchronize();

    dense_gradient_input<<<num, dense_layer_1->in_num,
        dense_layer_1->out_num * sizeof(float)>>>(
                dense_layer_1->in_num,
                dense_layer_1->out_num,
                flat_layer->output_gradient->content,
                dense_layer_1->weight,
                dense_layer_1->output_gradient);
    cudaDeviceSynchronize();

    dense_gradient_update<<<1, dense_layer_1->in_num>>>(
            dense_layer_1->in_num,
            dense_layer_1->out_num,
            dense_layer_1->output_gradient,
            dense_layer_1->weight,
            dense_layer_1->weight_gradient,
            dense_layer_1->bias,
            dense_layer_1->bias_gradient,
            rate,
            num);
    cudaDeviceSynchronize();

    leaky_relu_gradient_kernel<<<num, num_thread_2>>>(
            flat_layer->input->content,
            flat_layer->output_gradient->content,
            flat_layer->output_gradient->length
            * flat_layer->output_gradient->length
            * flat_layer->output_gradient->channel,
            flat_layer->output_gradient->channel);
    cudaDeviceSynchronize();

    max_pooling_gradient_kernel<<<num, pool_2>>>(
            conv_layer_2->output_gradient,
            pool_layer_2->output_gradient,
            pool_layer_2->index,
            pool_layer_2->stride);
    cudaDeviceSynchronize();

    convolution_gradient_kernel<<<num, conv_kernel_2, conv_layer_2->backward_size_1 * sizeof(float)>>>(
            conv_layer_2->input,
            conv_layer_2->output_gradient,
            conv_layer_2->gradient,
            conv_layer_2->padding);
    cudaDeviceSynchronize();

    convolution_gradient_input<<<num, conv_input_2, conv_layer_2->backward_size_2 * sizeof(float)>>>(
            conv_layer_2->kernel,
            conv_layer_2->output_gradient,
            pool_layer_1->output_gradient,
            conv_layer_2->padding);
    cudaDeviceSynchronize();

    convolution_gradient_update<<<1, conv_layer_2->kernel->channel_in
        * conv_layer_2->kernel->length * conv_layer_2->kernel->length>>>(
                conv_layer_2->kernel,
                conv_layer_2->gradient,
                rate,
                num);
    cudaDeviceSynchronize();

    leaky_relu_gradient_kernel<<<num, num_thread_1>>>(
            conv_layer_2->input->content,
            pool_layer_1->output_gradient->content,
            pool_layer_1->output_gradient->length
            * pool_layer_1->output_gradient->length
            * pool_layer_1->output_gradient->channel,
            pool_layer_1->output_gradient->channel);
    cudaDeviceSynchronize();

    max_pooling_gradient_kernel<<<num, pool_1>>>(
            conv_layer_1->output_gradient,
            pool_layer_1->output_gradient,
            pool_layer_1->index,
            pool_layer_1->stride);
    cudaDeviceSynchronize();

    convolution_gradient_kernel<<<num, conv_kernel_1, conv_layer_1->backward_size_1 * sizeof(float)>>>(
            conv_layer_1->input,
            conv_layer_1->output_gradient,
            conv_layer_1->gradient,
            conv_layer_1->padding);
    cudaDeviceSynchronize();

    convolution_gradient_update<<<1, conv_layer_1->kernel->channel_in
        * conv_layer_1->kernel->length * conv_layer_1->kernel->length>>>(
                conv_layer_1->kernel,
                conv_layer_1->gradient,
                rate,
                num);
    cudaDeviceSynchronize();

}


// host function for training process
__host__ void train(net *network) {

    int n = network->n;
    int batch = network->batch;
    float rate = network->rate;
    int epoch = network->epoch;

    float *data = network->data;
    float *label = network->label;

    float *label_batch, *data_batch;
    float error;
    int correct;

    inp *input_layer = network->input_layer;
    outp *output_layer = network->output_layer;

    int data_length = input_layer->length
            * input_layer->length
            * input_layer->channel;

    int label_length = output_layer->in_num;

    for (int j = 0; j < epoch; j++) {

        for (int i = 0; i < n; i += batch) {

            cudaMemcpy(
                input_layer->content,
                &*(data + i * data_length),
                batch * data_length * sizeof(float),
                cudaMemcpyHostToDevice);

            cudaMemcpy(
                output_layer->label,
                &*(label + i * label_length),
                batch * label_length * sizeof(float),
                cudaMemcpyHostToDevice);

            forward(input_layer, batch);

            backward(output_layer, rate, batch);

        }

        label_batch = output_layer->label;
        data_batch = output_layer->input;

        error =softmax_corss_entropy(
                output_layer->in_num,
                data_batch,
                label_batch,
                batch);

        correct = predict(
                output_layer->in_num,
                data_batch,
                label_batch,
                batch);

        printf("Epoch: %d, Error: %.8f, Correct: %d\n", j, error * n / batch, correct);
    }
}


// host function for compute convolution
__host__ void convolution(
        img *input,
        img *output,
        ker *kernel,
        int padding,
        int stride,
        int num) {

    int n = (input->length + 2*padding) * (input->length + 2*padding) * input->channel
            + kernel->length * kernel->length * kernel->channel_out * kernel->channel_in
            + kernel->channel_out;

	dim3 dimBlock(output->length, output->length);

	convolution_kernel<<<num, dimBlock, n * sizeof(float)>>>
	    (input, output, kernel, padding, stride);
	cudaDeviceSynchronize();
}


// host function for compute accelerated convolution
__host__ void accelerated_convolution(
        img *input,
        img *output,
        ker *kernel,
        int padding,
        int stride,
        int num) {

    int n = (input->length + 2*padding) * (input->length + 2*padding) * input->channel
            + kernel->length * kernel->length * kernel->channel_out * kernel->channel_in
            + kernel->channel_out
            + kernel->channel_in * kernel->length * kernel-> length * output->length * output->length;

	dim3 dimBlock(output->length, output->length);

	accelerated_convolution_kernel<<<num, dimBlock, n * sizeof(float)>>>
	    (input, output, kernel, padding, stride);
	cudaDeviceSynchronize();
}