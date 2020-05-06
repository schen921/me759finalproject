#include <iostream>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "network.h"


using namespace std;


// initialize input layer
inp* init_in(
        int input_length,
        int input_channel) {

    inp *input_layer;
    cudaMallocManaged((void **)&input_layer, sizeof(inp));

    input_layer->length = input_length;
    input_layer->channel = input_channel;

    return input_layer;
}


// initialize output layer
outp* init_out(
        int in_num,
        int n) {

    outp *output_layer;
    cudaMallocManaged((void **)&output_layer, sizeof(outp));

    output_layer->in_num = in_num;

    float *input, *label;

    cudaMallocManaged((void **) &input, n * in_num * sizeof(float));
    cudaMallocManaged((void **) &label, n * in_num * sizeof(float));

    output_layer->input = input;
    output_layer->label = label;

    return output_layer;
}


// initialize fully connected layer
den* init_dense(
        int in_num,
        int out_num,
        int n) {

    den *dense_layer;
    cudaMallocManaged((void **)&dense_layer, sizeof(den));

    dense_layer->in_num = in_num;
    dense_layer->out_num = out_num;

    float *input, *weight;
    float *weight_gradient, *output_gradient;
    float *bias_gradient;

    cudaMallocManaged((void **) &input, n * in_num * sizeof(float));
    cudaMallocManaged((void **) &weight, in_num * out_num * sizeof(float));

    cudaMallocManaged((void **) &weight_gradient, n * in_num * out_num * sizeof(float));
    cudaMallocManaged((void **) &output_gradient, n * out_num * sizeof(float));
    cudaMallocManaged((void **) &bias_gradient, out_num * sizeof(float));

    for (int i = 0; i< in_num * out_num; i++) {
        *(weight + i) =  rand() / float(RAND_MAX) * 0.2 - 0.1;
    }

    dense_layer->bias = rand() / float(RAND_MAX) * 0.2 - 0.1;

    dense_layer->input = input;
    dense_layer->weight = weight;
    dense_layer->weight_gradient = weight_gradient;
    dense_layer->output_gradient = output_gradient;
    dense_layer->bias_gradient = bias_gradient;

    return dense_layer;
}


// initialize convolution layer
conv* init_conv(
        int stride,
        int padding,
        int input_length,
        int input_channel,
        int kernel_length,
        int output_channel,
        int n) {

    conv *conv_layer;
    cudaMallocManaged((void **)&conv_layer, sizeof(conv));

	img *input, *output_gradient;
	ker *kernel, *gradient;

	cudaMallocManaged((void **)&input, sizeof(img));
    cudaMallocManaged((void **)&output_gradient, sizeof(img));
    cudaMallocManaged((void **)&kernel, sizeof(ker));
    cudaMallocManaged((void **)&gradient, sizeof(ker));

	conv_layer->stride = stride;
    conv_layer->padding = padding;

    input->length = input_length;
    input->channel = input_channel;

    kernel->length = kernel_length;
    kernel->channel_in = input_channel;
    kernel->channel_out = output_channel;

    gradient->length = kernel_length;
    gradient->channel_in = input_channel;
    gradient->channel_out = output_channel;

    output_gradient->length = ((input_length + 2 * padding)
            - kernel_length) / stride + 1;
    output_gradient->channel = output_channel;

    cudaMallocManaged((void **)&input->content,
            n * input->length * input->length * input->channel * sizeof(float));

    cudaMallocManaged((void **)&output_gradient->content,
            n * output_gradient->length * output_gradient->length * output_gradient->channel *sizeof(float));

    cudaMallocManaged((void **)&kernel->content,
            kernel->length * kernel->length * kernel->channel_out * kernel->channel_in * sizeof(float));
    cudaMallocManaged((void **)&kernel->bias, kernel->channel_out*sizeof(float));

    cudaMallocManaged((void **)&gradient->content,
            n * gradient->length * gradient->length * gradient->channel_out * gradient->channel_in * sizeof(float));
    cudaMallocManaged((void **)&gradient->bias, n * gradient->channel_out*sizeof(float));

    for (int i = 0; i< kernel->length * kernel->length * kernel->channel_out * kernel->channel_in; i++) {
        *(kernel->content + i) =  rand() / float(RAND_MAX) * 0.2 - 0.1;
    }

    for (int i = 0; i< kernel->channel_out; i++) {
        *(kernel->bias + i) =  rand() / float(RAND_MAX) * 0.2 - 0.1;
    }

    conv_layer->input = input;
    conv_layer->output_gradient = output_gradient;
    conv_layer->kernel = kernel;
    conv_layer->gradient = gradient;

    conv_layer->forward_size =
            (input_length + 2*padding) * (input_length + 2*padding) * input_channel
            + kernel_length * kernel_length * output_channel * input_channel
            + output_channel;

    conv_layer->backward_size_1 =
            (input_length + 2*padding) * (input_length + 2*padding) * input_channel
            + output_gradient->length * output_gradient->length * output_gradient->channel;

    int padding_length = output_gradient->length + 2 * (kernel->length - padding - 1);
    conv_layer->backward_size_2 =
            padding_length * padding_length * output_channel
            + kernel_length * kernel_length * output_channel * input_channel;

    return  conv_layer;
}


// initialize maxpooling layer
maxpool* init_pool(
        int stride,
        int input_length,
        int input_channel,
        int n) {

    maxpool *pool_layer;
    cudaMallocManaged((void **) &pool_layer, sizeof(maxpool));

    img *input, *output_gradient;
    int *index;

    cudaMallocManaged((void **) &input, sizeof(img));
    cudaMallocManaged((void **) &output_gradient, sizeof(img));

    pool_layer->stride = stride;

    input->length = input_length;
    input->channel = input_channel;

    output_gradient->length = input_length / 2;
    output_gradient->channel = input_channel;

    cudaMallocManaged((void **)&input->content,
            n * input->length * input->length * input->channel * sizeof(float));
    cudaMallocManaged((void **)&output_gradient->content,
            n * output_gradient->length * output_gradient->length * output_gradient->channel *sizeof(float));

    cudaMallocManaged((void **)&index,
            n * output_gradient->length * output_gradient->length * output_gradient->channel *sizeof(int));

    pool_layer->input = input;
    pool_layer->output_gradient = output_gradient;
    pool_layer->index = index;

    return pool_layer;
}


// initialize flatten layer
flat *init_flat(
        int input_length,
        int input_channel,
        int n) {

    flat *flat_layer;
    cudaMallocManaged((void **) &flat_layer, sizeof(flat));

    img *input;

    cudaMallocManaged((void **) &input, sizeof(img));

    input->length = input_length;
    input->channel = input_channel;

    flat_layer->input = input;

    return flat_layer;
}


// initialize network
net *init_net(int n) {

    int conv_stride = 1;

	int input_length = 28;
	int input_channel = 1;

	int conv_padding_1 = 2;
	int kernel_length_1 = 5;
	int output_channel_1 = 6;

	int conv_padding_2 = 0;
	int kernel_length_2 = 5;
	int output_channel_2 = 16;

	int pool_stride = 2;

	int out_num_1 = 120;
	int out_num_2 = 84;
	int out_num_3 = 10;

	net *network;
	cudaMallocManaged((void **) &network, sizeof(net));

	inp *input_layer = init_in(
	        input_length,
	        input_channel);

	conv *conv_layer_1 = init_conv(
	        conv_stride,
	        conv_padding_1,
	        input_length,
	        input_channel,
	        kernel_length_1,
	        output_channel_1,
	        n);

	maxpool *pool_layer_1 = init_pool(
	        pool_stride,
	        conv_layer_1->output_gradient->length,
	        conv_layer_1->output_gradient->channel,
	        n);

	conv *conv_layer_2 = init_conv(
	        conv_stride,
	        conv_padding_2,
	        pool_layer_1->output_gradient->length,
	        pool_layer_1->output_gradient->channel,
	        kernel_length_2,
	        output_channel_2,
	        n);

	maxpool *pool_layer_2 = init_pool(
	        pool_stride,
	        conv_layer_2->output_gradient->length,
	        conv_layer_2->output_gradient->channel,
	        n);

	flat *flat_layer = init_flat(
	        pool_layer_2->output_gradient->length,
	        pool_layer_2->output_gradient->channel,
	        n);

	den *dense_layer_1 = init_dense(
            pool_layer_2->output_gradient->length
            * pool_layer_2->output_gradient->length
            * pool_layer_2->output_gradient->channel,
            out_num_1,
            n);

	den *dense_layer_2 = init_dense(
	        out_num_1,
	        out_num_2,
	        n);

	den *dense_layer_3 = init_dense(
	        out_num_2,
	        out_num_3,
	        n);

	outp *output_layer =  init_out(
	        out_num_3,
	        n);

	input_layer->next = (void *) conv_layer_1;
	conv_layer_1->previous = (void *) input_layer;

	conv_layer_1->next = (void *) pool_layer_1;
	pool_layer_1->previous = (void*) conv_layer_1;

	pool_layer_1->next = (void *) conv_layer_2;
	conv_layer_2->previous = (void *) pool_layer_1;

	conv_layer_2->next = (void *) pool_layer_2;
	pool_layer_2->previous = (void*) conv_layer_2;

	pool_layer_2->next = (void *) flat_layer;
	flat_layer->previous = (void *) pool_layer_2;

	flat_layer->next = (void *) dense_layer_1;
	dense_layer_1->previous = (void *) flat_layer;

	dense_layer_1->next = (void *) dense_layer_2;
	dense_layer_2->previous = (void *) dense_layer_1;

	dense_layer_2->next = (void *) dense_layer_3;
	dense_layer_3->previous = (void *) dense_layer_2;

    dense_layer_3->next = (void *) output_layer;
	output_layer->previous = (void *) dense_layer_3;

	input_layer->content = conv_layer_1->input->content;
	flat_layer->input->content = dense_layer_1->input;
	flat_layer->output_gradient = pool_layer_2->output_gradient;

	network->input_layer = input_layer;
	network->output_layer = output_layer;

	return network;
}


// free network
void free_net(net *network) {

    inp *input_layer;
    conv *conv_layer;
    maxpool *pool_layer;
    flat *flat_layer;
    den *dense_layer;
    den *dense_layer_2;
    outp *output_layer;

    input_layer = network->input_layer;

    conv_layer = (conv *) input_layer->next;
    cudaFree(input_layer);

    pool_layer = (maxpool *) conv_layer->next;
    cudaFree(conv_layer->input->content);
    cudaFree(conv_layer->input);
    cudaFree(conv_layer->output_gradient->content);
    cudaFree(conv_layer->output_gradient);
    cudaFree(conv_layer->kernel->content);
    cudaFree(conv_layer->kernel->bias);
    cudaFree(conv_layer->kernel);
    cudaFree(conv_layer->gradient->content);
    cudaFree(conv_layer->gradient->bias);
    cudaFree(conv_layer->gradient);
    cudaFree(conv_layer);

    conv_layer = (conv *) pool_layer->next;
    cudaFree(pool_layer->input->content);
    cudaFree(pool_layer->input);
    cudaFree(pool_layer->output_gradient->content);
    cudaFree(pool_layer->output_gradient);
    cudaFree(pool_layer->index);
    cudaFree(pool_layer);

    pool_layer = (maxpool *) conv_layer->next;
    cudaFree(conv_layer->input->content);
    cudaFree(conv_layer->input);
    cudaFree(conv_layer->output_gradient->content);
    cudaFree(conv_layer->output_gradient);
    cudaFree(conv_layer->kernel->content);
    cudaFree(conv_layer->kernel->bias);
    cudaFree(conv_layer->kernel);
    cudaFree(conv_layer->gradient->content);
    cudaFree(conv_layer->gradient->bias);
    cudaFree(conv_layer->gradient);
    cudaFree(conv_layer);

    flat_layer = (flat *) pool_layer->next;
    cudaFree(pool_layer->input->content);
    cudaFree(pool_layer->input);
    cudaFree(pool_layer->output_gradient->content);
    cudaFree(pool_layer->output_gradient);
    cudaFree(pool_layer->index);
    cudaFree(pool_layer);

    dense_layer = (den *) flat_layer->next;
    cudaFree(pool_layer->input);
    cudaFree(pool_layer);

    dense_layer_2 = (den *) dense_layer->next;
    cudaFree(dense_layer->input);
    cudaFree(dense_layer->output_gradient);
    cudaFree(dense_layer->weight);
    cudaFree(dense_layer->weight_gradient);
    cudaFree(dense_layer->bias_gradient);
    cudaFree(dense_layer);

    dense_layer = (den *) dense_layer_2->next;
    cudaFree(dense_layer_2->input);
    cudaFree(dense_layer_2->output_gradient);
    cudaFree(dense_layer_2->weight);
    cudaFree(dense_layer_2->weight_gradient);
    cudaFree(dense_layer_2->bias_gradient);
    cudaFree(dense_layer_2);

    output_layer = (outp *) dense_layer->next;
    cudaFree(dense_layer->input);
    cudaFree(dense_layer->output_gradient);
    cudaFree(dense_layer->weight);
    cudaFree(dense_layer->weight_gradient);
    cudaFree(dense_layer->bias_gradient);
    cudaFree(dense_layer);

    cudaFree(output_layer->input);
    cudaFree(output_layer->label);
    cudaFree(output_layer);

    cudaFree(network->label);
    cudaFree(network->data);
    cudaFree(network);
}


// read data
float *read_data(int n, int length, char* name) {

    n = n * length * length;

    float *data;
    cudaMallocManaged((void **) &data, n * sizeof(float));

    ifstream infile(name);

    for (int i = 0; i < n; i++) {
        infile>>data[i];
        data[i] /= 255;
    }

    return data;
}


// read label
float *read_label(int n, int length, char* name) {

    n = n * length;

    float *data;
    cudaMallocManaged((void **) &data, n * sizeof(float));

    ifstream infile(name);

    for (int i = 0; i < n; i++) {
        infile>>data[i];
    }

    return data;
}

int main(int argc, char *argv[]) {

    int n = 10000;
	int batch = 100;
	int epoch = 20;
	float rate = 0.01;

	int image_length = 28;
	int label_length = 10;

	// read data
	char train_data_name[15] = "traindata.txt";
	float *train_data  = read_data(n, image_length, train_data_name);

	// read label
	char train_label_name[15] = "trainlabel.txt";
	float *train_label  = read_label(n, label_length, train_label_name);

	// initialize network
	net *network = init_net(batch);

	network->data = train_data;
	network->label = train_label;

	network->batch = batch;
	network->rate = rate;
	network->n = n;
	network->epoch = epoch;

	// train network
	train(network);

	// free network
	free_net(network);

}