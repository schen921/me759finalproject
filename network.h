#ifndef NETWORK_H
#define NETWORK_H

#include <cstddef>

typedef struct Img {
    int length;
    int channel;
    float *content;
} img;

typedef struct Kernel {
    int length;
    int channel_in;
    int channel_out;
    float *bias;
    float *content;
} ker;

typedef struct ConvLayer{
    int padding;
    int stride;
    img *input;
    img *output_gradient;
    ker *kernel;
    ker *gradient;
    void *next;
    void *previous;
    int forward_size;
    int backward_size_1;
    int backward_size_2;
} conv;

typedef struct MaxPoolingLayer{
    int stride;
    img *input;
    img *output_gradient;
    int *index;
    void *next;
    void *previous;
} maxpool;

typedef struct FlattenLayer{
    img *input;
    img *output_gradient;
    void *next;
    void *previous;
} flat;

typedef struct DenseLayer{
    int in_num;
    int out_num;
    float *input;
    float *output_gradient;
    float *weight;
    float *weight_gradient;
    float bias;
    float *bias_gradient;
    void *next;
    void *previous;
} den;

typedef struct InputputLayer{
    int length;
    int channel;
    float *content;
    void *next;
} inp;

typedef struct OutputLayer{
    int in_num;
    float *input;
    float *label;
    void *previous;
} outp;

typedef struct Net{
    inp *input_layer;
    outp *output_layer;
    float *data;
    float *label;
    int n;
    int batch;
    int epoch;
    float rate;
} net;

__host__ void train(
        net *network);

__host__ void forward(
        void *layer,
        int num);

__host__ void backward(
        void *layer,
        float rate,
        int num);

__host__ void convolution(
        img *input,
        img *output,
        ker *kernel,
        int padding,
        int stride,
        int num);

__host__ void accelerated_convolution(
        img *input,
        img *output,
        ker *kernel,
        int padding,
        int stride,
        int num);

#endif
