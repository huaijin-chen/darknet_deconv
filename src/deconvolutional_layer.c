#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

int deconvolutional_out_height(deconvolutional_layer l)
{
    int h = l.stride*(l.h - 1) + l.size;
    return h;
}

int deconvolutional_out_width(deconvolutional_layer l)
{
    int w = l.stride*(l.w - 1) + l.size;
    return w;
}

int deconvolutional_out_size(deconvolutional_layer l)
{
    return deconvolutional_out_height(l) * deconvolutional_out_width(l);
}

image get_deconvolutional_image(deconvolutional_layer l)
{
    int h,w,c;
    h = deconvolutional_out_height(l);
    w = deconvolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.output);
}

image get_deconvolutional_delta(deconvolutional_layer l)
{
    int h,w,c;
    h = deconvolutional_out_height(l);
    w = deconvolutional_out_width(l);
    c = l.n;
    return float_to_image(w,h,c,l.delta);
}

deconvolutional_layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, ACTIVATION activation)
{
    int i;
    deconvolutional_layer l = {0};
    l.type = DECONVOLUTIONAL;

    l.h = h; // input h
    l.w = w; // input w
    l.c = c; // input channels
    l.n = n; // output channels
    l.batch = batch;
    l.stride = stride;
    l.size = size;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.weight_updates = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));
    float scale = 1./sqrt(size*size*c);
    for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_normal();
    for(i = 0; i < n; ++i){
        l.biases[i] = scale;
    }
    int out_h = deconvolutional_out_height(l);
    int out_w = deconvolutional_out_width(l);

    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;

    //l.col_image = calloc(out_h*out_w*size*size*n, sizeof(float));
    l.col_image = calloc(h*w*size*size*n, sizeof(float));
    l.output = calloc(l.batch*out_h * out_w * n, sizeof(float));
    l.delta  = calloc(l.batch*out_h * out_w * n, sizeof(float));

    l.forward = forward_deconvolutional_layer;
    l.backward = backward_deconvolutional_layer;
    l.update = update_deconvolutional_layer;

    #ifdef GPU
    //cc
    l.forward_gpu = forward_deconvolutional_layer_gpu;
    l.backward_gpu = backward_deconvolutional_layer_gpu;
    l.update_gpu = update_deconvolutional_layer_gpu;


    l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
    l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

    l.biases_gpu = cuda_make_array(l.biases, n);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

    //l.col_image_gpu = cuda_make_array(l.col_image, out_h*out_w*size*size*n);
    l.col_image_gpu = cuda_make_array(l.col_image, h*w*size*size*n);
    l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
    l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
    #endif

    l.activation = activation;

    //fprintf(stderr, "Deconvolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);
    //cc
    fprintf(stderr, "deconv  %3d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

    return l;
}

void resize_deconvolutional_layer(deconvolutional_layer *l, int h, int w)
{
    l->h = h;
    l->w = w;
    
    int out_h = deconvolutional_out_height(*l);
    int out_w = deconvolutional_out_width(*l);

    //cc
    l->out_h = out_h;
    l->out_w = out_w;
    l->outputs = out_w * out_h * l->out_c;
    l->inputs = w * h * l->c;

    //l->col_image = realloc(l->col_image,
    //                            out_h*out_w*l->size*l->size*l->c*sizeof(float));
    l->col_image = realloc(l->col_image,
                                h*w*l->size*l->size*l->n*sizeof(float));
    l->output = realloc(l->output,
                                l->batch*out_h * out_w * l->n*sizeof(float));
    l->delta  = realloc(l->delta,
                                l->batch*out_h * out_w * l->n*sizeof(float));
    #ifdef GPU
    cuda_free(l->col_image_gpu);
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->col_image_gpu = cuda_make_array(l->col_image,h*w*l->size*l->size*l->n);
    l->delta_gpu = cuda_make_array(l->delta, l->batch*out_h*out_w*l->n);
    l->output_gpu = cuda_make_array(l->output, l->batch*out_h*out_w*l->n);
    #endif
    //cc
    l->workspace_size = get_workspace_size(*l);
}

void cc_printf(float* data, int _c, int _h, int _w)
{
    int c, h, w;
    
    for (c = 0; c < _c; c++){
        for(h = 0; h < _h; h++){
            for(w = 0; w < _w; w++){
                printf("%.02f,", data[c*_h*_w + h*_w + w]);
            }
            printf("\n");
        }
        printf("--------------------------------\n");
    }
}
void test_deconvolutional_layer()
{
    //(int batch, int h, int w, int c, int n, int size, int stride, ACTIVATION activation)
    deconvolutional_layer l = make_deconvolutional_layer(1, 5, 5, 3, 1, 3, 1, LEAKY);
    float data[] = {1,1,1,1,1, // 5x5x3
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    float k_data[] = {1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    2, 2, 2,
    2, 2, 2,
    2, 2, 2,
    3, 3, 3,
    3, 3, 3,
    3, 3, 3};
    float b_data[] = {0.5, 0.5,0.5};

    l.biases = b_data;
    l.weights = k_data;
    network_state state = {0};
    state.input = data;

    printf("---------------data--------------\n");
    cc_printf(state.input, 3, 5, 5);
    printf("---------------weights--------------\n");
    cc_printf(l.weights, 3, 3, 3);
    printf("---------------biases--------------\n");
    cc_printf(l.biases, 3, 1, 1);

    forward_deconvolutional_layer(l, state);
    //print output
    printf("------------------------------output---------------------\n");
    cc_printf(l.output, l.out_c, l.out_h, l.out_w);

    float *delta = (float*)calloc(l.batch*l.out_c*l.out_h*l.out_w, sizeof(float));
    fill_cpu(l.batch*l.out_c*l.out_h*l.out_w, 10.0, delta, 1);
    l.delta = delta;
    float *state_delta = (float*)calloc(l.batch*l.c*l.h*l.w, sizeof(float));
    fill_cpu(l.batch*l.c*l.h*l.w, 0, state_delta, 1);
    state.delta = state_delta;

    printf("-----state  delta----- before------\n");
    cc_printf(state.delta, l.n, l.out_h, l.out_w);
    backward_deconvolutional_layer(l, state);
    printf("-----weights up-----------\n");
    cc_printf(l.weight_updates, l.c, l.size, l.size);
    printf("-----state  delta----- after------\n");
    cc_printf(state.delta, l.n, l.out_h, l.out_w);

    free(delta);
    free(state_delta);
}
void forward_deconvolutional_layer(const deconvolutional_layer l, network_state state)
{
    int i;
    int out_h = deconvolutional_out_height(l);
    int out_w = deconvolutional_out_width(l);
    int size = out_h*out_w;

    int m = l.size*l.size*l.n; //cc kernel's count
    int n = l.h*l.w; // singele input feature map's count
    int k = l.c; // input channels

    // l.output -- the count of all output feature maps 
    // l.batch  -- batch size 
    fill_cpu(l.outputs*l.batch, 0, l.output, 1); 

    for(i = 0; i < l.batch; ++i){
        // for eche image
        float *a = l.weights;
        float *b = state.input + i*l.c*l.h*l.w;
        float *c = l.col_image;
        //cc 
        /* a -- weights
         * b -- input 
         * c -- output
         */


        //printf("mmmmmmm----nnnnnnn --- %d, %d\n", m, n);
        //printf("deconvolutional:gemm M=%d N=%d K=%d  lda=%d ldb=%d ldc=%d\n", m, n, k, m, n, n);
        // printf("2222mmmmmmm----nnnnnnn --- %d, %d\n", m, n);
        //printf("a --> %d = %d * %d * %d * %d\n", l.c*l.n*l.size*l.size, l.c,l.n,l.size, l.size);
        //printf("b --> %d = %d * %d * %d\n", l.c*l.h*l.w, l.c,l.h,l.w);
        //printf("c --> %d = %d * %d * %d* %d * %d\n", l.out_h*l.out_w*l.size*l.size*l.c,
        //       l.out_h,l.out_w,l.size,l.size,l.c);

        gemm(1,0,m,n,k,1,a,m,b,n,0,c,n);
        //printf("---------c -------------\n");
        //cc_printf(c, 2, n, k);
        col2im_cpu(c, l.n, out_h, out_w, l.size, l.stride, 0, l.output+i*l.n*size);
    }
    add_bias(l.output, l.biases, l.batch, l.n, size);
    activate_array(l.output, l.batch*l.n*size, l.activation);
}

void backward_deconvolutional_layer(deconvolutional_layer l, network_state state)
{
    float alpha = 1./l.batch;
    int out_h = deconvolutional_out_height(l);
    int out_w = deconvolutional_out_width(l);
    int size = out_h*out_w;
    int i;

    gradient_array(l.output, size*l.n*l.batch, l.activation, l.delta);
    backward_bias(l.bias_updates, l.delta, l.batch, l.n, size);

    for(i = 0; i < l.batch; ++i){
        int m = l.c;
        int n = l.size*l.size*l.n;
        int k = l.h*l.w;

        float *a = state.input + i*m*k;
        float *b = l.col_image;
        float *c = l.weight_updates;


        //printf("------------------backward state.input------\n");
        //cc_printf(state.input, l.c, l.h, l.w);
        //printf("------------------weight_update before\n");
        //cc_printf(l.weight_updates, l.c, l.size, l.size);
        im2col_cpu(l.delta + i*l.n*size, l.n, out_h, out_w, 
                l.size, l.stride, 0, b);
        //input ** state.deta
        gemm(0,1,m,n,k,alpha,a,k,b,k,1,c,n);
        //printf("------------------weight_update after\n");
        //cc_printf(l.weight_updates, l.c, l.size, l.size);

        if(state.delta){
            int m = l.c;
            int n = l.h*l.w;
            int k = l.size*l.size*l.n;

            float *a = l.weights;
            float *b = l.col_image;
            float *c = state.delta + i*n*m;

            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
}

void update_deconvolutional_layer(deconvolutional_layer l, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    axpy_cpu(size, -decay, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}



