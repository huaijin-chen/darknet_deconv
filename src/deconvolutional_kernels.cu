#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
//cc
#include "activations.h"
}


//cc
void cc_printf_gpu(float* data, int _c, int _h, int _w)
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
void test_deconvolutional_layer_gpu()
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

    l.biases_gpu = cuda_make_array(b_data, 3*sizeof(float));

    l.weights_gpu = cuda_make_array(k_data, 3*3*3*sizeof(float));
    network_state state = {0};
    state.input = cuda_make_array(data, 5*5*3*sizeof(float));

    printf("---------------data--------------\n");
    float* cpu_data = (float*)malloc(5*5*3*sizeof(float));
    cudaMemcpy(cpu_data, state.input, 5*5*3*sizeof(float), cudaMemcpyDeviceToHost);
    cc_printf_gpu( cpu_data, 3, 5, 5);
    printf("---------------weights--------------\n");
    cudaMemcpy(l.weights, l.weights_gpu, 27*sizeof(float), cudaMemcpyDeviceToHost);
    cc_printf_gpu(l.weights, 3, 3, 3);
    printf("---------------biases--------------\n");
    cudaMemcpy(l.biases, l.biases_gpu, 3*sizeof(float), cudaMemcpyDeviceToHost);
    cc_printf_gpu(l.biases, 3, 1, 1);

    forward_deconvolutional_layer_gpu(l, state);
    //print output
    printf("------------------------------output---------------------\n");
    cudaMemcpy(l.output, l.output_gpu, l.out_c*l.out_h*l.out_w*l.batch*sizeof(float), cudaMemcpyDeviceToHost);
    cc_printf_gpu(l.output, l.out_c, l.out_h, l.out_w);

    //backward
    float *delta = (float*)calloc(l.batch*l.out_c*l.out_h*l.out_w, sizeof(float));
    float *state_delta = (float*)calloc(l.batch*l.c*l.h*l.w, sizeof(float));
    l.delta_gpu = cuda_make_array(delta, l.batch*l.out_c*l.out_h*l.out_w*sizeof(float));
    fill_ongpu(l.batch*l.out_c*l.out_h*l.out_w, 10.0, l.delta_gpu, 1);
    state.delta = cuda_make_array(state_delta, l.batch*l.c*l.h*l.w*sizeof(float));
    fill_ongpu(l.batch*l.c*l.h*l.w, 0.0, state.delta, 1);
    backward_deconvolutional_layer_gpu(l, state);

    printf("-----weights up-----------\n"); 
    cudaMemcpy(l.weight_updates,  l.weight_updates_gpu, l.c*l.size*l.size*sizeof(float), cudaMemcpyDeviceToHost);       
    cc_printf_gpu(l.weight_updates, l.c, l.size, l.size);
    printf("-----state  delta----- after------\n");
    float* cc_delta = (float*)calloc(l.batch*l.c*l.h*l.w, sizeof(float));
     cudaMemcpy(cc_delta, state.delta,l.batch*l.c*l.h*l.w*sizeof(float),  cudaMemcpyDeviceToHost);

    cc_printf_gpu(cc_delta, l.n, l.out_h, l.out_w);

    
    //free data 
    cuda_free(l.biases_gpu);
    cuda_free(l.weights_gpu);
    cuda_free(l.output_gpu);
    cuda_free(state.input);
    cuda_free(state.delta);
    cuda_free(l.delta_gpu);
    free(cpu_data);
    free(cc_delta);
    
    
}
extern "C" void forward_deconvolutional_layer_gpu(deconvolutional_layer layer, network_state state)
{
    int i;
    int out_h = deconvolutional_out_height(layer);
    int out_w = deconvolutional_out_width(layer);
    int size = out_h*out_w;

    int m = layer.size*layer.size*layer.n;
    int n = layer.h*layer.w;
    int k = layer.c;

    fill_ongpu(layer.outputs*layer.batch, 0, layer.output_gpu, 1);

    for(i = 0; i < layer.batch; ++i){
        float *a = layer.weights_gpu;
        float *b = state.input + i*layer.c*layer.h*layer.w;
        float *c = layer.col_image_gpu;
        //printf("deconvolutional:gemm M=%d N=%d K=%d  lda=%d ldb=%d ldc=%d\n", m, n, k, m, n, n);

        gemm_ongpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

        col2im_ongpu(c, layer.n, out_h, out_w, layer.size, layer.stride, 0, layer.output_gpu+i*layer.n*size);
    }
    add_bias_gpu(layer.output_gpu, layer.biases_gpu, layer.batch, layer.n, size);
    //activate_array(layer.output_gpu, layer.batch*layer.n*size, layer.activation);
    // cc
    activate_array_ongpu(layer.output_gpu, layer.batch*layer.n*size, layer.activation);
}

extern "C" void backward_deconvolutional_layer_gpu(deconvolutional_layer layer, network_state state)
{
    float alpha = 1./layer.batch;
    int out_h = deconvolutional_out_height(layer);
    int out_w = deconvolutional_out_width(layer);
    int size = out_h*out_w;
    int i;


    //cc
    //gradient_array(layer.output_gpu, size*layer.n*layer.batch, layer.activation, layer.delta_gpu);
    //backward_bias(layer.bias_updates_gpu, layer.delta, layer.batch, layer.n, size);
    gradient_array_ongpu(layer.output_gpu, size*layer.n*layer.batch, layer.activation, layer.delta_gpu);
    backward_bias_gpu(layer.bias_updates_gpu, layer.delta_gpu, layer.batch, layer.n, size);

    //cc
    //if(state.delta) memset(state.delta, 0, layer.batch*layer.h*layer.w*layer.c*sizeof(float));
    //if(state.delta) cuda_make_array(state.delta, layer.batch*layer.h*layer.w*layer.c*sizeof(float));
   if(state.delta)fill_ongpu(layer.batch*layer.h*layer.w*layer.c, 0, state.delta, 1);

    for(i = 0; i < layer.batch; ++i){
        int m = layer.c;
        //hhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
        int n = layer.size*layer.size*layer.n;
        int k = layer.h*layer.w;

        float *a = state.input + i*m*k;
        float *b = layer.col_image_gpu;
        float *c = layer.weight_updates_gpu;

        im2col_ongpu(layer.delta_gpu + i*layer.n*size, layer.n, out_h, out_w, 
                layer.size, layer.stride, 0, b);
        //printf("hhhh  wwwwww %d %d \n", out_h, out_w);
        gemm_ongpu(0,1,m,n,k,alpha,a,k,b,k,1,c,n);

        if(state.delta){
            int m = layer.c;
            int n = layer.h*layer.w;
            int k = layer.size*layer.size*layer.n;

            float *a = layer.weights_gpu;
            float *b = layer.col_image_gpu;
            float *c = state.delta + i*n*m;
            
            //cc
            //gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
            gemm_ongpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
    /*
    printf("deconv-back\n");
    printf("-----weights up-----------\n"); 
    cudaMemcpy(layer.weight_updates,  layer.weight_updates_gpu, layer.c*layer.size*layer.size*sizeof(float), cudaMemcpyDeviceToHost);       
    cc_printf_gpu(layer.weight_updates, layer.c, layer.size, layer.size);
    printf("-----state  delta----- after------\n");
    float* cc_delta = (float*)calloc(layer.batch*layer.c*layer.h*layer.w, sizeof(float));
     cudaMemcpy(cc_delta, state.delta,layer.batch*layer.c*layer.h*layer.w*sizeof(float),  cudaMemcpyDeviceToHost);

    cc_printf_gpu(cc_delta, layer.n, layer.out_h, layer.out_w);
    free(cc_delta);
    */
}

extern "C" void pull_deconvolutional_layer(deconvolutional_layer layer)
{
    cuda_pull_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    printf("deconv-pull\n");
}

extern "C" void push_deconvolutional_layer(deconvolutional_layer layer)
{
    cuda_push_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    printf("deconv-push\n");
}

extern "C" void update_deconvolutional_layer_gpu(deconvolutional_layer layer, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;

    axpy_ongpu(layer.n, learning_rate, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    axpy_ongpu(size, -decay, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);
    axpy_ongpu(size, learning_rate, layer.weight_updates_gpu, 1, layer.weights_gpu, 1);
    scal_ongpu(size, momentum, layer.weight_updates_gpu, 1);

    printf("deconv-update\n");
}

