//
// Created by zeng on 2019/4/8.
//

#ifndef UNTITLED1_NN_H
#define UNTITLED1_NN_H
#include <iostream>
#include "matrix.h"
#include "normal_random.h"
#include "activation.h"

class Linear {
public:
    int n_i;
    int n_o;
    Matrix* p_weight=NULL; // weight
    Matrix* p_bias=NULL; // bias
    Matrix* p_hidden=NULL; // hidden = W input, 激活函数之前的值。forward之后一直存在。backward后释放
    Matrix* p_input=NULL; // input。forward之后一直存在。backward后释放
    Matrix* p_g_weight=NULL; // 这层weight的梯度。backward之后一直存在
    Matrix* p_g_bias=NULL; // 这层bias的梯度。backward之后一直存在
    double (*activation_func)(double);
    double (*d_activation_func)(double);
    Linear(int num_input, int num_output, double (*act)(double)=pass, double (*d_act)(double)=d_pass);
    ~Linear();
    Matrix* Forward(Matrix* p_input, bool keep_input=false);
    Matrix* Backward(Matrix* p_g_output, bool keep_input=false);
    void Optimize(double lr=1e-4);

};


class FullyConnectedNetwork {
public:
    int n_input;
    int n_output;
    int n_hidden;
    int hidden_layer;
    int n_layer;
    FullyConnectedNetwork(int num_input, int num_output, int num_hidden, int num_layer, double (*act)(double)=sigmoid, double (d_act)(double)=d_sigmoid);
    ~FullyConnectedNetwork();
    void InitParam(double (*func)()=Norm_random);
    Linear** pp_layer=NULL;
    Matrix* Forward(Matrix* _p_input);
    Matrix* Backward(Matrix* _p_g_input);
    void Optimize(double lr=1e-4);
};

int gradient_check_net(FullyConnectedNetwork* p_net, double delta=1e-4);
int gradient_check_linear(Linear* p_fc, double delta=1e-4);

#endif //UNTITLED1_NN_H
