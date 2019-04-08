//
// Created by zeng on 2019/4/8.
//
#include <iostream>
#include "nn.h"
using namespace std;


int main() {
    cout << "异或" << endl;


    FullyConnectedNetwork* p_net = new FullyConnectedNetwork(3, 2, 5, 2, sigmoid, d_sigmoid);
    p_net->InitParam(Norm_random);
    gradient_check_net(p_net);
}

