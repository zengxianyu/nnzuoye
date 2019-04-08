//
// Created by zeng on 2019/4/8.
//

#include "activation.h"

double sigmoid(double x) {
    return 1/(1+exp(-1*x));
}

double d_sigmoid(double x) {
    double s = sigmoid(x);
    return s*(1-s);
}


double pass(double x) {
    return x;
}

double d_pass(double x) {
    double dx = 1;
    return dx;
}
