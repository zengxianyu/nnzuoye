//
// Created by zeng on 2019/4/8.
//
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <cassert>
#include "nn.h"
using namespace std;


Matrix* Label_XOR(Matrix* x_vec) {
    assert((x_vec->row == 2));
    Matrix* y_vec = new Matrix(1, x_vec->col);
    for(int j=0;j<x_vec->col;++j){
        y_vec->p_row[0][j] = double(bool(x_vec->p_row[0][j]) ^ bool(x_vec->p_row[1][j]));
    }
    return y_vec;
}


Matrix* Get_XOR_data(int num) {
    srand(time(0));
    Matrix* x_vec = new Matrix(2, num);
    for(int j=0;j<num;++j){
        x_vec->p_row[0][j] = rand()%2;
        x_vec->p_row[1][j] = rand()%2;
    }
    return x_vec;
}


double Train_Net(FullyConnectedNetwork* p_net, Matrix* x_train, Matrix* y_train, ofstream* p_outfile=NULL) {
    assert((p_net->n_input == x_train->row));
    Matrix* x;
    Matrix* y;
    Matrix* pred;
    Matrix* g_y_pred;
    double loss = 0;
    for(int t=0;t<x_train->col;++t){
        x = x_train->Get_col(t);
        y = y_train->Get_col(t);

        pred = p_net->Forward(x);
        g_y_pred = (*pred)-(*y);

        p_net->Backward(g_y_pred);
        p_net->Optimize(0.1);
        loss += g_y_pred->p_row[0][0]*g_y_pred->p_row[0][0]*0.5;
        if(p_outfile != NULL)
            *p_outfile << loss/(t+1) << endl;
        cout << "训练损失"<<loss/(t+1) << endl;
        delete pred; delete g_y_pred; delete x; delete y;
    }
    return loss / x_train->col;
}


double Validate_Net(FullyConnectedNetwork* p_net, Matrix* x_val, Matrix* y_val, double* return_loss=NULL) {
    Matrix* x;
    Matrix* pred;
    Matrix* g_y_pred;
    Matrix* y;
    Matrix* pred_all = new Matrix(1, y_val->col);
    double loss = 0;
    for(int t=0;t<x_val->col;++t) {
        x = x_val->Get_col(t);
        y = y_val->Get_col(t);
        pred = p_net->Forward(x);
        pred_all->p_row[0][t] = pred->p_row[0][0];
        g_y_pred = (*pred)-(*y);
        loss += g_y_pred->p_row[0][0]*g_y_pred->p_row[0][0]*0.5;
        delete pred; delete x; delete y; delete g_y_pred;
    }
    loss /= x_val->col;
    pred_all->Threshold_(0.5, 1, 0);
    Matrix* err_vec = (*pred_all)!=(*y_val);
    double err_rate = err_vec->Sum() / y_val->col;
    delete err_vec;
    delete pred_all;
    if(return_loss) *return_loss=loss;
    return err_rate;
}


int XOR() {

    cout << "异或" << endl;
    int num_train = 10000;
    int num_val = 100;
    double loss = 0;
    Matrix* x_train_all = Get_XOR_data(num_train);
    Matrix* y_train_all = Label_XOR(x_train_all);
    Matrix* x_val_all = Get_XOR_data(num_val);
    Matrix* y_val_all = Label_XOR(x_val_all);

    ofstream outfile;
    outfile.open("train_loss.txt", ios::trunc);

    FullyConnectedNetwork* p_net = new FullyConnectedNetwork(2, 1, 2, 1, sigmoid, d_sigmoid);
    p_net->InitParam(Norm_random);
    // train
    Train_Net(p_net, x_train_all, y_train_all, &outfile);
    //训练集正确率
    double err_rate = Validate_Net(p_net, x_train_all, y_train_all);
    cout << "训练集正确率" << 1-err_rate << endl;
    //检验集正确率, 误差
    err_rate = Validate_Net(p_net, x_val_all, y_val_all, &loss);
    cout << "检验损失" << loss / num_val << endl;
    cout << "检验集正确率" << 1-err_rate << endl;
    cout << "训练迭代次数" << num_train << endl;
    p_net->Print();
    outfile.close();

}


Matrix* Label_PC(Matrix* x_vec) {
    Matrix* y_vec = new Matrix(1, x_vec->col);
    bool key;
    for(int j=0;j<x_vec->col;++j) {
        key = 0;
        for (int i = 0; i < x_vec->row; ++i) {
            key = key ^ bool(x_vec->p_row[i][j]);
        }
        y_vec->p_row[0][j] = double(key);
    }
    return y_vec;
}


Matrix* Get_PC_data(int num, int dim) {
    srand(time(0));
    Matrix* x_vec = new Matrix(dim, num);
    for(int j=0;j<num;++j){
        for(int i=0;i<dim;++i)
            x_vec->p_row[i][j] = rand()%2;
    }
    return x_vec;
}


int Parity_check() {
    cout << "奇偶校验" << endl;
    int num_train = 100000;
    int dim = 8;
    int num_val = 100;
    double loss = 0;
    Matrix* x_train_all = Get_PC_data(num_train, dim);
    Matrix* y_train_all = Label_PC(x_train_all);
    Matrix* x_val_all = Get_PC_data(num_val, dim);
    Matrix* y_val_all = Label_PC(x_val_all);

    ofstream outfile;
    outfile.open("train_loss2.txt", ios::trunc);

    FullyConnectedNetwork* p_net = new FullyConnectedNetwork(dim, 1, dim, 1, sigmoid, d_sigmoid);
    p_net->InitParam(Norm_random);
    // train
    Train_Net(p_net, x_train_all, y_train_all, &outfile);
    //训练集正确率
    double err_rate = Validate_Net(p_net, x_train_all, y_train_all);
    cout << "训练集正确率" << 1-err_rate << endl;
    //检验集正确率, 误差
    err_rate = Validate_Net(p_net, x_val_all, y_val_all, &loss);
    cout << "检验损失" << loss / num_val << endl;
    cout << "检验集正确率" << 1-err_rate << endl;
    cout << "训练迭代次数" << num_train << endl;
    p_net->Print();
    outfile.close();

}


int main(){
//    XOR();
    Parity_check();
}

