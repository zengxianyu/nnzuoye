// main.cpp
// Created by zeng on 2019/3/4.
//
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "hw1.h"

using namespace std;

Matrix* Label(const Matrix& data, int dim=1){
    Matrix* p_label = new Matrix(data.row, 1);
    for(int i=0;i<data.row;++i){
        if(data.p_row[i][dim]>=0) p_label->p_row[i][0] = 1;
        else p_label->p_row[i][0] = -1;
    }
    return p_label;
}

double Error_rate(Matrix* p_data, Matrix* p_label, Matrix* p_weight) {
    Matrix* p_pred = p_data->Dot(*p_weight);
    p_pred->Threshold_();
    // error rate of new weight
    Matrix* p_pred_corr = (*p_pred != *p_label);
    double err = p_pred_corr->Sum() / p_label->row;
    delete p_pred; delete p_pred_corr;
    return err;
}

Matrix* Train(Matrix* p_train_data_one, Matrix* p_train_label, double eta=0.01){
    // initial values
    Matrix* p_weight = new Matrix(p_train_data_one->col, 1);
    Matrix* p_new_weight = new Matrix(p_train_data_one->col, 1);
    Matrix* p_delta = new Matrix(p_train_data_one->col, 1);
    Matrix* p_pred = p_train_data_one->Dot(*p_weight);
    p_pred->Threshold_();
    Matrix* p_pred_corr = (*p_pred != *p_train_label);
    double err = p_pred_corr->Sum() / p_train_label->row;
    cout << "初始错误率 = " << err << endl;

//    cout << "The first training sample:"<< endl;
    Matrix* p_data = (*p_train_data_one)[0];
    Matrix* p_data_T = p_data->Transpose();
//    p_data->Print();

//    cout << "Initial weight: " << endl;
//    p_weight->Print();

    int i = 0;
    int count = 0;

    // train
    while(err > 0)
    {
        if(i >= p_train_data_one->row) i = 0;
        // 只对分类错误的训练样本更新
        if(p_pred->p_row[i][0] == p_train_label->p_row[i][0]) {
            ++i;
            continue;
        }
//        cout << "更新次数" << count << endl;
        count += 1;
        // take a sample and its label
        delete p_data;
        p_data = (*p_train_data_one)[i];
        double label = p_train_label->p_row[i][0];
        ++i;
        delete p_data_T;
        p_data_T = p_data->Transpose();

        // update weight
        delete p_delta;
        p_delta = (*p_data_T)*(eta*label);
        delete p_new_weight;
        p_new_weight = (*p_weight)+(*p_delta);
        (*p_weight) = (*p_new_weight);

        // prediction of new weight
        delete p_pred;
        p_pred = p_train_data_one->Dot(*p_weight);
        p_pred->Threshold_();
        // error rate of new weight
        delete p_pred_corr;
        p_pred_corr = (*p_pred != *p_train_label);
        err = p_pred_corr->Sum() / p_train_label->row;
//        cout << "training error rate = " << err<< endl;
    }
    cout << "更新次数:" << count << endl;
    cout << "DONE" << endl;
    delete p_new_weight;
    delete p_pred;
    delete p_pred_corr;
    delete p_delta;
    delete p_data;
    delete p_data_T;

    return p_weight;
}

int _main() {
//    srand(time(NULL));

    int dim = 5;
    int num_train_1 = 10; // training samples
    int num_train_2 = 20; // training samples
    int num_train_3 = 30; // training samples
    int num_val = 30; // validation samples
    double eta = 0.001; // learning rate

    // training data and labels
    Matrix* p_one = new Matrix(num_val, 1, 1);
    Matrix* p_val_data = new Matrix(num_val, dim, Norm_random);
    Matrix* p_val_data_one = Matrix::Cat(*p_val_data, *p_one, 1);
    delete p_one;
    Matrix* p_val_label = Label(*p_val_data);

    p_one = new Matrix(num_train_1, 1, 1);
    Matrix* p_train_data_1 = new Matrix(num_train_1, dim, Norm_random);
    Matrix* p_train_data_one_1 = Matrix::Cat(*p_train_data_1, *p_one, 1);
    Matrix* p_train_label_1 = Label(*p_train_data_1);

    p_one = new Matrix(num_train_2, 1, 1);
    Matrix* p_train_data_2 = new Matrix(num_train_2, dim, Norm_random);
    Matrix* p_train_data_one_2 = Matrix::Cat(*p_train_data_2, *p_one, 1);
    Matrix* p_train_label_2 = Label(*p_train_data_2);

    p_one = new Matrix(num_train_3, 1, 1);
    Matrix* p_train_data_3 = new Matrix(num_train_3, dim, Norm_random);
    Matrix* p_train_data_one_3 = Matrix::Cat(*p_train_data_3, *p_one, 1);
    Matrix* p_train_label_3 = Label(*p_train_data_3);

    cout << "======== 第一组训练数据 ==========" << endl;
    Matrix* p_weight_1 = Train(p_train_data_one_1, p_train_label_1, eta);
    cout << "收敛时的权矢量值 W = " << endl;
    p_weight_1->Print();
    cout << "检验集上的正确率 R = " << 1-Error_rate(p_val_data_one, p_val_label, p_weight_1) << endl;

    cout << "======== 第2组训练数据 ==========" << endl;
    Matrix* p_weight_2 = Train(p_train_data_one_2, p_train_label_2, eta);
    cout << "收敛时的权矢量值 W = " << endl;
    p_weight_2->Print();
    cout << "检验集上的正确率 R = " << 1-Error_rate(p_val_data_one, p_val_label, p_weight_2) << endl;

    cout << "======== 第3组训练数据 ==========" << endl;
    Matrix* p_weight_3 = Train(p_train_data_one_3, p_train_label_3, eta);
    cout << "收敛时的权矢量值 W = " << endl;
    p_weight_3->Print();
    cout << "检验集上的正确率 R = " << 1-Error_rate(p_val_data_one, p_val_label, p_weight_3) << endl;

    return 0;
}