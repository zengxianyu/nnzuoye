//
// Created by zeng on 2019/4/3.
//

#include "nn.h"

using namespace std;



Linear::~Linear() {
    if(p_weight != NULL) delete p_weight;
    if(p_bias != NULL) delete p_bias;
    if(p_hidden != NULL) delete p_hidden;
    if(p_input != NULL) delete p_input;
    if(p_g_weight != NULL) delete p_g_weight;
    if(p_g_bias != NULL) delete p_g_bias;
}

Linear::Linear(int num_input, int num_output, double (*act)(double), double (d_act)(double)) {
    n_i = num_input;
    n_o = num_output;
    p_weight = new Matrix(n_o, n_i);
    p_bias = new Matrix(num_output, 1);
    activation_func = act;
    d_activation_func = d_act;

}


Matrix* Linear::Forward(Matrix *_p_input, bool keep_input) {
    /*
     * 输入：这一层的输入
     * 输出：这一层的输出（激活函数后的值）
     * 保存：输入、激活函数之前的
     */
    if(p_input != NULL && !keep_input) delete p_input;
    p_input = _p_input;
    Matrix* p_hidden_wo_bias = p_weight->Dot(*p_input);
    if(p_hidden != NULL) delete p_hidden;
    p_hidden = (*p_hidden_wo_bias) + (*p_bias);
    delete p_hidden_wo_bias;
    Matrix* p_output = p_hidden->Apply(activation_func);
    return p_output;
}


Matrix* Linear::Backward(Matrix* _p_g_output, bool keep_input) {
    /*
     * 输入：目标函数关于输出的梯度
     * 输出：目标函数关于输入的梯度
     * 保存：weight和bias的梯度
     * 删除forward时保存的输入和激活前的值
     */
    Matrix* p_d_phi = p_hidden->Apply(d_activation_func);
    Matrix* p_delta = (*_p_g_output)*(*p_d_phi);
    delete p_d_phi;
    if(p_g_weight != NULL) delete p_g_weight;
    p_g_weight = p_delta->DotT(*p_input);
    if(p_g_bias != NULL) delete p_g_bias;
    p_g_bias = p_delta;
    Matrix* p_grad_input = p_weight->TDot(*p_delta);
    if(p_input != NULL && !keep_input) {
        delete p_input;
        p_input = NULL;
    }
    if(p_hidden != NULL) {
        delete p_hidden;
        p_hidden = NULL;
    }
    return p_grad_input;
}


void Linear::Optimize(double lr) {
    if(p_g_weight == NULL){
        cout << "weight的梯度还没有计算" << endl;
        exit(1);
    }
    else{
        for(int i=0;i<p_weight->row;++i)
            for(int j=0;j<p_weight->col;++j) p_weight->p_row[i][j] -= lr * p_g_weight->p_row[i][j];
        p_g_weight = NULL;
    }
    if(p_g_bias == NULL){
        cout << "bias的梯度还没有计算" << endl;
        exit(1);
    }
    else{
        for(int i=0;i<p_bias->row;++i)
            for(int j=0;j<p_bias->col;++j) p_bias->p_row[i][j] -= lr * p_g_bias->p_row[i][j];
        p_g_bias = NULL;
    }
}


void FullyConnectedNetwork::Print() {

    cout << "NN的weight和bias如下：" << endl;
    cout << "输入层weight" << endl;
    pp_layer[0]->p_weight->Print();
    cout << "输入层bias" << endl;
    pp_layer[0]->p_bias->Print();
    for(int l=1;l<n_layer-1;++l) {
        cout << "隐藏层 "<<l<<" weight" << endl;
        pp_layer[l]->p_weight->Print();
        cout << "隐藏层 "<<l<<" bias" << endl;
        pp_layer[l]->p_bias->Print();
    }
    cout << "输出层weight" << endl;
    pp_layer[n_layer-1]->p_weight->Print();
    cout << "输出层bias" << endl;
    pp_layer[n_layer-1]->p_bias->Print();
}



FullyConnectedNetwork::~FullyConnectedNetwork() {
    for(int l=0;l<n_layer;++l) delete pp_layer[l];
    delete pp_layer;
    cout << "hehe" << endl;
}


FullyConnectedNetwork::FullyConnectedNetwork(int num_input, int num_output, int num_hidden, int num_hidden_layer, double (*act)(double), double (d_act)(double)) {
    n_input = num_input;
    n_output = num_output;
    n_hidden = num_hidden;
    hidden_layer = num_hidden_layer;
    n_layer = hidden_layer + 2;
    pp_layer = new Linear* [hidden_layer+2];
    pp_layer[0] = new Linear(n_input, n_hidden, pass, d_pass);
    for(int i=1;i<hidden_layer+1;++i) {
        pp_layer[i] = new Linear(n_hidden, n_hidden, act, d_act);
    }
    pp_layer[hidden_layer+1] = new Linear(n_hidden, n_output, act, d_act);

}

Matrix* FullyConnectedNetwork::Forward(Matrix *_p_input) {
    Matrix* p_temp_input = new Matrix(*_p_input);
    for(int i=0;i<n_layer;++i)
    {
        p_temp_input = pp_layer[i]->Forward(p_temp_input);
    }
    return p_temp_input;
}

Matrix* FullyConnectedNetwork::Backward(Matrix *_p_g_output) {
    Matrix* p_temp_g_output = new Matrix(*_p_g_output);
    Matrix* new_p;
   for(int i=n_layer-1;i>=0;--i) {
       new_p = pp_layer[i]->Backward(p_temp_g_output);
       delete p_temp_g_output;
       p_temp_g_output = new_p;
   }
    return p_temp_g_output;
}

void FullyConnectedNetwork::Optimize(double lr) {
    for(int l=0;l<n_layer;++l) pp_layer[l]->Optimize(lr);
}

void FullyConnectedNetwork::InitParam(double (*func)()) {
    for(int i=0;i<n_layer;++i) {
        pp_layer[i]->p_weight->Apply_(func);
        pp_layer[i]->p_bias->Apply_(func);

    }
}

int gradient_check_net(FullyConnectedNetwork* p_net, double delta) {
    /*
     * E = \sum_j y_j
     * grad y_j = 1
     */
    Matrix* p_x = new Matrix(p_net->n_input, 1);
    p_x->Apply_(Norm_random);
    Matrix* p_y = p_net->Forward(p_x);
    p_y->Print();

    Matrix* p_g_y = new Matrix(p_net->n_output, 1);
    p_g_y->Fill_(1);
    p_net->Backward(p_g_y);

    double E0 = p_y->Sum();
    Matrix* p_y1;
    double E1;
    for(int l=0;l<p_net->n_layer;++l) {
        cout << "checking layer " << l << endl;
        cout << "weight" << endl;
        for(int i=0;i<p_net->pp_layer[l]->p_weight->row;++i)
            for(int j=0;j<p_net->pp_layer[l]->p_weight->col;++j) {
                p_net->pp_layer[l]->p_weight->p_row[i][j] += delta;
                p_y1 = p_net->Forward(p_x);
                E1 = p_y1->Sum();
                delete p_y1;
                cout << (E1-E0)/delta-p_net->pp_layer[l]->p_g_weight->p_row[i][j] << endl;
                p_net->pp_layer[l]->p_weight->p_row[i][j] -= delta;
            }
        cout << "bias" << endl;
        for(int i=0;i<p_net->pp_layer[l]->p_bias->row;++i)
            for(int j=0;j<p_net->pp_layer[l]->p_bias->col;++j) {
                p_net->pp_layer[l]->p_bias->p_row[i][j] += delta;
                p_y1 = p_net->Forward(p_x);
                E1 = p_y1->Sum();
                delete p_y1;
                cout << (E1-E0)/delta-p_net->pp_layer[l]->p_g_bias->p_row[i][j] << endl;
                p_net->pp_layer[l]->p_bias->p_row[i][j] -= delta;
            }

    }
    delete p_x; delete p_y; delete p_g_y;
    cout << "test Optimize" << endl;
    p_net->Optimize();

}


int gradient_check_linear(Linear* p_fc, double delta) {
    /*
     * E = \sum_j y_j
     * grad y_j = 1
     */

    Matrix* p_x = new Matrix(p_fc->n_i, 1);
    p_x->Apply_(Norm_random);

    Matrix* p_y = p_fc->Forward(p_x, true);

    Matrix* p_g_y = new Matrix(p_fc->n_o, 1);
    p_g_y->Fill_(1);
    p_fc->Backward(p_g_y, true);
//    cout << p_fc->p_g_weight->p_row[0][0] << endl;

    double E0 = p_y->Sum();
    double E1;
    Matrix* p_y1;

    cout << "checking gradient of weights" << endl;
    for(int i=0;i<p_fc->p_weight->row;++i)
        for(int j=0;j<p_fc->p_weight->col;++j) {
            p_fc->p_weight->p_row[i][j] += delta;
            p_y1 = p_fc->Forward(p_x, true);
            E1 = p_y1->Sum();
            delete p_y1;
            cout << (E1-E0)/delta-p_fc->p_g_weight->p_row[i][j] << endl;
            p_fc->p_weight->p_row[i][j] -= delta;

        }
    cout << "checking gradient of bias" << endl;
    for(int i=0;i<p_fc->p_bias->row;++i)
        for(int j=0;j<p_fc->p_bias->col;++j) {
            p_fc->p_bias->p_row[i][j] += delta;
            p_y1 = p_fc->Forward(p_x, true);
            E1 = p_y1->Sum();
            delete p_y1;
            cout << (E1-E0)/delta-p_fc->p_g_bias->p_row[i][j] << endl;
            p_fc->p_bias->p_row[i][j] -= delta;

        }
    delete p_x; delete p_y; delete p_g_y;
}


