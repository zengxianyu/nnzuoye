// matrix.cpp
// Created by zeng on 2019/3/4.
//
#include "matrix.h"
//#include "normal_random.h"
using namespace std;


void Matrix::Init(int m, int n)
{
    _row = m;
    _col = n;
    p_row = new double*[m];
    for(int i=0;i<m;++i) p_row[i] = new double[n];
}

Matrix::Matrix(const Matrix& A) {
    _row = A.row;
    _col = A.col;
    Init(_row, _col);
    for(int i=0;i<_row;++i)
        for(int j=0;j<_col;++j) p_row[i][j] = A.p_row[i][j];
}

Matrix::Matrix()
{
    p_row = NULL;
    _row = 0;
    _col = 0;
}


Matrix::Matrix(int m, int n)
{
    Init(m, n);
}

Matrix::Matrix(int m, int n, double (*function)()) {
    Init(m, n);
    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j){
            p_row[i][j] = (*function)();
        }
}

Matrix::Matrix(int m, int n, double val) {
    Init(m, n);
    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j){
            p_row[i][j] = val;
        }
}

Matrix::~Matrix()
{
    if(p_row != NULL) {
        for (int i = 0; i < row; ++i) delete[] p_row[i];
        delete[] p_row;
    }
    p_row = NULL;
}

void Matrix::Print()
{
    for(int i=0;i<row;++i) {
        for (int j = 0; j < col; ++j) {
            cout << p_row[i][j] << ' ';
        }
        cout << endl;
    }
}

Matrix* Matrix::Dot(Matrix &B){
    if(this->col != B.row)
    {
        cout <<'('<<this->row<<'x'<<this->col<<')'<<'('<<B.row<<'x'<<B.col<<')'<<"矩阵性状不符合乘法的要求" << endl;
        throw EXPT_MAT_SHAP_INCON;
    }
    Matrix* p_res = new Matrix(this->row, B.col);
    for(int i=0;i<p_res->row;++i)
        for(int j=0;j<p_res->col;++j){
            double sum = 0;
            for(int k=0;k<this->col;++k) sum += this->p_row[i][k]*B.p_row[k][j];
            p_res->p_row[i][j] = sum;
        }
    return p_res;
}

Matrix* Matrix::DotT(Matrix &B){
    if(this->col != B.col)
    {
        cout <<'('<<this->row<<'x'<<this->col<<')'<<'('<<B.row<<'x'<<B.col<<')'<<"矩阵性状不符合乘法转置的要求" << endl;
        throw EXPT_MAT_SHAP_INCON;
    }
    Matrix* p_res = new Matrix(this->row, B.row);
    for(int i=0;i<p_res->row;++i)
        for(int j=0;j<p_res->col;++j){
            double sum = 0;
            for(int k=0;k<this->col;++k) sum += this->p_row[i][k]*B.p_row[j][k];
            p_res->p_row[i][j] = sum;
        }
    return p_res;
}


Matrix* Matrix::TDot(Matrix &B){
    if(this->row != B.row)
    {
        cout <<'('<<this->row<<'x'<<this->col<<')'<<'('<<B.row<<'x'<<B.col<<')'<<"矩阵性状不符合转置乘法的要求" << endl;
        throw EXPT_MAT_SHAP_INCON;
    }
    Matrix* p_res = new Matrix(this->col, B.col);
    for(int i=0;i<p_res->row;++i)
        for(int j=0;j<p_res->col;++j){
            double sum = 0;
            for(int k=0;k<this->row;++k) sum += this->p_row[k][i]*B.p_row[k][j];
            p_res->p_row[i][j] = sum;
        }
    return p_res;
}

Matrix* Matrix::Transpose(){
    Matrix* p_res = new Matrix(this->col, this->row);
    for(int i=0;i<this->row;++i)
        for(int j=0;j<this->col;++j){
            p_res->p_row[j][i] = p_row[i][j];
        }
    return p_res;
}

Matrix* Matrix::operator*(Matrix &B){
    if(this->col != B.col || this->row != B.row)
    {
        cout<<'('<<this->row<<'x'<<this->col<<')'<<'('<<B.row<<'x'<<B.col<<')' << "矩阵性状不符合元素相乘的要求" << endl;
        throw EXPT_MAT_SHAP_INCON;
    }
    Matrix* p_res = new Matrix(this->row, B.col);
    for(int i=0;i<p_res->row;++i)
        for(int j=0;j<p_res->col;++j){
            p_res->p_row[i][j] = this->p_row[i][j] * B.p_row[i][j];
        }
    return p_res;
}


void Matrix::Apply_(double (*function)(double)) {
    for(int i=0;i<row;++i)
        for(int j=0;j<col;++j) {
            p_row[i][j] = (*function)(p_row[i][j]);
        }
}

void Matrix::Fill_(double val) {
    for(int i=0;i<row;++i)
        for(int j=0;j<col;++j) {
            p_row[i][j] = val;
        }
}

void Matrix::Apply_(double (*function)()) {
    for(int i=0;i<row;++i)
        for(int j=0;j<col;++j) {
            p_row[i][j] = (*function)();
        }
}


Matrix* Matrix::Apply(double (*function)(double)) {
    Matrix* p_res = new Matrix(row, col);
    for(int i=0;i<row;++i)
        for(int j=0;j<col;++j) {
            p_res->p_row[i][j] = (*function)(p_row[i][j]);
        }
    return p_res;
}



Matrix* Matrix::operator*(double a){
    Matrix* p_res = new Matrix(this->row, this->col);
    for(int i=0;i<p_res->row;++i)
        for(int j=0;j<p_res->col;++j){
            p_res->p_row[i][j] = this->p_row[i][j] * a;
        }
    return p_res;
}


void Matrix::operator+=(double a){
    for(int i=0;i<row;++i)
        for(int j=0;j<col;++j){
            p_row[i][j] += a;
        }
}


void Matrix::Add_(double a){
    for(int i=0;i<row;++i)
        for(int j=0;j<col;++j){
            p_row[i][j] += a;
        }
}


Matrix* Matrix::operator+(Matrix &B){
    if(this->col != B.col || this->row != B.row)
    {
        cout<<'('<<this->row<<'x'<<this->col<<')'<<'('<<B.row<<'x'<<B.col<<')' << "矩阵性状不符合加法的要求" << endl;
        throw EXPT_MAT_SHAP_INCON;
    }
    Matrix* p_res = new Matrix(this->row, B.col);
    for(int i=0;i<p_res->row;++i)
        for(int j=0;j<p_res->col;++j){
            p_res->p_row[i][j] = this->p_row[i][j] + B.p_row[i][j];
        }
    return p_res;
}

Matrix* Matrix::operator-(Matrix &B){
    if(this->col != B.col || this->row != B.row)
    {
        cout<<'('<<this->row<<'x'<<this->col<<')'<<'('<<B.row<<'x'<<B.col<<')' << "矩阵性状不符合减法的要求" << endl;
        throw EXPT_MAT_SHAP_INCON;
    }
    Matrix* p_res = new Matrix(this->row, B.col);
    for(int i=0;i<p_res->row;++i)
        for(int j=0;j<p_res->col;++j){
            p_res->p_row[i][j] = this->p_row[i][j] - B.p_row[i][j];
        }
    return p_res;
}

Matrix* Matrix::operator==(Matrix &B){
    if(this->col != B.col || this->row != B.row)
    {
        cout<<'('<<this->row<<'x'<<this->col<<')'<<'('<<B.row<<'x'<<B.col<<')' << "矩阵性状不符合相等的要求" << endl;
        throw EXPT_MAT_SHAP_INCON;
    }
    Matrix* p_res = new Matrix(this->row, B.col);
    for(int i=0;i<p_res->row;++i)
        for(int j=0;j<p_res->col;++j){
            if(p_row[i][j]==B.p_row[i][j]) p_res->p_row[i][j] = 1;
            else p_res->p_row[i][j] = 0;
        }
    return p_res;
}

Matrix* Matrix::operator!=(Matrix &B){
    if(this->col != B.col || this->row != B.row)
    {
        cout<<'('<<this->row<<'x'<<this->col<<')'<<'('<<B.row<<'x'<<B.col<<')' << "矩阵性状不符合相等的要求" << endl;
        throw EXPT_MAT_SHAP_INCON;
    }
    Matrix* p_res = new Matrix(this->row, B.col);
    for(int i=0;i<p_res->row;++i)
        for(int j=0;j<p_res->col;++j){
            if(p_row[i][j]!=B.p_row[i][j]) p_res->p_row[i][j] = 1;
            else p_res->p_row[i][j] = 0;
        }
    return p_res;
}

Matrix& Matrix::operator=(Matrix const &B){
    if(this->p_row!=NULL){
        for (int i = 0; i < row; ++i) delete[] p_row[i];
        delete[] p_row;
    }
    this->_col = B.col;
    this->_row = B.row;
    Init(this->row, this->col);
    for(int i=0;i<B.row;++i)
        for(int j=0;j<B.col;++j) this->p_row[i][j] = B.p_row[i][j];
    return *this;
}

Matrix* Matrix::operator[](int i){
    if(p_row==NULL){
        cout << "(0x0)矩阵" << endl;
        throw EXPT_MAT_SHAP_INCON;
    }
    Matrix* p_res = new Matrix(1, col);
    for(int j=0;j<col;++j) p_res->p_row[0][j] = p_row[i][j];
    return p_res;
}

void Matrix::Threshold_(double th, double pos, double neg){
    for(int i=0;i<row;++i)
        for(int j=0;j<col;++j){
            if(p_row[i][j]>=th) p_row[i][j] = pos;
            else p_row[i][j] = neg;
        }
}

double Matrix::Sum(){
    double sum = 0;
    for(int i=0;i<row;++i)
        for(int j=0;j<col;++j) sum += p_row[i][j];
    return sum;
}

Matrix* Matrix::Cat(Matrix A, Matrix B, int dim){
    if(dim !=0 && dim != 1){
        cout << "dim应为0或1" << endl;
        throw EXPT_PARAM_ERR;
    }
    else if(dim == 0){
        if(A.col != B.col){
            cout <<'('<<A.row<<'x'<<A.col<<')'<<'('<<B.row<<'x'<<B.col<<')'<< "矩阵性状不符合串联的要求" << endl;
            throw EXPT_MAT_SHAP_INCON;
        }
        Matrix* p_res = new Matrix(A.row+B.row, A.col);
        for(int i=0;i<A.row;++i)
            for(int j=0;j<A.col;++j) p_res->p_row[i][j] = A.p_row[i][j];
        for(int i=A.row;i<A.row+B.row;++i)
            for(int j=0;j<A.col;++j) p_res->p_row[i][j] = B.p_row[i-A.row][j];
        return p_res;

    }
    else {
        if(A.row != B.row){
            cout <<'('<<A.row<<'x'<<A.col<<')'<<'('<<B.row<<'x'<<B.col<<')'<< "矩阵性状不符合串联的要求" << endl;
            throw EXPT_MAT_SHAP_INCON;
        }
        Matrix* p_res = new Matrix(A.row, A.col+B.col);
        for(int i=0;i<A.row;++i) {
            for (int j = 0; j < A.col; ++j) p_res->p_row[i][j] = A.p_row[i][j];
            for (int j = A.col; j < A.col+B.col; ++j) p_res->p_row[i][j] = B.p_row[i][j-A.col];
        }
        return p_res;
    }
}

//int main()
//{
//    Matrix sb(2, 2, 3);
//    cout << "sb = " << sb.row << ", "<< sb.col << endl;
//    sb.Print();
//    Matrix hehe = sb*sb;
//    cout << "hehe = sb*sb = " << endl;
//    hehe.Print();
//    hehe[0][1] = 1;
//    cout << "modify hehe" << endl;
//    hehe.Print();
//    cout << "sb-hehe = " << endl;
//    Matrix bb = sb-hehe;
//    bb.Print();
//    cout << "sb+hehe=" << endl;
//    bb = sb+hehe;
//    bb.Print();
//    Matrix v(2, 1, 1);
//    cout << "v=" << endl;
//    v.Print();
//    cout << "(sb+hehe)*v=" << endl;
//    bb = bb*v;
//    bb.Print();
//    Matrix rr(2, 3, norm_random);
//    rr.Print();
//}

