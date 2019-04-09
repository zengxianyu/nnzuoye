// matrix.h
// Created by zeng on 2019/3/4.
//

#ifndef UNTITLED1_MATRIX_MULTIPLY_H
#define UNTITLED1_MATRIX_MULTIPLY_H


#define EXPT_MAT_SHAP_INCON 1
#define EXPT_PARAM_ERR 2
#include <iostream>
class Matrix
{
private:
    int _row, _col;
    void Init(int m, int n);
public:
    Matrix();
    Matrix(const Matrix& A);
    Matrix(int m, int n);
    Matrix(int m, int n, double (*function)(void));
    Matrix(int m, int n, double val);
    const int &row = _row;
    const int &col = _col;
    double** p_row = NULL;
    void Print();
    Matrix* operator*(Matrix &B);
    Matrix* operator*(double a);
    Matrix* Dot(Matrix &B);
    Matrix* TDot(Matrix &B);
    Matrix* DotT(Matrix &B);
    Matrix* Transpose();
    Matrix* operator+(Matrix &B);
    Matrix* operator-(Matrix &B);
    Matrix* operator==(Matrix &B);
    Matrix* operator!=(Matrix &B);
    Matrix* Get_col(int c);
    Matrix* Get_row(int r);

    Matrix& operator=(Matrix const &B);
    Matrix* operator[](int i);
    void Threshold_(double th=0, double pos=1, double neg=-1);
    void Apply_(double (*function)(double));
    void Apply_(double (*function)());
    void Fill_(double val);
    void operator+=(double val);
    void Add_(double val);
    Matrix* Apply(double (*function)(double));
    double Sum();
    ~Matrix();
    static Matrix* Cat(Matrix A, Matrix B, int dim);
};

#endif //UNTITLED1_MATRIX_MULTIPLY_H
