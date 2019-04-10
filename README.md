# nnzuoye
神经网络课作业

## nn forward

$y^{(l)} = \phi(s^{(l)})$

$s^{(l)} = W^{(l)} y^{(l-1)} + \theta^{(l)}$

## nn backward
输入损失函数关于这层forwad输出的梯度$\nabla_{y^{(l-1)}}$，保存这层参数的梯度，输出关于这层forwad输入的梯度$\nabla_{y^{(l-2)}}$，用作下一层backward的输入
$\nabla_{s^{l-1}} =\nabla_{y^{(l-1)}}\odot\phi'(s^{(l-1)})$，记为$\delta^{(l-1)}$ --所以forward时要保存这一层的激活前值$s^{(l-1)}$

$\nabla_{W^{(l-1)}} = \delta^{(l-1)}y^{(l-2)T}$--所以forward时要保存这一层的输入$y^{(l-2)}$

$\nabla_{\theta^{(l-1)}}=\delta^{(l-1)}$

输出：$\nabla_{y^{(l-2)}}=W^{(l-1)T}\delta^{(l-1)}$
