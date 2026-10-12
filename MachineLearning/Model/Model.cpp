#include "Model.h"

double Model::train_step(const Matrix& input, const Matrix& real)
{
    auto result = forward(input);
    auto loss = lossFunction->forward(result, real);
    auto gard = lossFunction->backward();
    backward(gard);
    op->step();
    return loss;
}

Matrix Model::forward(const Matrix& input)
{
    Matrix temp = input;
    for (size_t i = 0; i < layers.size(); i++) {
        temp = layers[i]->forward(temp);
    }
    return temp;
}

// 网络前向，别名，给计时用
Matrix Model::net_forward(const Matrix& input)
{
    return forward(input);
}

// loss前向计算
double Model::loss_forward(const Matrix& net_out, const Matrix& real)
{
    return lossFunction->forward(net_out, real);
}

// loss反向求梯度
Matrix Model::loss_backward()
{
    return lossFunction->backward();
}

// 网络反向传播
void Model::net_backward(const Matrix& gard)
{
    backward(gard);
}

// 优化器一步更新
void Model::opt_step()
{
    op->step();
}

void Model::backward(const Matrix& gard)
{
    Matrix temp = gard;
    for (auto iter = layers.rbegin(); iter != layers.rend(); ++iter) {
        temp = (*iter)->backward(temp);
    }
}

Matrix Model::predict(const Matrix& input)
{
    return forward(input);
}


