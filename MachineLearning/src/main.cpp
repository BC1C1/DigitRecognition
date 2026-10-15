#include "iostream"

#include "SuperParam.h"
#include "Utils.h"
#include "Model.h"

#include "qdebug.h"
#include "qstring.h"

#include "Linear.h"
#include "ReLU.h"
#include "Conv2D.h"
#include "MSELoss.h"
#include "CrossEntropyLoss.h"
#include "Optimizer.h"

#include "DataReader.h"
#include "DataReader_CIFAR.h"
#include "cuda_bridge.h"

#include <chrono>

using namespace cfg;
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>; 
using MsDur = std::chrono::duration<double, std::milli>;
double evaluate(std::shared_ptr<Model> model, const Matrix& images, const Matrix& labels, size_t M, size_t chunk = 1000);
void augmentShift(Matrix& batch, size_t C, size_t H, size_t W, int maxShift);

int main(int argc, char *argv[])
{
    cuda_initAll();
    //DataReader reader;
    //auto images = reader.readPicture(Mtrain, datasetDir, "train");
    //auto labels = reader.readLabel(Mtrain, datasetDir, "train");
    DataReader_CIFAR reader;
    reader.readTrain();
    auto& images = reader.trainData;
    auto& labels = reader.trainLabel;

    //auto loss = std::make_shared<CrossEntropyLoss>();
    //auto opt = std::make_shared<Optimizer>();
    auto model = Model::loadFromJson(cfg::getModel());

    std::mt19937 rng(42);
    std::vector<size_t> indices(Mtrain);
    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), rng);
    std::vector<size_t> validIdx(indices.begin(), indices.begin() + testForTrain);
    size_t* temp;
    cudaMallocAsync_throw(&temp, sizeof(size_t) * validIdx.size());
    cudaMemcpy(temp, validIdx.data(), sizeof(size_t) * validIdx.size(), cudaMemcpyHostToDevice);
    Matrix validImgs = images.sliceRowsByIndex(temp, validIdx.size());
    Matrix validLabels = labels.sliceRowsByIndex(temp, validIdx.size());
    cudaFreeAsync_throw(temp);

    std::vector<size_t> train_idx(indices.begin() + testForTrain, indices.end());

    double maxAcc = 0;
    int patience = 3;
    int noImprove = 0;
    size_t* gidxes;
    cudaMallocAsync_throw(&gidxes, sizeof(size_t) * train_idx.size());
    for (int epoch = 0; epoch < training_times; epoch++)
    {
        std::cout << "epoch: " << epoch << std::endl;
        std::shuffle(train_idx.begin(), train_idx.end(), rng);
        cudaMemcpy(gidxes, train_idx.data(), sizeof(size_t) * train_idx.size(), cudaMemcpyHostToDevice);
        double running_loss = 0.0;
        int cnt = 0;
        size_t trainSize = Mtrain - testForTrain;
        for (size_t offset = 0; offset < trainSize; offset += batchSize)
        {
            size_t cur = std::min(batchSize, trainSize - offset);

            // 1. 数据准备
            Matrix batch_X = images.sliceRowsByIndex(gidxes + offset, cur);
            Matrix batch_T = labels.sliceRowsByIndex(gidxes + offset, cur);
            // 数据增强：平移
            augmentShift(batch_X, 3, 32, 32, 3);

            // 2. 网络forward
            Matrix net_out = model->net_forward(batch_X);

            // 3. loss forward
            double L = model->loss_forward(net_out, batch_T);

            // 4. loss backward + 网络backward
            Matrix grad_loss = model->loss_backward();
            model->net_backward(grad_loss);

            // 5. optimizer更新
            model->opt_step();

            running_loss += L;
            cnt++;
        }
        std::cout << "avg loss: " << running_loss / cnt << std::endl;
        double validAcc = evaluate(model, validImgs, validLabels, testForTrain, batchSize);
        std::cout << "now acc in train: " << evaluate(model, images, labels, trainSize, batchSize) << std::endl;
        std::cout << "now acc in valid: " << validAcc << std::endl;
        if (validAcc > maxAcc) {
            maxAcc = validAcc; 
            noImprove = 0;
        }
        else {
            noImprove++;
            if (noImprove > patience) {
                noImprove = 0;
                model->setLearningRate(model->getLearningRate() / 2);
                std::cout << "now learning rate is: " << model->getLearningRate() << std::endl;
            }
        }
        std::cout << std::endl;
    }
    cudaFreeAsync_throw(gidxes);

    //auto testImages = reader.readPicture(Mtest, datasetDir, "t10k");
    //auto testLebels = reader.readLabel(Mtest, datasetDir, "t10k");
    reader.readTest();
    auto& testImages = reader.testData;
    auto& testLebels = reader.testLabel;
    auto acc = evaluate(model, testImages, testLebels, Mtest, test_chunk);

    std::cout << "final acc is " << acc << std::endl;
}

double evaluate(std::shared_ptr<Model> model, const Matrix& images, const Matrix& labels,
    size_t M, size_t chunk) {
    size_t correct = 0, total = 0;
    for (size_t offset = 0; offset < M; offset += chunk) {
        size_t cur = std::min(chunk, M - offset);
        std::vector<size_t> idx(cur);
        std::iota(idx.begin(), idx.end(), offset);
        size_t* temp;
        cudaMallocAsync_throw(&temp, sizeof(size_t) * idx.size());
        cudaMemcpy(temp, idx.data(), sizeof(size_t) * idx.size(), cudaMemcpyHostToDevice);
        Matrix batch_X = images.sliceRowsByIndex(temp, idx.size());
        Matrix batch_T = labels.sliceRowsByIndex(temp, idx.size());
        Matrix result = model->predict(batch_X);
        auto ret = launchOneHotAndCompare(result.data_gpu(), batch_T.data_gpu(), cur, result.cols());
        for (size_t i = 0; i < cur; i++) {
            total++;
            if (*(ret + i)) correct++;
        }
        delete[] ret;
        cudaFreeAsync_throw(temp);
    }
    return (double)correct / total;
}

// 这个是平移，可以理解成不让模型记住数字在图片的绝对位置，而是学习有数字部分的图像特征
void augmentShift(Matrix& batch, size_t C, size_t H, size_t W, int maxShift)
{
    assert(batch.cols() == C * H * W);
    const size_t N = batch.rows();
    Matrix out = Matrix::zeroMatrix(N, C * H * W);   
    float* random_values;
    cudaMallocAsync_throw(&random_values, sizeof(float) * N * 2);
    launchFillUniform(random_values, N * 2);
    launchShiftImg(batch.data_gpu(), out.data_gpu(), N, C, H, W, random_values, maxShift);
    cudaFreeAsync_throw(random_values);
    batch = out;
}
