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

#include <chrono>

using namespace cfg;
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>; 
using MsDur = std::chrono::duration<double, std::milli>;
double evaluate(std::shared_ptr<Model> model, const Matrix& images, const Matrix& labels, size_t M, size_t chunk = 1000);
void augmentShift(Matrix& batch, size_t H, size_t W, int maxShift, std::mt19937& rng);
void printMatrix(const Matrix& m) {
    for (size_t i = 0; i < m.rows(); i++) {
        QString line = "";
        for (size_t j = 0; j < m.cols(); j++)
            line.append(QString::number(m(i, j)) + " ");
        std::cout << line.toStdString();
    }
    std::cout << "\n";
}

int main(int argc, char *argv[])
{
    DataReader reader;
    auto images = reader.readPicture(Mtrain, train_pic_route);
    auto labels = reader.readLabel(Mtrain, train_lab_route);

    auto loss = std::make_shared<CrossEntropyLoss>();
    auto opt = std::make_shared<Optimizer>();
    auto model = Model::loadFromJson(cfg::getModel());

    std::mt19937 rng(42);
    std::vector<size_t> indices(Mtrain);
    std::iota(indices.begin(), indices.end(), 0);

    // 全局累计，单位ms
    double sum_data = 0.0;
    double sum_netfw = 0.0;
    double sum_lossfw = 0.0;
    double sum_netbw = 0.0;
    double sum_opt = 0.0;
    int    batch_count = 0;

    std::shuffle(indices.begin(), indices.end(), rng);
    std::vector<size_t> validIdx(indices.begin(), indices.begin() + testForTrain);
    Matrix validImgs = images.sliceRowsByIndex(validIdx);
    Matrix validLabels = labels.sliceRowsByIndex(validIdx);
    std::vector<size_t> train_idx(indices.begin() + testForTrain, indices.end());

    double maxAcc = 0;
    int patience = 3;
    int noImprove = 0;
    for (int epoch = 0; epoch < training_times; epoch++)
    {
        std::cout << "epoch: " << epoch << std::endl;
        std::shuffle(train_idx.begin(), train_idx.end(), rng);
        double running_loss = 0.0;
        int cnt = 0;
        size_t trainSize = Mtrain - testForTrain;
        for (size_t offset = 0; offset < trainSize; offset += batchSize)
        {
            size_t cur = std::min(batchSize, trainSize - offset);

            // 1. 数据准备
            auto t0 = Clock::now();
            std::vector<size_t> batch_idx(train_idx.begin() + offset, train_idx.begin() + offset + cur);
            Matrix batch_X = images.sliceRowsByIndex(batch_idx);
            Matrix batch_T = labels.sliceRowsByIndex(batch_idx);
            // 数据增强：平移
            augmentShift(batch_X, H, W, 3, rng);
            auto t1 = Clock::now();

            // 2. 网络forward
            auto t2 = Clock::now();
            Matrix net_out = model->net_forward(batch_X);
            auto t3 = Clock::now();

            // 3. loss forward
            auto t4 = Clock::now();
            double L = model->loss_forward(net_out, batch_T);
            auto t5 = Clock::now();

            // 4. loss backward + 网络backward
            auto t6 = Clock::now();
            Matrix grad_loss = model->loss_backward();
            model->net_backward(grad_loss);
            auto t7 = Clock::now();

            // 5. optimizer更新
            auto t8 = Clock::now();
            model->opt_step();
            auto t9 = Clock::now();

            // 转毫秒
            double t_data = MsDur(t1 - t0).count();
            double t_netfw = MsDur(t3 - t2).count();
            double t_lossfw = MsDur(t5 - t4).count();
            double t_netbw = MsDur(t7 - t6).count();
            double t_opt = MsDur(t9 - t8).count();

            sum_data += t_data;
            sum_netfw += t_netfw;
            sum_lossfw += t_lossfw;
            sum_netbw += t_netbw;
            sum_opt += t_opt;
            batch_count++;

            running_loss += L;
            cnt++;
        }
        double validAcc = evaluate(model, validImgs, validLabels, testForTrain);
        std::cout << "avg loss: " << running_loss / cnt << std::endl;
        std::cout << "now acc in train: " << evaluate(model, images, labels, trainSize) << std::endl;
        std::cout << "now acc in valid: " << validAcc << std::endl;
        if (validAcc > maxAcc)
            maxAcc = validAcc;
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

    std::cout << "\n===== PER‑BATCH AVERAGE (ms) =====" << std::endl;
    std::cout << "data_prep   :" << sum_data / batch_count << std::endl;
    std::cout << "net_forward :" << sum_netfw / batch_count << std::endl;
    std::cout << "loss_forward:" << sum_lossfw / batch_count << std::endl;
    std::cout << "net_backward:" << sum_netbw / batch_count << std::endl;
    std::cout << "optimizer   :" << sum_opt / batch_count << std::endl;

    // 验证
    auto testImages = reader.readPicture(Mtest, test_pic_route);
    auto testLebels = reader.readLabel(Mtest, test_lab_route);
    size_t correct_cnt = 0;
    size_t total = 0;
    for (size_t offset = 0; offset < Mtest; offset += test_chunk) {
        size_t cur = std::min(test_chunk, Mtest - offset);
        std::vector<size_t> idx(cur);
        std::iota(idx.begin(), idx.end(), offset);
        Matrix batch_X = testImages.sliceRowsByIndex(idx);
        Matrix batch_T = testLebels.sliceRowsByIndex(idx);
        Matrix result = model->predict(batch_X);
        for (size_t i = 0; i < cur; i++) {
            size_t pred = 0;
            double cache = result(i, 0);
            for (size_t j = 1; j < 10; j++) { 
                if (result(i, j) > cache) {
                    pred = j;
                    cache = result(i, j);
                }
            }
            size_t real = 0;
            for (size_t j = 0; j < 10; j++) {
                if (batch_T(i, j) == 1) {
                    real = j;
                    break;
                }
            }
            if (pred == real) correct_cnt++;
            total++;
        }
        //std::cout << "processed" << total << "/" << Mtest
        //    << "acc so far:" << (double)correct_cnt / total << std::endl;
    }

    std::cout << "final acc is" << (double)correct_cnt / total << std::endl;
}

double evaluate(std::shared_ptr<Model> model, const Matrix& images, const Matrix& labels,
    size_t M, size_t chunk) {
    size_t correct = 0, total = 0;
    for (size_t offset = 0; offset < M; offset += chunk) {
        size_t cur = std::min(chunk, M - offset);
        std::vector<size_t> idx(cur);
        std::iota(idx.begin(), idx.end(), offset);

        Matrix batch_X = images.sliceRowsByIndex(idx);
        Matrix batch_T = labels.sliceRowsByIndex(idx);
        Matrix result = model->predict(batch_X);

        for (size_t i = 0; i < cur; i++) {
            size_t pred = 0;
            double best = result(i, 0);
            for (size_t j = 1; j < 10; j++)
                if (result(i, j) > best) { pred = j; best = result(i, j); }

            size_t real = 0;
            for (size_t j = 0; j < 10; j++)
                if (batch_T(i, j) == 1) { real = j; break; }

            if (pred == real) correct++;
            total++;
        }
    }
    return (double)correct / total;
}

// 这个是平移，可以理解成不让模型记住数字在图片的绝对位置，而是学习有数字部分的图像特征
void augmentShift(Matrix& batch, size_t H, size_t W, int maxShift, std::mt19937& rng)
{
    std::uniform_int_distribution<int> d(-maxShift, maxShift);
    const size_t N = batch.rows();
    Matrix out = Matrix::zeroMatrix(N, H * W);   // 平移空出来的位置就是背景 0
    for (size_t n = 0; n < N; ++n) {
        const int dx = d(rng), dy = d(rng);
        for (int y = 0; y < (int)H; ++y) {
            const int sy = y - dy;
            if (sy < 0 || sy >= (int)H) continue;
            for (int x = 0; x < (int)W; ++x) {
                const int sx = x - dx;
                if (sx < 0 || sx >= (int)W) continue;
                out(n, (size_t)y * W + (size_t)x) = batch(n, (size_t)sy * W + (size_t)sx);
            }
        }
    }
    batch = out;   // 不能原地做：平移会覆盖还没读的像素
}
