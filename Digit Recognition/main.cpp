#include "MainWindow.h"
#include <QtWidgets/QApplication>

#include "SuperParam.h"
#include "Utils.h"
#include "Model.h"

#include "qdebug.h"
#include "qstring.h"

#include "Linear.h"
#include "ReLU.h"
#include "MSELoss.h"
#include "CrossEntropyLoss.h"
#include "Optimizer.h"

#include "DataReader.h"

#include <chrono>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>; 
using MsDur = std::chrono::duration<double, std::milli>;

void printMatrix(const Matrix& m) {
    for (size_t i = 0; i < m.rows(); i++) {
        QString line = "";
        for (size_t j = 0; j < m.cols(); j++)
            line.append(QString::number(m(i, j)) + " ");
        qDebug() << line;
    }
    qDebug() << "\n";
}

using LayerPointer = std::shared_ptr<Layer>;
LayerPointer linear(size_t in, size_t out) {
    return std::make_shared<Linear>(in, out);
}

LayerPointer relu() {
    return std::make_shared<ReLU>();
}

int main(int argc, char *argv[])
{
    size_t M = 60000;
    DataReader reader;
    auto testImages = reader.readPicture(M, "D:\\code\\c\\Digit Recognition\\t10k-images.idx3-ubyte");
    auto testLebels = reader.readLabel(M, "D:\\code\\c\\Digit Recognition\\t10k-labels.idx1-ubyte");
    auto images = reader.readPicture(M, "D:\\code\\c\\Digit Recognition\\train-images.idx3-ubyte");
    auto labels = reader.readLabel(M, "D:\\code\\c\\Digit Recognition\\train-labels.idx1-ubyte");

    auto loss = std::make_shared<CrossEntropyLoss>();
    auto opt = std::make_shared<Optimizer>();
    Model model;
    model.setOptimizer(opt);
    model.setLossFunction(loss);
    model.addLayer(linear(784, 512));
    model.addLayer(relu());
    model.addLayer(linear(512, 128));
    model.addLayer(relu());
    model.addLayer(linear(128, 10));

    std::mt19937 rng(42);
    std::vector<size_t> indices(M);
    std::iota(indices.begin(), indices.end(), 0);
    size_t batchSize = 64;
    int times = 3;

    // 全局累计，单位ms
    double sum_data = 0.0;
    double sum_netfw = 0.0;
    double sum_lossfw = 0.0;
    double sum_netbw = 0.0;
    double sum_opt = 0.0;
    int    batch_count = 0;

    for (int epoch = 0; epoch < times; epoch++)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        double running_loss = 0.0;
        int cnt = 0;

        for (size_t offset = 0; offset < M; offset += batchSize)
        {
            size_t cur = std::min(batchSize, M - offset);

            // 1. 数据准备
            auto t0 = Clock::now();
            std::vector<size_t> batch_idx(indices.begin() + offset, indices.begin() + offset + cur);
            Matrix batch_X = images.sliceRowsByIndex(batch_idx);
            Matrix batch_T = labels.sliceRowsByIndex(batch_idx);
            auto t1 = Clock::now();

            // 2. 网络forward
            auto t2 = Clock::now();
            Matrix net_out = model.net_forward(batch_X);
            auto t3 = Clock::now();

            // 3. loss forward
            auto t4 = Clock::now();
            double L = model.loss_forward(net_out, batch_T);
            auto t5 = Clock::now();

            // 4. loss backward + 网络backward
            auto t6 = Clock::now();
            Matrix grad_loss = model.loss_backward();
            model.net_backward(grad_loss);
            auto t7 = Clock::now();

            // 5. optimizer更新
            auto t8 = Clock::now();
            model.opt_step();
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
        qDebug() << "epoch" << epoch << "avg loss:" << running_loss / cnt;
    }

    qDebug() << "\n===== PER‑BATCH AVERAGE (ms) =====";
    qDebug() << "data_prep   :" << sum_data / batch_count;
    qDebug() << "net_forward :" << sum_netfw / batch_count;
    qDebug() << "loss_forward:" << sum_lossfw / batch_count;
    qDebug() << "net_backward:" << sum_netbw / batch_count;
    qDebug() << "optimizer   :" << sum_opt / batch_count;

    // 验证
    M = 10000;
    auto result = model.predict(testImages);
    size_t correct_cnt = 0;
    for (size_t i = 0; i < M; i++) {
        size_t pred = 0;
        double cache = result(i, 0);
        for (size_t j = 0; j < 10; j++) {
            if (result(i, j) > cache) {
                pred = j;
                cache = result(i, j);
            }
        }
        size_t real = 0;
        for (size_t j = 0; j < 10; j++) {
            if (testLebels(i, j) == 1) {
                real = j;
                break;
            }
        }
        if (pred == real) correct_cnt++;
    }
    qDebug() << "acc is" << (double)correct_cnt / M;

}
