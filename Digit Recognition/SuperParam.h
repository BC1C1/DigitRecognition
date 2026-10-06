#pragma once

#include <qobject.h>
#include <qjsondocument.h>
#include <qjsonobject.h>
namespace cfg {
    constexpr quint32 magicSeed = 42;
    constexpr double M_PI = 3.1415926;

    constexpr size_t Mtrain = 60000;
    constexpr size_t Mtest = 10000;
    constexpr size_t test_chunk = 1000;

    constexpr double learning_rate = 0.015;
    constexpr float momentum = 0.9;
    constexpr size_t batchSize = 64;
    constexpr int training_times = 30;

    constexpr size_t H = 28;
    constexpr size_t W = 28;

    constexpr size_t testForTrain = 5000;

    inline const QJsonObject& getModel() {
        static QJsonObject obj = QJsonDocument::fromJson(R"({
        "loss": "CrossEntropy",
        "optimizer": "SGD",
        "layers": [
            { "type": "conv2d", "params": [
                { "inputdim":  1 }, { "outputdim": 8 },
                { "ksize": 3 }, { "stride": 1 }, { "padding": 1 }
            ]},
            { "type": "relu", "params": [] },
            { "type": "maxpool2d", "params": [
                { "channel": 8 }, { "ksize": 2 }, { "stride": 2 }
            ]},
            { "type": "linear", "params": [ 1568, 128 ] },
            { "type": "relu", "params": [] },
            { "type": "linear", "params": [ 128, 10 ] }
        ]
    })").object();
        return obj;
    }

    constexpr const char* train_pic_route = "D:\\code\\c\\Digit Recognition\\train-images.idx3-ubyte";
    constexpr const char* train_lab_route = "D:\\code\\c\\Digit Recognition\\train-labels.idx1-ubyte";
    constexpr const char* test_pic_route = "D:\\code\\c\\Digit Recognition\\t10k-images.idx3-ubyte";
    constexpr const char* test_lab_route = "D:\\code\\c\\Digit Recognition\\t10k-labels.idx1-ubyte";
}
