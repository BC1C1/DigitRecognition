#pragma once

#include <qobject.h>
#include <qjsondocument.h>
#include <qjsonobject.h>
#include <qhash.h>
namespace cfg {
    constexpr quint32 magicSeed = 42;
    constexpr float M_PI = 3.1415926;

    constexpr size_t Mtrain = 50000;
    constexpr size_t Mtest = 10000;
    constexpr size_t test_chunk = 1000;

    constexpr float learning_rate = 0.001;
    constexpr float momentum = 0.9;
    constexpr float b2 = 0.999;
    constexpr float epsilon = 1e-8;
    constexpr size_t batchSize = 64;
    constexpr int training_times = 1;

    constexpr size_t testForTrain = 5000;

 /*
 * Model JSON 格式说明
 *
 * 顶层字段:
 *   loss:      "MSE" | "CrossEntropy"
 *   optimizer: "Adam"
 *   layers:    层数组
 *
 * 每个层的格式:
 *   { "type": <层类型>, "params": <参数对象> }
 *
 * 层类型与参数:
 *   conv2d:
 *     outputdim (必填): 输出通道数
 *     ksize     (必填): 卷积核大小
 *     stride    (必填): 步长
 *     padding   (必填): 填充
 *     inputdim  (秩扩张时必填): 输入通道数
 *     inputH    (秩扩张时必填): 输入高度
 *     inputW    (秩扩张时必填): 输入宽度
 *
 *   maxpool2d:
 *     ksize  (必填)
 *     stride (必填)
 *     channel (秩扩张时必填): 输入通道数
 *     inputH  (秩扩张时必填)
 *     inputW  (秩扩张时必填)
 *
 *   linear:
 *     outputdim (必填)
 *     inputdim  (秩扩张时必填; 否则可选, 仅做校验)
 *
 *   relu:
 *     params 必须为空对象 {}
 *
 * 秩推导规则:
 *   - 每层声明自己的 layerRank (conv2d/maxpool2d = 3, linear = 1)
 *   - 当前秩 < layerRank: 必须显式提供新增维度
 *   - 当前秩 >= layerRank: 自动推导, JSON 中若提供则仅做校验
 */

    inline const QJsonObject& getModel() {
        static QJsonObject obj = QJsonDocument::fromJson(R"({
    "loss": "CrossEntropy",
    "optimizer": "Adam",
    "layers": [
        { "type": "conv2d", "params": {
            "inputdim": 3, "inputH": 32, "inputW": 32,
            "outputdim": 32, "ksize": 3, "stride": 1, "padding": 1
        }},
        { "type": "relu", "params": {} },
        { "type": "maxpool2d", "params": { "ksize": 2, "stride": 2 }},
        
        { "type": "conv2d", "params": {
            "outputdim": 64, "ksize": 3, "stride": 1, "padding": 1
        }},
        { "type": "relu", "params": {} },
        { "type": "maxpool2d", "params": { "ksize": 2, "stride": 2 }},
        
        { "type": "conv2d", "params": {
            "outputdim": 64, "ksize": 3, "stride": 1, "padding": 1
        }},
        { "type": "relu", "params": {} },
        { "type": "maxpool2d", "params": { "ksize": 2, "stride": 2 }},
        
        { "type": "linear", "params": { "outputdim": 128 }},
        { "type": "relu", "params": {} },
        { "type": "linear", "params": { "outputdim": 10 }}
    ]
})").object();
        return obj;
    }

    // 数据集目录。用绝对路径 —— 相对路径会跟着**工作目录**走，而这个程序是从
    // build\MachineLearning\Release\ 启动的，相对路径必然找不到文件（而且找不到
    // 时不报错：readPicture 返回空矩阵，训练"跑完"并给出 100% 准确率）。
    // 四个 .idx 文件的**文件名**由 DataReader 自己拼（MNIST 用点号
    // train-images.idx3-ubyte，fashion 用连字符 train-images-idx3-ubyte）。
    // 换数据集只改这一行。
    // 可选项：D:/code/c/MachineLearning_cmake/MNIST 或 .../fashion
    constexpr const char* datasetDir = "D:/code/c/MachineLearning_cmake/fashion";
    const static QString cifarDatasetDir = "CIFAR-10-100";
    const static QString layersNameString = "layers";
    const static QString lossNameString = "loss";
    const static QString optimizerNameString = "optimizer";
    const static QString typeNameString = "type";
    const static QString paramsNameString = "params";
    const static QString linearNameString = "linear";
    const static QString reluNameString = "relu";
    const static QString conv2dNameString = "conv2d";
    const static QString maxpool2dNameString = "maxpool2d";
    const static QString MSELossNameString = "MSE";
    const static QString crossEntropyLossNameString = "CrossEntropy";
    const static QString SGDOptimizerNameString = "SGD";
    const static QString AdamOptimizerNameString = "Adam";
    const static QString inputDimNameString = "inputdim";
    const static QString outputDimNameString = "outputdim";
    const static QString kernelsizeNameString = "ksize";
    const static QString strideNameString = "stride";
    const static QString paddingNameString = "padding";
    const static QString channelNameString = "channel";
    const static QString inputHNameString = "inputH";
    const static QString inputWNameString = "inputW";
    const static QHash<QString, size_t> layerDimMap{
        {linearNameString, 1},
        {conv2dNameString, 3},
        {maxpool2dNameString, 3}
    };
}
