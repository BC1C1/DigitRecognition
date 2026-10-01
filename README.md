# Digit Recognition — 从零手写神经网络（C++ / Qt）

不用 PyTorch、不用任何机器学习库，**矩阵运算、层、损失、优化器、反向传播全部自己实现**，在 MNIST 手写数字数据集上训练。

这是一个「把深度学习框架拆开重新装一遍」的练习：先自己写 `Matrix`，再把线性层、激活、损失、优化器一层层搭起来，用 Qt 做个界面看结果。

## 实际的网络结构（以 `main.cpp` 为准）

```
输入 784 (28×28 展平)
  → Linear 784 → 512 → ReLU
  → Linear 512 → 128 → ReLU
  → Linear 128 → 10
损失：CrossEntropyLoss      优化器：Optimizer（含 SGD 实现）
```

```cpp
model.addLayer(linear(784, 512));
model.addLayer(relu());
model.addLayer(linear(512, 128));
model.addLayer(relu());
model.addLayer(linear(128, 10));
```

**当前跑起来的是全连接网络（MLP）。**
`Tensor` 与 `Conv2D` 是为卷积网络准备的基础件 —— `Tensor` 已实现，`Conv2D` 尚在起步，**还没接进上面这条链路**。

## 训练配置

| 项 | 值 | 位置 |
|---|---|---|
| 训练样本 | 60000 | `main.cpp` |
| batch size | 64 | `main.cpp` |
| epoch | 10 | `main.cpp` |
| 学习率 | 0.05 | `SuperParam.h` |
| 随机种子 | 42（`std::mt19937`，且 `shuffle` 每 epoch 重排） | `main.cpp` / `SuperParam.h` |

## 自己实现了哪些东西

| 模块 | 文件 | 说明 |
|---|---|---|
| 矩阵 | `Matrix.h/.cpp` | 整个项目的地基：加减乘、转置、按索引取行、逐元素运算 |
| 张量 | `Tensor.h/.cpp` | 为卷积/多维数据准备（已实现，尚未接入网络） |
| 层抽象 | `Layer.h/.cpp`、`Linear.h/.cpp`、`ReLU.h/.cpp` | 统一 `forward` / `backward` 接口 |
| 卷积 | `Conv2D.h/.cpp` | 起步阶段，未完成 |
| 损失 | `LossFunction.h/.cpp`、`CrossEntropyLoss.*`、`MSELoss.*` | 交叉熵 + 均方误差（后者未启用） |
| 优化器 | `Optimizer.h/.cpp`、`SGD.h/.cpp` | 参数更新 |
| 网络容器 | `Model.h/.cpp` | 串联各层，`net_forward` / `net_backward` / `predict` |
| 数据读取 | `DataReader.h/.cpp` | 直接解析 MNIST 的 `idx3-ubyte` / `idx1-ubyte` 二进制格式 |
| 类型定义 | `TypeDefine.h/.cpp` | 自定义的数据类型与张量语义 |
| 计时 | `main.cpp` | 自己给"数据准备 / 前向 / 损失 / 反向 / 优化"五段分别计时 |

## 怎么跑起来

1. **环境**：Visual Studio 2022 + Qt（工程引用的是 `5.14.2_msvc2015_64`），`QtMsBuild` 由 Qt VS Tools 提供。
2. **数据**：MNIST 四个文件放在项目根目录（与 `.sln` 同级）：

   ```
   train-images.idx3-ubyte   train-labels.idx1-ubyte
   t10k-images.idx3-ubyte    t10k-labels.idx1-ubyte
   ```

   来源：<http://yann.lecun.com/exdb/mnist/>（或 `MNIST_data.zip`）。这些文件**不入库**（见 `.gitignore`），需要自行下载解压。
3. **路径**：`main.cpp` 里目前是**绝对路径**（`D:\code\c\Digit Recognition\...`）。换机器需要改这两行，或者后续改成读相对路径 / 命令行参数。
4. **构建**：用 VS 打开 `Digit Recognition.sln`，配置选 `Release|x64` 构建运行。

## 已知问题（未修）

- **测试集读取越界**：测试阶段用 `M = 10000` 做结果判定，但测试数据是**按 `M = 60000` 读进来的**（`t10k-*` 只有 10000 条）。这会读到不属于测试集的标签。训练部分不受影响（训练集确实 60000）。
- **数据路径硬编码**为绝对路径（`D:\code\c\Digit Recognition\...`）。
- **数据有两份**：项目根目录是解压后的 `idx3/idx1`（代码读的是这份），`MNIST_data\` 下另有一份 `.gz` 压缩原件与 `MNIST_data.zip`（未使用，也不入库）。

## 说明

- 这个仓库记录的是**边学边写**的过程，代码里保留了注释、实验痕迹与计时输出。
- 数据集与编译产物不入库：`.vs/`、`x64/`、`*.ipch`、`*.pdb`、`*.tlog`、MNIST 数据均已忽略。
