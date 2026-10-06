# Digit Recognition — 从零手写卷积神经网络（C++ / Qt）

不用 PyTorch、不用任何机器学习库。**矩阵运算、SIMD GEMM、im2col/col2im、卷积与池化的前反向、损失、优化器、数据增强全部自己实现**，在 MNIST 上训练到 **98.99%**（10000 张测试集）。

这是一个「把深度学习框架拆开重新装一遍」的练习：先自己写 `Matrix`（含 AVX2 手写内核），再用 `im2col + GEMM` 实现卷积，最后把各层、损失、优化器一层层搭起来。

## 实际的网络结构

配置写在 `SuperParam.h` 的 `cfg::getModel()` 里，是一段 JSON，由 `Model::loadFromJson` 解析并交给层工厂构造：

```
输入 28×28×1
  → Conv2D(1 → 8, k=3, s=1, p=1)   → 28×28×8
  → ReLU
  → MaxPool2D(2×2, s=2)            → 14×14×8
  → Linear(1568 → 128)             → 1568 = 8×14×14
  → ReLU
  → Linear(128 → 10)

损失：CrossEntropyLoss        优化器：动量 SGD
```

## 训练配置

| 项 | 值 | 位置 |
|---|---|---|
| 训练样本 | 60000 中随机划出 5000 作验证集，其余 55000 训练 | `main.cpp` |
| batch size | 64 | `SuperParam.h` |
| epoch | 30 | `SuperParam.h` |
| 学习率 | 0.015 起，每 7 轮减半（末尾 0.0009375） | `SuperParam.h` / `main.cpp` |
| 动量 | 0.9（有效步长 = lr/(1−β)） | `SuperParam.h` |
| 数据增强 | 随机平移 ±3 像素，每 epoch 重新随机 | `main.cpp` |
| 随机种子 | 42（`std::mt19937`） | `SuperParam.h` |

## 效果

| 数据集 | 准确率 |
|---|---|
| 训练集 | 99.27% |
| 验证集（5000） | 99.20% |
| **测试集（10000）** | **98.99%** |

训练与验证的差距只有 0.07%，过拟合基本为零。注意测试集 10000 张、准确率 99% 附近的标准误约 0.1%——所以 98.99% 与 99.00% 的差别只相当于 **1 张图**。

## 自己实现了哪些东西

| 模块 | 文件 | 说明 |
|---|---|---|
| 矩阵核心 | `TypeDefine.h/.cpp` | `Matrix`：加减乘、分块转置、`im2col`/`col2im`、按索引取行；四种 GEMM 内核（`matmul` / `_nn` / `_tn` / `_nt`）用 AVX2 手写 + OpenMP 并行 |
| 卷积 | `Conv2D.h/.cpp` | 前向 `im2col` → `matmul_nt`；反向用 `matmul_tn` 求 dW、`matmul_nn` 求 dA、`col2im` 还原 dX；支持 padding |
| 池化 | `MaxPool2D.h/.cpp` | 最大池化：forward 记录 argmax 的源索引，backward 散射累加（窗口重叠时同一像素被多次选中，必须用 `+=`） |
| 层抽象 | `Layer.h`、`Linear.h/.cpp`、`ReLU.h/.cpp` | 统一 `forward` / `backward`；参数通过 `ParamPtr{value, grad, momentum}` 暴露给优化器 |
| 损失 | `LossFunction.h`、`CrossEntropyLoss.*`、`MSELoss.*` | 交叉熵（启用）+ 均方误差 |
| 优化器 | `Optimizer.h/.cpp` | 动量 SGD，集中式更新：优化器只认 `ParamPtr`，完全不认识具体层类型 |
| 网络容器 | `Model.h/.cpp` | 从 JSON 加载网络结构 + 层工厂（含空间尺寸与通道数跟踪） |
| 数据读取 | `DataReader.h/.cpp` | 直接解析 MNIST 的 `idx3-ubyte` / `idx1-ubyte` 二进制格式 |
| 配置 | `SuperParam.h` | 超参数 + JSON 网络结构（`namespace cfg`） |
| 工具 | `Utils.h` | 正态分布随机数等 |
| 界面 / 计时 | `MainWindow.h/.cpp`、`main.cpp` | Qt 查看界面；训练循环自带五段计时 |

## 测试

矩阵与卷积相关的测试在同机另一个控制台工程 **`MatrixTest`** 中（未入库），与主工程**共用同一份 `TypeDefine.h`**：

- **GEMM**（S1–S3）：12 组维度 × 3 个内核与朴素三重循环对拍；累加语义（`out` 初值非零）；`matmul_nt` 与 `transpose + matmul_nn` 交叉验证
- **im2col**（T1–T7）：列布局、通道分块偏移、多形状逐元素对拍、C=1 与 C>1 的完整通路、padding 版与有符号坐标独立参照
- **col2im**（C1–C4）：恒等式 `col2im(im2col(X)) == X ⊙ count`、**伴随检验** `⟨im2col(X), G⟩ == ⟨X, col2im(G)⟩`、padding 位置的贡献必须为 0
- **池化**（`MaxPoolTest.h`，在本仓库）：手算期望、六种形状与独立参照对拍、argmax 路由、重叠累加（4 窗口同时选中中心像素时期望 4 倍）、**数值梯度中心差分对拍**

## 怎么跑起来

1. **环境**：Visual Studio 2022 + Qt（工程引用 `5.14.2_msvc2015_64`），`QtMsBuild` 由 Qt VS Tools 提供。
2. **数据**：MNIST 四个文件放在 `.sln` 同级目录（**不入库**，需自行下载解压）：

   ```
   train-images.idx3-ubyte   train-labels.idx1-ubyte
   t10k-images.idx3-ubyte    t10k-labels.idx1-ubyte
   ```

   来源：<http://yann.lecun.com/exdb/mnist/>
3. **路径**：`SuperParam.h` 里是**绝对路径**（`D:\code\c\Digit Recognition\...`），换机器要改这四行。
4. **构建**：VS 打开 `Digit Recognition.sln`，选 `Release|x64` 构建运行。工程已启用 AVX2、`/Ox`、`/GL` 与 **OpenMP**。

## 性能（batch = 64，i7-14650HX）

| 阶段 | ms/batch |
|---|---|
| data_prep（含数据增强） | 0.31 |
| net_forward | 12.50 |
| loss_forward | 0.02 |
| net_backward | 13.02 |
| optimizer | 0.36 |
| **合计** | **26.2**（约 22.5 秒 / epoch） |

逐层计时显示：conv 层占总时间约 **45%**，而其中真正的 GEMM 只有 0.5ms——**瓶颈在 im2col 与 NCHW/NHWC 重排的内存搬运，不在乘加**。

## 已知问题

- **数据路径硬编码**为绝对路径，换机器需改 `SuperParam.h`。
- **输入尺寸固定 28×28**（`cfg::H` / `cfg::W`），换数据集需一并调整。
- **测试套件分散**：GEMM / im2col / col2im 三套在单独的 `MatrixTest` 工程里，未入库。
- `Tensor.h/.cpp` 已实现但**未接入**当前网络链路。

## 说明

- 仓库记录的是**边学边写**的过程，保留了注释、实验痕迹与计时输出。
- 数据集与编译产物不入库：`.vs/`、`x64/`、`*.ipch`、`*.pdb`、`*.tlog`、MNIST 数据均已忽略。
