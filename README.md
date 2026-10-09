# MachineLearning

从零手写的 C++ 卷积网络，跑 MNIST 手写数字识别。

没有框架、没有 BLAS、没有 cuDNN —— 矩阵乘法（AVX2 + FMA 手写内核）、im2col 卷积、
最大池化、log-softmax 交叉熵、带动量的 SGD，全部自己写。

这是一个**学习项目**：目的是把每一步都亲手实现并验证一遍，不是为了刷分。
所以代码里解释「为什么这么写」的注释比重很高，测试也写得比一般练习项目认真。

当前状态：CPU 训练，30 轮 test acc **0.9913**，每批约 **10.5 ms**。
CUDA 已接入构建系统，但还没有设备端代码 —— 那是下一步。

---

## 目录结构

```
MachineLearning_cmake/
├── CMakeLists.txt              顶层：声明 CXX + CUDA，CUDA_ARCHITECTURES=89
├── Matrix/                     静态库：矩阵核心
│   ├── include/TypeDefine.h      Matrix 类
│   └── src/TypeDefine.cpp        Matrix::blocksize 的定义
├── MatrixTest/                 测试程序
│   ├── include/MatmulTest.h      GEMM 三内核对拍
│   ├── include/Im2ColTest.h      im2col
│   ├── include/Col2ImTest.h      col2im（im2col 的伴随算子）
│   ├── include/TestCommon.h      断言、容差、maxDiff
│   └── include/Dispatchers.h     已停用，见【已知问题】第 5 条
└── MachineLearning/            控制台程序
    ├── Layers/                   Conv2D / Linear / ReLU / MaxPool2D / Layer
    ├── LossFunctions/            CrossEntropyLoss / MSELoss / LossFunction
    ├── Optimizer/                Optimizer 是 SGD + 动量，SGD 是接口
    ├── Model/                    Model：层容器 + 前反向 + JSON 建网
    ├── Dao/                      DataReader：读 MNIST idx 文件
    ├── config/SuperParam.h       超参数 + 网络结构的 JSON
    ├── utils/Utils.h             空壳，暂时留着
    └── src/main.cpp              训练循环 + 分段计时
```

依赖方向：`MatrixTest → Matrix`，`MachineLearning → Matrix`。

---

## 构建

### 依赖

- CMake ≥ 3.20
- x86-64 CPU，要求 **AVX2 + FMA**（内核直接用 `_mm256_fmadd_ps`，没有降级路径）
- OpenMP
- Qt5 Core —— **只用来做 JSON 解析和文件读取，没有 GUI**
- CUDA Toolkit（构建系统已接入，尚无 `.cu`）

### 生成与编译

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

### Windows 上踩过的坑

1. 装了多个 VS 实例时，上面那条 `-G` 会挑到其中一个。要指定用哪个，加
   `-DCMAKE_GENERATOR_INSTANCE="C:/Program Files/Microsoft Visual Studio/2022/Professional"`。
   注意这跟 PATH 里解析到哪个 `cmake.exe` **是两个独立的开关**。
2. 命令行 configure 时，MSBuild 的 CUDA 工具链靠 `CUDA_PATH_V13_4` 找 nvcc，
   需要在当前 shell 里先设好这个环境变量。
3. `set(CMAKE_CUDA_ARCHITECTURES 89)` **必须放在任何 `add_executable` 之前**。
   放后面不会报错，只是静默失效 —— 生成出来的 cubin 还是 nvcc 的默认架构
   （用 `cuobjdump -lelf` 能看出来）。
4. `CMAKE_CUDA_ARCHITECTURES` 的缓存值可能不是你以为的那个：CMake 会去正则匹配
   nvcc 自己打印的默认 `-arch compute_NNN`。
5. 用 VS 直接打开 `build/MachineLearning.sln` 也行。改了 `CMakeLists.txt` 不用手动重配，
   ZERO_CHECK 会在下次构建时自动重新 configure；但**在属性页里改的任何设置，
   下次 configure 都会被抹掉** —— 要改就改 CMakeLists。

### 数据

MNIST 的四个 `.idx3-ubyte` / `.idx1-ubyte` 文件路径写在 `config/SuperParam.h` 里，
目前是**绝对路径**，换机器或移动目录要改。

### 运行

```powershell
build\MachineLearning\Release\MachineLearning.exe   # 训练
build\MatrixTest\Release\MatrixTest.exe             # 测试
```

---

## 网络结构

`config/SuperParam.h::getModel()` 里是一段 JSON，`Model::loadFromJson` 解析它建网。
加层、改层不用动 C++。

| 层 | 参数 | 输出 |
|---|---|---|
| 输入 | | (N, 784) |
| Conv2D | 1→8, k=3, s=1, p=1 | (N, 8·28·28) |
| ReLU | | |
| MaxPool2D | k=2, s=2 | (N, 8·14·14) = (N, 1568) |
| Linear | 1568→128 | (N, 128) |
| ReLU | | |
| Linear | 128→10 | (N, 10)，logits |
| CrossEntropyLoss | 内含 log-softmax | 标量 |

### 数据布局

所有张量都是 `Matrix`，形状写成 `(N, C*H*W)` —— 一张图的所有通道拍平在一行，
通道优先（`c*H*W + h*W + w`）。

没有真正的四维张量，全靠这个约定撑着。代价是 `Conv2D` 前向末尾要把
`(N·OH·OW, OC)` 重排成 `(N, OC·OH·OW)`，反向开头再逆重排 —— 两处手写四重循环。

---

## 训练配置

| 项 | 值 |
|---|---|
| 数据 | MNIST 60000 张；shuffle 后前 5000 作验证集，剩 55000 训练 |
| batch | 64 |
| 轮数 | 30 |
| 初始学习率 | 0.015 |
| 动量 | 0.9 |
| 权重初始化 | `randn × sqrt(2 / fan_in)`（He） |
| 数据增强 | 每张图随机平移 ±3 像素 |
| 随机种子 | 固定 42（`std::mt19937`），同样的代码跑两次结果逐位相同 |
| 测试集 | 10000 张，只在最后跑一次 |

平移增强是先建一张零矩阵再写进去的 —— 原地做会覆盖还没读到的像素。

### 一次循环的五个阶段

`main.cpp` 里按 `std::chrono` 的计时点切分：

| 阶段 | 实测（Release，8 线程） | 内容 |
|---|---|---|
| data_prep | 0.141 ms | 按下标切出 64 张图 + one-hot 标签，做平移增强 |
| net_forward | 5.033 ms | 逐层前向 |
| loss_forward | 0.009 ms | log-softmax 后取对应项求和 |
| net_backward | 5.052 ms | 反向逐层求梯度 |
| optimizer | 0.302 ms | `v = v*0.9 + g; w -= v*lr` |
| **合计** | **10.54 ms** | |

⚠ 这个计时**只覆盖训练循环本身**，不含每个 epoch 末尾的两次评估
（valid 5000 张 + train 55000 张）。所以 `10.54 ms × 860 批` 不是一轮的真实墙钟时间。

### 学习率衰减

不是固定周期，而是看验证集：

```cpp
if (validAcc > maxAcc) maxAcc = validAcc;
else { noImprove++; if (noImprove > 3) { noImprove = 0; lr /= 2; } }
```

这里有个偏离：**创新高的分支没有把 `noImprove` 归零**。所以它数的是
「自上次减半以来、没刷新最高分的 epoch 总数」，而不是标准的「连续 3 次不提升」。
结果是减半比标准 patience 更频繁 —— 实测在 epoch 11 / 17 / 22 / 27 各减半一次，
lr 从 0.015 一路走到 0.0009375。详见【已知问题】第 1 条。

---

## 矩阵核心

`MatrixData` 持有 `float*`，`Matrix` 只是句柄：拷贝是深拷贝，移动是转移指针。

三个 GEMM 变体：

| 内核 | 形状 | 谁在用 |
|---|---|---|
| `matmul_nn` | A(M,K) × B(K,N) | Conv2D 反向求 dA、Linear 反向求 dx |
| `matmul_nt` | A(M,K) × B(N,K)ᵀ | 卷积前向、全连接前向 |
| `matmul_tn` | A(K,M)ᵀ × B(K,N) | 反向求 dW |

⚠ **三个都是累加语义**，内部写 `out += ...`，调用前必须清零。
这条约定由 `MatmulTest` 的 S2 钉住 —— 哪天内核改成覆盖写，S2 会红，
提醒你回头改调用方（Conv2D 之类）。

实现：AVX2 + FMA，`__m256` 一次吞 8 个 float，K 方向 4 路展开（一轮 32 个），
`num_threads(8)` 硬编码，`schedule(dynamic)`，分块大小 `Matrix::blocksize = 32`。

卷积走的是 im2col + GEMM：`im2col` 把每个滑动窗口摊平成一行，于是卷积变成一次 `matmul_nt`。
反向用 `col2im` 把梯度散射回输入尺寸 —— 它是 im2col 的伴随算子，padding 区的贡献必须丢弃。

---

## 测试

```powershell
build\MatrixTest\Release\MatrixTest.exe
```

退出码 = 失败条数（全绿是 0），可以直接挂 CTest。

| 文件 | 内容 |
|---|---|
| `MatmulTest.h` | S1：12 组形状 × 3 内核 vs 朴素三重循环；S2：累加语义；S3：`nt` 与 `transpose()+nn` 两条独立路径互证 |
| `Im2ColTest.h` | 取值、形状、padding、卷积通路对拍 |
| `Col2ImTest.h` | 恒等式、伴随检验 `<im2col(X),G> == <X,col2im(G)>`、多形状扫描、padding 归零 |

**参照实现全部在测试里重写，绝不复用被测内核。** 否则就是自证 ——
内核错了参照也跟着错，全绿毫无意义。

### 容差为什么是 1e-3

两个常量定义在 `TestCommon.h`：

- `kExactTol = 1e-12` —— 给**纯搬运**用。im2col 只重排位置、不做算术，
  改成 float 之后依然逐位精确（实测全绿）。
- `kFloatTol = 1e-3` —— 给**参与乘加**的路径用。

依据：float 只有 24 位尾数（eps ≈ 1.19e-7），K 维累加后的绝对误差按
`eps · sqrt(K) · |值|` 量级增长。实测（K 最大 120）最大误差 **7.6e-6**；
而「索引 / 搬运 / 转置写错」会给出 O(1) 以上的差。两者隔着五个数量级，
1e-3 落在中间 —— 离实测留 131 倍余量，离真 bug 还差 1000 倍。
将来 K 放大到几千（im2col 卷积的 12544），误差也只涨到 1e-4 量级，仍然不会误报。

---

## 实测

Release，30 轮：

- test acc（10000 张）**0.9913**
- 每批平均 **10.54 ms**（明细见上）
- 训练曲线平滑上升，train 与 valid 的差距很小

---

## 已知问题

1. **lr 衰减的计数器不复位**。偏离标准 patience 语义，减半比标准实现更频繁。**未修**。

2. **`Model::loadFromJson` 只能调用一次**。`layerFromNameWithParams` 里的 `curH` / `curW`
   是函数内静态变量，尺寸会跨调用残留，第二次调用会从上次的尺寸接着往下推。
   当前只有一个网络，不影响使用。**不打算修**。

3. **数据路径是绝对路径**（`config/SuperParam.h`），换机器或移动目录要改。

4. **「train acc」的口径**。`main.cpp` 里那次评估用的是 `images` 的前 55000 行（原始顺序），
   其中约 4600 张属于验证集、没参与训练。所以这个数比真实的训练集准确率**偏低** ——
   也就是说，**过拟合程度比显示的更严重**。10k 测试集不受影响，最终 acc 是干净的。

5. **`MatrixTest/include/Dispatchers.h` 已停用**。它是 float 化之前的 double 版 AVX 内核
   加一套线程/块扫参工具，现在编不过，已从 `main.cpp` 的 include 里摘出。
   留着就得改成 float 或退回标量，不留就删。

6. **`Matrix::fill` 有三个重载、两种语义**：`fill(float) const` 返回新矩阵，
   `fill(float)` 和 `fill(const Matrix&)` 原地改。实际只用后两个。

7. **CUDA 还没有设备端代码**。构建系统已声明 CUDA 语言并链接 `cudart`，
   但 `Matrix` 里还没有 `.cu` 文件。

---

## 下一步：搬到 GPU

- `Matrix` 增加设备端模式，矩阵可以驻留显存
- 跨设备访问必须**显式**（不隐式拷贝），配一个计数器让隐藏拷贝现形
- 显存池 / arena 要和「GPU 为主」模式一起上，否则每层的 `cudaMalloc` 会吃掉全部收益
  —— 实测单次 matmul 的固定开销约 0.2 ms，全来自三次 `cudaMalloc` + 三次 `cudaFree`
- 顺序：先让 kernel 算对（靠 MatrixTest 对拍），再谈常驻显存、去掉逐层同步、共享内存分块
