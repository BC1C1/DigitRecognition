# MachineLearning

从零手写的 C++ / CUDA 卷积网络，跑 MNIST 与 Fashion-MNIST。

没有框架、没有 cuBLAS、没有 cuDNN —— 矩阵运算、im2col 卷积、最大池化、
log-softmax 交叉熵、Adam，全部自己写 kernel。张量常驻显存，
`Matrix` 是一个带引用计数的句柄。

这是一个**学习项目**：目的是把每一步都亲手实现并验证一遍，不是为了刷分。
所以代码里解释「为什么这么写」的注释比重很高，测试也写得比一般练习项目认真。

当前状态：**GPU 训练**。30 轮 test acc —— MNIST **0.9902**、Fashion-MNIST **0.9286**；
显存稳定在 **393 MiB**；MNIST 跑满 30 轮含每轮两次评估约 90 秒。

---

## 目录结构

```
MachineLearning_cmake/
├── CMakeLists.txt              顶层：声明 CXX + CUDA，CUDA_ARCHITECTURES=89
├── Matrix/                     静态库：矩阵核心（全部设备端）
│   ├── include/TypeDefine.h      Matrix 类 + MatrixData（引用计数、显存持有）
│   ├── include/cuda_bridge.h     33 个 launch* 启动函数声明（extern "C"）
│   ├── include/cuda_utils.h      cudaMalloc_throw、cuRAND 生成器单例
│   └── src/Matrix.cu             全部 kernel 与 launch 实现
├── MatrixTest/                 测试程序
│   ├── include/GpuTestCommon.h   显式 cudaMemcpy 的上传/下载、LCG 造数、断言
│   ├── include/GpuTestElementwise.h  元素级算子
│   ├── include/GpuTestMatmul.h       GEMM 三变体
│   ├── include/GpuTestReduce.h       归约
│   ├── include/GpuTestConv.h         im2col / col2im / convGemm / maxpool
│   ├── include/GpuTestSlice.h        按索引挑行
│   ├── include/GpuTestData.h         平移增强 / 随机填充
│   ├── include/GpuTestModel.h        层 / 损失 / 优化器
│   ├── include/GpuTest.h             总入口
│   ├── include/{MatmulTest,Im2ColTest,Col2ImTest,Dispatchers,TestCommon}.h
│   │                             CPU 版测试，已摘出编译，留作参考
│   └── src/main.cpp              退出码 = 失败条数
├── MachineLearning/            控制台程序
│   ├── Layers/                   Conv2D / Linear / ReLU / MaxPool2D / Layer
│   ├── LossFunctions/            CrossEntropyLoss / MSELoss / LossFunction
│   ├── Optimizer/                Optimizer 是 Adam 实现，SGD 只是接口名
│   ├── Model/                    Model：层容器 + 前后向 + JSON 建网
│   ├── Dao/                      DataReader：读 idx 文件、两种命名约定都试
│   ├── config/SuperParam.h       超参数 + 网络结构 JSON + datasetDir
│   ├── utils/Utils.h             空壳
│   └── src/main.cpp              训练循环 + 评估
├── MNIST/                       train-images.idx3-ubyte 等四个文件（不进仓库）
└── fashion/                     train-images-idx3-ubyte 等四个文件（不进仓库）
```

依赖方向：`MatrixTest → Matrix`，`MachineLearning → Matrix`。

---

## 构建

### 依赖

- CMake ≥ 3.20
- CUDA Toolkit（实测 13.4），`CMAKE_CUDA_ARCHITECTURES=89`
- Qt5 Core —— **只用来做 JSON 解析、文件读取和路径拼接，没有 GUI**
- 不再需要 AVX2：host 侧的矩阵运算已经全部搬走

### 生成与编译

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

### Windows 上踩过的坑

1. 装了多个 VS 实例时，上面那条 `-G` 会挑到其中一个。要指定用哪个，加
   `-DCMAKE_GENERATOR_INSTANCE="C:/Program Files/Microsoft Visual Studio/2022/Community"`。
   注意这跟 PATH 里解析到哪个 `cmake.exe` **是两个独立的开关**。
2. 命令行 configure 时，MSBuild 的 CUDA 工具链靠 `CUDA_PATH_V13_4` 找 nvcc，
   需要在当前 shell 里先设好这个环境变量。
3. `set(CMAKE_CUDA_ARCHITECTURES 89)` **必须放在任何 `add_executable` 之前**。
   放后面不会报错，只是静默失效 —— 生成出来的 cubin 还是 nvcc 的默认架构
   （用 `cuobjdump -lelf` 能看出来）。
4. **重新链接时会失败（LNK1104），如果目标 exe 正在运行。** 训练跑着的时候
   想重编，得先停掉它。
5. 用 VS 直接打开 `build/MachineLearning.sln` 也行。改了 `CMakeLists.txt` 不用手动重配，
   ZERO_CHECK 会在下次构建时自动重新 configure；但**在属性页里改的任何设置，
   下次 configure 都会被抹掉** —— 要改就改 CMakeLists。
6. 从 VS 里按 F5 启动默认是 **Debug 配置**（`/Od`、`/RTC1`），会慢好几倍。
   测性能请跑 `build\MachineLearning\Release\MachineLearning.exe`。

### 数据

四个 `.idx` 文件放在仓库根目录下的 `MNIST/` 或 `fashion/` 里，`SuperParam.h`
的 `datasetDir` 指向其中一个（**绝对路径** —— 相对路径会跟着工作目录走，
而这个程序是从 `build\MachineLearning\Release\` 启动的）。

**两套命名的差异，换数据集时唯一要留意的坑：**

| 数据集 | 文件名 |
|---|---|
| MNIST | `train-images.idx3-ubyte`（**点号**） |
| Fashion-MNIST | `train-images-idx3-ubyte`（**连字符**） |

`DataReader` 的 `findDataFile` 两种都试，所以切换时只需要改 `datasetDir` 一行。

⚠ 换数据集时**不要只改训练集**：`readPicture` / `readLabel` 收一个 `split`
参数（`"train"` / `"t10k"`），四个调用点必须各自传对。这个参数是后加的 ——
在它之前测试集评估读的其实是训练集，而因为 `10000 < 60000` 不触发裁剪，
**整个过程静默无提示**，报出来的"测试集准确率"其实是训练集前 10000 张。

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
| Conv2D | 1→32, k=3, s=1, p=1 | (N, 32·28·28) |
| ReLU | | |
| MaxPool2D | k=2, s=2 | (N, 32·14·14) = (N, 6272) |
| Linear | 6272→128 | (N, 128) |
| ReLU | | |
| Linear | 128→10 | (N, 10)，logits |
| CrossEntropyLoss | 内含 log-softmax | 标量 |

### 维度的秩推导

建网时用一个 `QVector<size_t> dimlist` 逐层传递当前形状，每层声明自己需要几维
（`layerDimMap`：conv2d/maxpool2d = 3，linear = 1）：

- 当前秩 < 该层需要的秩 → **JSON 必须显式给**新增维度（首次进卷积要写
  `inputdim`/`inputH`/`inputW`）
- 当前秩 ≥ 需要 → 自动从 `dimlist` 推导；JSON 里若也写了，只做校验（不一致时
  打印 warning，然后用推导值）
- `linear` 会清空 `dimlist` 只留输出维度；`relu` 不碰它

### 数据布局

所有张量都是 `Matrix`，形状写成 `(N, C*H*W)` —— 一张图的所有通道拍平在一行，
通道优先（`c*H*W + h*W + w`）。没有真正的四维张量，全靠这个约定撑着。

代价是 `Conv2D` 前向要把 `(N·OH·OW, OC)` 重排成 `(N, OC·OH·OW)`，反向开头再
逆重排 —— 这两步现在是 `launchConv2DGemmKernel` / `launchConv2DBackwardReshape`
两个 kernel。

---

## 训练配置

| 项 | 值 |
|---|---|
| 数据 | 60000 张；shuffle 后前 5000 作验证集，剩 55000 训练 |
| batch | 64 |
| 轮数 | 30 |
| 优化器 | **Adam**（lr 0.001、β₁ 0.9、β₂ 0.999、ε 1e-8） |
| 学习率衰减 | 验证集不提升时减半 |
| 权重初始化 | `randn × sqrt(2 / fan_in)`（He），由 cuRAND 生成，固定种子 42 |
| 数据增强 | 每张图随机平移 ±3 像素（kernel 实现） |
| 测试集 | 10000 张，只在最后跑一次 |

### 显存占用

**稳定在 393 MiB**。构成大致是：

| 项 | 大小 |
|---|---|
| `images` (60000, 784) float | 179 MiB |
| `validImgs` + `validLabels` (5000) | 15 MiB |
| CUDA 上下文 + cuRAND 生成器 | 50~60 MiB |
| 各层权重与缓冲（训练态） | 几个 MiB |

**这里踩过一个坑，值得记下来**：评估最初用 `test_chunk = 1000` 而训练 batch 是
64，于是每个 epoch 末尾所有层的 `reshape` 都判定"尺寸变了"、把整套缓冲重新
分配一遍（conv2 的 `A`/`dA` 各约 226 MB），评估完再分配回去 —— **每轮一次
约 1.5 GiB 的显存尖峰**，而低谷只有 400 MiB。让评估也用 `batchSize` 之后缓冲
完全复用，曲线变成一条平线。

### 一次循环的五个阶段

计时曾经按 `std::chrono` 切五段（data_prep / net_forward / loss_forward /
net_backward / optimizer）。**GPU 版不能这么测**：kernel 是异步提交的，
`Clock::now()` 只测到"提交完成"，不是"算完"。要恢复这个表，得在每个计时点
后面补 `cudaDeviceSynchronize()`。

### 学习率衰减

```cpp
if (validAcc > maxAcc) { maxAcc = validAcc; noImprove = 0; }
else { noImprove++; if (noImprove > patience) { noImprove = 0; lr /= 2; } }
```

注意 `noImprove > patience`（patience = 3）实际是"连续 5 轮不提升才减半"，
比字面上更宽松。

⚠ **`maxAcc` 只记录历史最优，但不回滚最优权重** —— 所以 `final acc` 是最后一轮
的模型，不是最好的那个。

---

## 矩阵核心

`MatrixData` 持有显存指针、一个懒拷贝的 host 副本、以及引用计数；
`Matrix` 只是句柄：拷贝构造/赋值是**共享同一块显存**（加引用计数），
移动是转移指针，引用计数归零才 `cudaFree`。

- `Matrix(0, c)` / `Matrix(r, 0)` 不分配（`data = nullptr`）
- `cudaMalloc` 失败抛 `std::runtime_error`（`cudaMalloc_throw`）
- 零矩阵/随机矩阵是工厂函数：`zeroMatrix` 走 fill kernel，`randnMatrix` 走 cuRAND

⚠ **`Matrix::operator()` 仍然存在，但它解引用的是显存指针**，在 host 上调用即
非法访问。所有 host 侧的读写都必须显式 `cudaMemcpy`（`MatrixTest` 的
`upload` / `download` 就是这么做的）。

三个 GEMM 变体（都在 `Matrix.cu`）：

| 内核 | 形状 | 谁在用 |
|---|---|---|
| `launchMatmulNN` | A(M,K) × B(K,N) | Conv2D 反向求 dA、Linear 反向求 dx |
| `launchMatmulNT` | A(M,K) × B(N,K)ᵀ | 卷积前向、全连接前向 |
| `launchMatmulTN` | A(K,M)ᵀ × B(K,N) | 反向求 dW |

⚠ **三个都是覆盖写**（`gc[...] = value`），不是累加 —— 这与 CPU 版语义相反
（那边是 `out += ...`，调用前要清零）。调用方按"输出会被覆盖"写。

卷积走 im2col + GEMM：`launchIm2col` 把每个滑动窗口摊平成一行，于是卷积变成
一次矩阵乘法；反向用 `launchCol2im` 把梯度散射回输入尺寸 —— 它是 im2col 的
伴随算子，padding 区的贡献必须丢弃。

**当前实现是朴素的**：GEMM 没有共享内存分块、没有寄存器复用，每个线程沿 K 串行
累加。实测 GPU 利用率约 89%、显存带宽只有 27%、功耗 34 W —— 说明执行单元在
等访存延迟而不是在算。**分块优化是下一步**（README 末尾有排序）。

---

## 测试

```powershell
build\MatrixTest\Release\MatrixTest.exe
```

退出码 = 失败条数（全绿是 0），可以直接挂 CTest。

分七组，按依赖从下往上：

| 文件 | 内容 |
|---|---|
| `GpuTestElementwise.h` | fill / add / sub / scaleMul / elementwiseMul，含原地版本与长度边界 |
| `GpuTestMatmul.h` | GEMM 三变体多形状 + 覆盖写语义 |
| `GpuTestReduce.h` | sumRows / sumCols / sumKernel |
| `GpuTestConv.h` | im2col / col2im / 伴随检验 / convGemm / 逆重排 / maxpool |
| `GpuTestSlice.h` | 按索引挑行：哨兵检出漏写、下标乱序/重复/取满 |
| `GpuTestData.h` | 平移增强（极端位移、非方形、grid 覆盖）+ 随机填充 |
| `GpuTestModel.h` | Linear / ReLU / logSoftmax / MSE / Adam 一步 |

**参照实现全部在测试里重写，绝不复用被测内核。** 否则就是自证 ——
内核错了参照也跟着错，全绿毫无意义。

⚠ 用例的前提：`count <= M` 且 `idx[i] < M`。这条在底层只是一行 `assert`，
而 Release 下 `assert` 是空的 —— 违反它不会报错，会越界读。

### 两把尺子

定义在 `GpuTestCommon.h`：

- `kExactTol = 1e-12` —— 给**纯搬运**用。im2col、col2im、按索引挑行、平移增强
  只重排位置、不做算术，float 之间应当**逐位相等**。
- `kFloatTol = 1e-3` —— 给**参与乘加**的路径用。

依据：float 只有 24 位尾数（eps ≈ 1.19e-7），K 维累加后的绝对误差按
`eps · sqrt(K) · |值|` 量级增长。实测（K 最大 120）最大误差 **7.6e-6**；
而「索引 / 搬运 / 转置写错」会给出 O(1) 以上的差。两者隔着五个数量级，
1e-3 落在中间。将来 K 放大到几千（im2col 卷积的 12544），误差也只涨到 1e-4
量级，仍然不会误报。

---

## 实测

Release，30 轮，Adam，带 ±3 平移增强：

| 数据集 | test acc | 说明 |
|---|---|---|
| MNIST | **0.9902** | 不带增强时 0.9873；增强把 train/valid 的差距从 1.2% 压到 0.2% |
| Fashion-MNIST | **0.9286** | 同架构，未调参 |

- 显存稳定 **393 MiB**
- 一轮约 3 秒（含每轮末尾两次评估）
- 固定种子 42，同样的代码跑两次结果逐位相同

---

## 已知问题

按"会不会真咬人"排，前四条来自一次完整的代码审查。

1. **`Model.h` 的校验全是 `assert`，而 Release 已定义 `NDEBUG`** —— 全部被编译掉。
   `optimizerFromQString` / `lossFromQString` 认错时返回 `nullptr`，随后
   `op->addLayer(...)` 空指针解引用崩；未知 layer type 会把 `nullptr` 压进
   layers，前向时崩。**未修。**

2. **`Model.h` 里 `params[..].toInt()` 对缺键返回 0 且无兜底** —— 漏写 `stride`
   导致整数除零，漏写 `outputdim` 会构造 0 维 `Matrix`（`data = nullptr`）而
   kernel 静默不写。一个 typo 不报错。**未修。**
   最小成本的方向：在 `linear` / `conv2d` / `maxpool2d` 三个工厂函数里各加一行
   "维度不为 0" 检查 —— 那能覆盖全部维度参数，比逐字段校验省事。

3. **`DataReader` 读文件失败只 `qDebug` 后返回空 `Matrix()`** —— 没有异常、没有
   错误位。配合 `launchOneHotAndCompare` 里未初始化的 `retVector`，会打出**假的
   准确率数字**（读的是未初始化内存，不保证一定是 100%，但一定没有意义）。
   换数据集时路径写错就会重现。**未修。**

4. **`TypeDefine.h::sliceRowsByIndex` 里越界索引的检查被注释掉了** —— 调用方给错
   索引就是无防线的非法访存，Debug 下也不报。校验应该放在**上传索引表之前、
   在 host 侧对 `train_idx` 做**，而不是在设备指针上逐个读。**未修。**

5. **`main.cpp` 的 "train acc" 口径** —— 取的是 `images`/`labels` 的**文件顺序前
   55000 行**，而实际训练用的是 shuffle 后的 `train_idx`。两者不是同一集合：
   约 4583 个验证样本被算进了"训练准确率"，同时漏掉等量的训练样本。不影响梯度，
   但该指标偏乐观。**未修**（当作趋势看即可）。

6. **`Conv2D` / `MaxPool2D` 的 `reshape` 只比 `rows()`**，不比 cols 与通道数
   （ReLU / Linear / 两个 loss 都同时比了）。当前固定拓扑够用，换网络结构时可能
   漏重建缓冲。**未修。**

7. **训练循环全程不检查 CUDA 错误码** —— 这是上面几条"静默"能得逞的共同根因。
   另外异常路径上 `main.cpp` 的两处 `cudaFree` 走不到。**未修。**

8. **`main.cpp:135` 传给 `launchOneHotAndCompare` 的是 `cur` 而非 `result.rows()`**;
   `sliceRowsByIndex` 在 `n > rows()` 时静默截断成 `min(n, rows())` —— 尺寸一旦
   不一致就按 `cur` 越界读结果。**未修。**

9. **`MatrixTest/include/` 下五个 CPU 版测试文件已摘出编译** —— `MatmulTest.h`、
   `Im2ColTest.h`、`Col2ImTest.h`、`Dispatchers.h`、`TestCommon.h`。它们的参照
   实现全走 `Matrix::operator()`，在 GPU 矩阵上会段错误。**留作历史参考。**

10. **`Matrix::fill(const Matrix&)` 是空壳** —— 函数体只剩三个 `assert`，且其中
    一句注释文案与条件相反。`NDEBUG` 下整个函数是空操作，调用方会以为已完成填充。
    **未修。**

---

## 下一步

按收益排序：

1. **GEMM 的共享内存分块** —— 直接打那 89% 利用率 / 27% 带宽的错配。这是当前
   最大的性能空间（kernel 是朴素的，每个线程沿 K 串行读全局内存）。
2. **补分段计时** —— GPU 上 kernel 是异步的，每个计时点后要 `cudaDeviceSynchronize()`。
   README 里那张表已经作废，需要一个能反映真实分布的版本。
3. **修掉上面【已知问题】的 1~4 条** —— 都是"静默出错"那一族，各几行。
4. **显存池** —— `MatrixData` 的每次构造都 `cudaMalloc`、析构都 `cudaFree`。
   训练稳态下每批仍有约 10 次 malloc/free（`batch_X`、`batch_T`、增强输出、
   每 chunk 的索引表、几个标量缓冲）。10.54 ms/批的 CPU 版本里，单次 matmul 的
   固定开销约 0.2 ms 全部来自三次 `cudaMalloc` + 三次 `cudaFree`。
5. **去掉逐层同步**：现在每次 `cudaMemcpy` D2H（如取 loss 标量）都会排空流水线。
