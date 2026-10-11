#pragma once
// ============================================================================
//  GpuTest.h —— GPU 算子测试的总入口
//
//  为什么这份文件是重写的：原来这里是 G1/G2/G3 三个用例（逐元素加、matmul、
//  覆盖写语义），那是"只有两个 kernel"的阶段的写法。现在 Matrix.cu 里有 24 个
//  launch，测试按算子类别拆成了几个文件，入口只负责组织顺序和汇总。
//
//  同时，MatrixTest 里原来那三个 CPU 测试（MatmulTest / Im2ColTest / Col2ImTest）
//  已经从 CMakeLists 里摘掉了。原因是它们的参照实现全部走 Matrix::operator()，
//  而 operator() 现在解引用的是**显存指针** —— 在 GPU 矩阵上调用会段错误。
//  它们不是"测试失败了"，是"测试本身的前提没了"。等这些算子都有了对拍，
//  那几个文件可以作为历史参考留着，但不再参与编译。
//
//  测试分五组，顺序按依赖从下往上：
//      元素级   —— 所有算子都会用到的最底层
//      GEMM     —— 三个乘法变体
//      归约     —— sumRows / sumCols / sum
//      卷积通路 —— im2col / col2im / convGemm / reshape / maxpool
//      层·损失·优化器 —— Linear / ReLU / logSoftmax / MSE / momentum
// ============================================================================

#include "GpuTestCommon.h"
#include "GpuTestElementwise.h"
#include "GpuTestMatmul.h"
#include "GpuTestReduce.h"
#include "GpuTestConv.h"
#include "GpuTestModel.h"

namespace gputest {

inline int runAllGpuTests()
{
    resetFails();

    printf("\n########## GPU 算子测试 ##########\n");

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        cudaGetLastError();     // 清掉，免得污染后面的检查
        printf("没有可用的 CUDA 设备，全部跳过\n");
        printf("########## 失败 0 条（已跳过）##########\n");
        return 0;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        printf("设备 %d: %s  (sm_%d%d, %d 个 SM, %zu MB 显存)\n",
               dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
               prop.totalGlobalMem / (1024 * 1024));

    // 每次同步后的错误是 sticky 的：一旦某个 kernel 越界访问，后面所有 CUDA
    // 调用都会返回同一个错误。所以先探一次，粘住了就早点说清楚，免得后面
    // 几十条一起红、看不出真正的原因。
    cudaDeviceSynchronize();
    if (!cudaOk("启动前状态检查"))
        printf("⚠ 设备处于错误状态，后面的失败可能全都是同一个原因\n");

    runElementwiseTests();
    runMatmulTests();
    runReduceTests();
    runConvTests();
    runModelTests();

    printf("\n########## GPU 算子测试结束：失败 %d 条 ##########\n", fails());
    return fails();
}

} // namespace gputest
