// ============================================================================
//  MatrixTest 入口
//
//  这里只做一件事：跑 GPU 算子测试并返回失败条数。
//
//  原来这里还有 solution()（手推 im2col 的小例子）和三个 CPU 测试的调用。
//  它们全部依赖 Matrix::operator()，而 operator() 现在解引用显存指针 ——
//  在 GPU 矩阵上调它会段错误。CPU 版的那几个测试文件（MatmulTest /
//  Im2ColTest / Col2ImTest / Dispatchers）已经从 CMakeLists 里摘出编译，
//  文件本身留在目录里当参考。
//
//  退出码 = 失败条数（全绿为 0），可以直接挂 CTest 或给 CI 用。
// ============================================================================

#include <cstdio>

#include "TypeDefine.h"
#include "GpuTest.h"

int main()
{
    const int fails = gputest::runAllGpuTests();

    printf("\n===== 全部测试结束：失败 %d 条 =====\n", fails);
    return fails;
}
