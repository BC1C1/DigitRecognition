add_executable(MatrixTest
    include/GpuTestCommon.h
    include/GpuTestElementwise.h
    include/GpuTestMatmul.h
    include/GpuTestReduce.h
    include/GpuTestConv.h
    include/GpuTestModel.h
    include/GpuTestSlice.h
    include/GpuTestData.h
    include/GpuTest.h
    src/main.cpp
)

# ---- 以下文件已摘出编译，留在目录里作参考 ----------------
#   MatmulTest.h      CPU 版 GEMM 对拍
#   Im2ColTest.h      CPU 版 im2col 测试
#   Col2ImTest.h      CPU 版 col2im 测试
#   Dispatchers.h     double 时代的 AVX 内核 + 扫参工具
#   TestCommon.h      CPU 版断言与 maxDiff
#
# 原因：这些测试的参照实现全部走 Matrix::operator()，而 operator() 现在
# 解引用显存指针，在 GPU 矩阵上调用会段错误。要恢复它们，得先让
# operator() / raw_data() 在 GPU 矩阵上有明确的失败语义（assert 或者
# 编译期不可用），再把参照实现改成显式 cudaMemcpy。

target_include_directories(MatrixTest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(MatrixTest PRIVATE Matrix)
