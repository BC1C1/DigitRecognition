// 其它测试文件在工作区有保存，原代码备份.txt，对当前测试无影响

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include "TypeDefine.h"
// Dispatchers.h 暂时摘出编译：
//   1) 里面 3 个内核还是 double 版的 AVX 内核（_mm256_loadu_pd 收不了 float*），
//      float 化之后必然编不过 —— 它是当前 MatrixTest 唯一编不过的地方；
//   2) 当前没有任何测试调用它。它原本的使命（线程/块大小扫参、nt 与 transpose
//      对拍）已经由 MatmulTest.h 里的朴素三重循环参照接管。
//   待定：改成 float、或退回标量后重新启用，或直接删除。
//#include "Dispatchers.h"
#include "TestCommon.h"
#include "MatmulTest.h"
#include "Im2ColTest.h"
#include "Col2ImTest.h"
#include "functional"

	// 看上去希望把elementwiseMul变成matmul，elementwiseMul如果是正常矩阵，其实快于matmul
	// 然而它是切片，而且还有步长等问题，索引就完蛋了，内存墙问题严重
	// 从最基础的二维开始
	//[[1, 2, 3, 4],				   [[1, 2, 3],
	// [5, 6, 7, 8],   elementwiseMul	[4, 5, 6],  for four area
	// [9, 0, 1, 2],					[7, 8, 9]]
	// [3, 4, 5, 6]]

	// [1, 2, 3, 5, 6, 7, 9, 0, 1] eleMul [1, 2, 3, 4, 5, 6, 7, 8, 9]  ->  a(乘加和)
	// [2, 3, 4, 6, 7, 8, 0, 1, 2] eleMul [1, 2, 3, 4, 5, 6, 7, 8, 9]  ->  b(乘加和)
	// [5, 6, 7, 9, 0, 1, 3, 4, 5] eleMul [1, 2, 3, 4, 5, 6, 7, 8, 9]  ->  c(乘加和)
	// [6, 7, 8, 0, 1, 2, 4, 5, 6] eleMul [1, 2, 3, 4, 5, 6, 7, 8, 9]  ->  d(乘加和)

	//				A									B					  C

	// 把两边四个行向量各看作一个矩阵，右边的做一个转置，这就是矩阵乘法了，
	// abcd对应的其实是
	//[[a, b],
	// [c, d]]    
	// 很容易发现，卷积核拍成矩阵B这个操作是可逆的，而且列向量abcd到卷积结果abcd构成的
	// 2*2矩阵这个操作和前面这个操作完全一样
	// 如果是矩阵乘法，拿到的其实是
	//[[a, a, a, a],
	// [b, b, b, b],
	// [c, c, c, c],
	// [d, d, d, d]], 很显然重复了，所以B其实只需要是卷积核拍平以后的列向量即可。
	// 不难想到空出来的可以做什么，维度可以借此拓展，比如通道数或者数据集大小，后面再说
	// 还没解决的问题，被卷积对象如何变成矩阵A
	// 这个确实不知道怎么做比较好, 只好全量拷贝

void printMatrix(const Matrix& obj, const char* name) {
	printf_s("\n Matrix %s:\n", name);
	for (size_t i = 0; i < obj.rows(); i++) {
		for (size_t j = 0; j < obj.cols(); j++) {
			printf_s("%.1f ", obj(i, j));
		}
		printf_s("\n");
	}
}

void solution() {
	// 先构造矩阵
	Matrix im(1, 16);
	for (size_t i = 0; i < 16; i++) 
			im(0, i) = (i + 1) % 10;
	printMatrix(im, "im");
	Matrix A = im.im2col(1, 1, 4, 4, 3, 1);
	printMatrix(A, "A");
	Matrix ker(3, 3);
	for (size_t i = 0; i < 3; i++)
		for (size_t j = 0; j < 3; j++)
			ker(i, j) = i * 3 + j + 1;
	printMatrix(ker, "kernel");
	Matrix B(9, 1);
	for (size_t i = 0; i < 9; i++)
		B(i, 0) = i + 1;
	printMatrix(B, "B");
}

int main() {
	solution();

	// 三个 run*Tests() 各自在开头 resetFails()、结束时返回「自己那一段」的失败条数，
	// 所以在 main 里必须累加，直接读 dstest::fails() 只会拿到最后一段的。
	int fails = 0;

	// GEMM 内核：先测引擎，再测卷积通路
	fails += matmul_test::runMatmulTests();

	// im2col 正确性测试
	fails += im2col_test::runIm2ColTests();

	// col2im：im2col 的转置算子，反向传播要用
	fails += col2im_test::runCol2ImTests();

	printf("\n===== 全部测试结束：失败 %d 条 =====\n", fails);
	return fails;   // 全绿才是 0，交给外部（cmd / CTest）判断成败
}
