#include "cuda_utils.h"
// all ptr is in gpu
// operator never malloc. Temporary assignments don't count
// default is out place, unless has "in place" suffix
// if is out place, result will fill to final operand
// if is in place, result will fill to first operand
// the operands only include matrices

// fill kernel
__global__ void fillInPlaceKernel(float* g, size_t n, float value) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	g[i] = value;
}

extern "C" void launchFillInplace(float* g, size_t n, float value) {
	if (n == 0) return;   // grid=0 的 launch 会被 CUDA 拒绝，而 launch 不返回错误码 → 静默不执行
	unsigned int threads = 256;
	unsigned int blocks = (n + threads - 1) / threads;
	fillInPlaceKernel<<<blocks, threads, 0, getStream()>>>(g, n, value);
}

// add kernel
__global__ void addKernel(float* __restrict__ ga, float* __restrict__ gb, float* __restrict__ gc, size_t n) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	gc[i] = ga[i] + gb[i];
}

extern "C" void launchAdd(float* ga, float* gb, float* gc, size_t n) {
	unsigned int threads = 256;
	unsigned int blocks = (threads + n - 1) / threads;
	addKernel<<<blocks, threads, 0, getStream()>>>(ga, gb, gc, n);
}

// sub kernel
__global__ void subKernel(float* __restrict__ ga, float* __restrict__ gb, float* __restrict__ gc, size_t n) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	gc[i] = ga[i] - gb[i];
}

extern "C" void launchSub(float* ga, float* gb, float* gc, size_t n) {
	unsigned int threads = 256;
	unsigned int blocks = (threads + n - 1) / threads;
	subKernel<<<blocks, threads, 0, getStream()>>>(ga, gb, gc, n);
}

// sub in place kernel
__global__ void subInPlaceKernel(float* __restrict__ ga, float* __restrict__ gb, size_t n) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	ga[i] -= gb[i];
}

extern "C" void launchSubInPlace(float* ga, float* gb, size_t n) {
	unsigned int threads = 256;
	unsigned int blocks = (threads + n - 1) / threads;
	subInPlaceKernel<<<blocks, threads, 0, getStream()>>>(ga, gb, n);
}

// scale mul kernel
__global__ void scaleMulKernel(float* __restrict__ ga, float* __restrict__ gb, size_t n, float value) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	gb[i] = ga[i] * value;
}

extern "C" void launchScaleMul(float* ga, float* gb, size_t n, float value) {
	unsigned int threads = 256;
	unsigned int blocks = (threads + n - 1) / threads;
	scaleMulKernel<<<blocks, threads, 0, getStream()>>>(ga, gb, n, value);
}

// scale mul in place kernel
__global__ void scaleMulInPlaceKernel(float* g, size_t n, float value) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	g[i] *= value;
}

extern "C" void launchScaleMulInPlace(float* g, size_t n, float value) {
	unsigned int threads = 256;
	unsigned int blocks = (threads + n - 1) / threads;
	scaleMulInPlaceKernel<<<blocks, threads, 0, getStream()>>>(g, n, value);
}

// elementWise mul kernel
__global__ void elementwiseMulKernel(float* __restrict__ ga, float* __restrict__ gb, float* __restrict__ gc, size_t n) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	gc[i] = ga[i] * gb[i];
}

extern "C" void launchElementwiseMul(float* ga, float* gb, float* gc, size_t n) {
	unsigned int threads = 256;
	unsigned int blocks = (threads + n - 1) / threads;
	elementwiseMulKernel<<<blocks, threads, 0, getStream()>>>(ga, gb, gc, n);
}

// elementwise mul in place kernel
__global__ void elementwiseMulInPlaceKernel(float* __restrict__ ga, float* __restrict__ gb, size_t n) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	ga[i] *= gb[i];
}

extern "C" void launchElementwiseMulInPlace(float* ga, float* gb, size_t n) {
	unsigned int threads = 256;
	unsigned int blocks = (threads + n - 1) / threads;
	elementwiseMulInPlaceKernel<<<blocks, threads, 0, getStream()>>>(ga, gb, n);
}

// matmul_nn kernel
__global__ void matmulNNKernel(float* __restrict__ ga, float* __restrict__ gb, float* __restrict__ gc,
	size_t M, size_t N, size_t K, const size_t BM, const size_t BN, const size_t BK) {
	constexpr int reg_row = 2; constexpr int reg_col = 2; constexpr int reg_K = 4;
	constexpr int ret_size = reg_row * reg_col;
	constexpr int regA_size = reg_K * reg_row;
	constexpr int regB_size = reg_K * reg_col;
	extern __shared__ float smem[];
	float* blocka = smem, * blockb = smem + BM * BK;
	float regA[regA_size], regB[regB_size], ret[ret_size];
#pragma unroll
	for (int i = 0; i < ret_size; i++) {
		ret[i] = 0.0f;
	}
	auto TK = (K + BK - 1) / BK;
	// 搬运阶段重新分配工作，我们有BM*BK+BK*BN个数据要搬运，期望每个线程尽可能参与工作
	// 我预期每个线程搬运两个，BM=BK=BN=16，一共256个线程，512个数据
	unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x; // 当前线程的编号
	unsigned int nthreads = blockDim.x * blockDim.y; // 一共有这么多
	// 给矩阵的块编号，先按行增加，再按列增加，假设ga和gb是32*32的矩阵，那么每个都有4个块
	// 从ga的第(0, 0)个块搬运到blocka，gb的第(0, 0)个块搬运到blockb，运完算加到gc的第(0, 0)个块
	// 从ga的第(0, 1)个块搬运到blocka，gb的第(1, 0)个块搬运到blockb，运完算加到gc的第(0, 0)个块，得到第1个结果
	// 从ga的第(0, 0)个块搬运到blocka，gb的第(0, 1)个块搬运到blockb，运完算加到gc的第(0, 1)个块
	// 从ga的第(0, 1)个块搬运到blocka，gb的第(1, 1)个块搬运到blockb，运完算加到gc的第(0, 1)个块，得到第2个结果
	// 从ga的第(1, 0)个块搬运到blocka，gb的第(0, 0)个块搬运到blockb，运完算加到gc的第(1, 0)个块
	// 从ga的第(1, 1)个块搬运到blocka，gb的第(1, 0)个块搬运到blockb，运完算加到gc的第(1, 0)个块，得到第3个结果
	// 从ga的第(1, 0)个块搬运到blocka，gb的第(0, 1)个块搬运到blockb，运完算加到gc的第(1, 1)个块
	// 从ga的第(1, 1)个块搬运到blocka，gb的第(0, 0)个块搬运到blockb，运完算加到gc的第(1, 1)个块，得到第4个结果
	// 从ga的第(i, k)个块搬运到blocka，gb的第(k, j)个块搬运到blockb，运完算加到gc的第(i, j)个块，得到第i * TN + j + 1个结果
	// 想办法得到这里ijk的值，我才能知道我要搬运哪些
	// 我干脆先硬编码，然后再慢慢想如何一般化
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	auto ti = blockIdx.y; 
	auto tj = blockIdx.x;
	for (int tk = 0; tk < TK; tk++) {
		// 现在只搬运
		// 比如我现在正在搬第(0, 0)个a块和第(0, 0)个b块
		// 对于我这个特定的线程，我的编号是0，我要搬的是a块的第0个和b块的第0个
		// blocka[0] = ga[0 * K + 0];
		// blockb[0] = gb[0 * N + 0];
		// 一般化，我的编号是tid，我要搬的是第tid个
		// blocka[tid] = ga[tid / BK * K + tid % BK];
		// blockb[tid] = gb[tid / BN * N + tid % BN];
		// 假如说其实我要搬运1024个，我就得跨过所有的编号去下一个
		//for (size_t idx = tid; idx < BM * BK; idx += nthreads) {
		//	blocka[idx] = ga[idx / BK * K + idx % BK];
		//}
		//for (size_t idx = tid; idx < BK * BN; idx += nthreads) {
		//	blockb[idx] = gb[idx / BN * N + idx % BN];
		//}
		// 但是我打算搬第(0, 1)个a块和第(1, 0)个b块
		//for (size_t idx = tid; idx < BM * BK; idx += nthreads) {
		//	// a块是向右走的，所以加1在行上没有影响，在列上的影响是跨越BN列 <-- 这里有错误，A的形状混淆，是BK列
		//	blocka[idx] = ga[idx / BK * K + (idx % BK + BN)];
		//}
		//for (size_t idx = tid; idx < BK * BN; idx += nthreads) {
		//	// b块是向下走的，所以加1在列上没有影响，在行上的影响是跨越BM行 <-- 同上，这里是BK行
		//	blockb[idx] = gb[(idx / BN + BM) * N + idx % BN];
		//}
		// 接着更一般化，打算搬第(0, tk)和(tk, 0)
		//for (size_t idx = tid; idx < BM * BK; idx += nthreads) {
		//	blocka[idx] = ga[(idx / BK) * K + (idx % BK + BN * tk)];
		//}
		//for (size_t idx = tid; idx < BK * BN; idx += nthreads) {
		//	blockb[idx] = gb[(idx / BN + BM * tk) * N + idx % BN];
		//}
		// 继续一般化，刚才搞定了k的问题，如何搞定i和j的问题呢
		// 其实搬到这里就足够算一次，我是说算第一个结果的第一部分
		// 这意味着i和j只是标号，用于算某个偏移，而不是循环的一部分，到这里搬运就完成了
		// 但是我必须知道i和j是怎么来的。有什么可以求得它们呢？
		// i表示A矩阵的行块号，j表示B矩阵的列块号，哦，之前全局的i和j有办法，避免混淆把块的行号叫做ti，列号叫做tj
		//unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
		//unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
		//auto ti = i / BM; auto tj = j / BN;
		//for (size_t idx = tid; idx < BM * BK; idx += nthreads) {
		//	// 这相当于给A矩阵一个向下的偏移，偏移量是ti*BM行
		//	blocka[idx] = ga[((idx / BK) * K + (ti * BM)) + (idx % BK + BN * tk)];
		//}
		//for (size_t idx = tid; idx < BK * BN; idx += nthreads) {
		//	// 同理，给B矩阵一个向右的偏移，偏移量是ti*BN列 <-- 错误，应该是tj
		//	blockb[idx] = gb[(idx / BN + BM * tk) * N + (idx % BN + ti * BN)];
		//}
		// 更好的办法是人为规定一个块由一个block负责
		//auto ti = blockIdx.y; auto tj = blockIdx.x;
		//for (size_t idx = tid; idx < BM * BK; idx += nthreads) {
		//	blocka[idx] = ga[((idx / BK + (ti * BM)) * K ) + (idx % BK + BN * tk)];
		//}
		//for (size_t idx = tid; idx < BK * BN; idx += nthreads) {
		//	blockb[idx] = gb[(idx / BN + BM * tk) * N + (idx % BN + ti * BN)];
		//}
		// 这样就好了吗？总觉得可能没有这么规整，可能越界，因为最后的块可能不完整
		// cpu版本上用stdmin来解决，这里填0

		for (int idx = tid; idx < BM * BK; idx += nthreads) {
			auto r = idx / BK + ti * BM;
			auto c = idx % BK + BK * tk;
			blocka[idx] = (r < M && c < K) ? ga[r * K + c] : 0.0f;
		}
		for (int idx = tid; idx < BK * BN; idx += nthreads) {
			auto r = idx / BN + BK * tk;
			auto c = idx % BN + tj * BN;
			blockb[idx] = (r < K && c < N) ? gb[r * N + c] : 0.0f;
		}
		__syncthreads();
		// 接着是计算
		//for (int k = 0; k < BK; k++) {
		//	value += blocka[ty * BK + k] * blockb[k * BN + tx];
		//}
		// 算更多值有助于减少从共享内存搬运，也可以理解成复用加载
		// 考虑这样一个例子，共享内存里有两个4*4的矩阵
		// 假如我一个线程只算一个数据，那么每次要从共享内存搬运4+4=8个数据，有16个输出，那么就要搬运16*8=128次
		// 那么我计划在blocka搬运两行，然后在blockb搬运一列，而且blocka的两行在共享内存是连续的，编译器有可能优化成
		// 类似同时加载的指令，这样稍微快一点。两个输出也是竖着的一组。搬运了12个数据，却有2个输出，一共12*8=96次
		// 发现搬运减少。
		//float regA[8], regB[4], ret[2];
		//for (int ri = 0; ri < 8; ri++)
		//	regA[ri] = blocka[ri];
		//for (int rj = 0; rj < 4; rj++)
		//	regB[rj] = blockb[rj * BN];
		//for (int rt = 0; rt < 2; rt++) {
		//	ret[rt] = 0;
		//}
		//for (int ci = 0; ci < 2; ci++) {
		//	for (int k = 0; k < 4; k++) {
		//		ret[ci] += regA[ci * 4 + k] * regB[k];
		//	}
		//}
		// 其实不用太苛刻的要求连续加载，从共享内存加载已经足够快。更关键的是有多少寄存器可以用，这里没有试验过
		// 无论怎样，稍微增大都是合理的，让我增大到4*4，也就是搬运4行A，4列B，输出16个值
		// 深度思考中...(还是非常有喜感的)
		// 已思考（耗时60分钟）
		auto RK = (BK + reg_K - 1) / reg_K;
#pragma unroll
		for (int rk = 0; rk < RK; rk++) {
			// load to registers
#pragma unroll
			for (int ri = 0; ri < regA_size; ri++) {
				auto r = ri / reg_K + threadIdx.y * reg_row;
				auto c = ri % reg_K + rk * reg_K;
				regA[ri] = (r < BM && c < BK) ? blocka[r * BK + c] : 0.0f;
			}
#pragma unroll
			for (int rj = 0; rj < regB_size; rj++) {
				auto r = rj % reg_K + rk * reg_K;
				auto c = rj / reg_K + threadIdx.x * reg_col;
				regB[rj] = (r < BK && c < BN) ? blockb[r * BN + c] : 0.0f;
			}
			// calculate
#pragma unroll
			for (int ci = 0; ci < reg_row; ci++) {
#pragma unroll
				for (int cj = 0; cj < reg_col; cj++) {
#pragma unroll
					for (int k = 0; k < reg_K; k++) {
						ret[ci * reg_col + cj] += regA[ci * reg_K + k] * regB[cj * reg_K + k];
					}
				}
			}
		}

		__syncthreads();
	}
#pragma unroll
	for (int ci = 0; ci < reg_row; ci++)
#pragma unroll
		for (int cj = 0; cj < reg_col; cj++) {
			int row = ti * BM + threadIdx.y * reg_row + ci;
			int col = tj * BN + threadIdx.x * reg_col + cj;
			if (row < M && col < N)
				gc[row * N + col] = ret[ci * reg_col + cj];
		}
}

extern "C" void launchMatmulNN(float* ga, float* gb, float* gc, size_t M, size_t N, size_t K,
	size_t BM, size_t BN, size_t BK) {
	constexpr unsigned kRegRow = 2, kRegCol = 2;
	dim3 threads = { (unsigned)BN / kRegCol, (unsigned)BM / kRegRow };
	auto sharedSize = (BM * BK + BK * BN) * sizeof(float);
	dim3 blocks{ (unsigned)((N + BN - 1) / BN), (unsigned)((M + BM - 1) / BM) };
	matmulNNKernel<<<blocks, threads, sharedSize, getStream()>>>(ga, gb, gc, M, N, K, BM, BN, BK);
}

__global__ void matmulNTKernel(float* __restrict__ ga, float* __restrict__ gb, float* __restrict__ gc,
	size_t M, size_t N, size_t K) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= M || j >= N) return;
	// ga is (M, K), gb is (N, K), gc is (M, N)
	float value = 0;
	for (size_t k = 0; k < K; k++) {
		value += ga[i * K + k] * gb[j * K + k];
	}
	gc[i * N + j] = value;
}

extern "C" void launchMatmulNT(float* ga, float* gb, float* gc, size_t M, size_t N, size_t K) {
	dim3 threads = { 16, 16 };
	dim3 blocks = {
		static_cast<unsigned int>((threads.x + N - 1) / threads.x),
		static_cast<unsigned int>((threads.y + M - 1) / threads.y)
	};
	matmulNTKernel<<<blocks, threads, 0, getStream()>>>(ga, gb, gc, M, N, K);
}

// matmul_tn kernel
__global__ void matmulTNKernel(float* __restrict__ ga, float* __restrict__ gb, float* __restrict__ gc,
	size_t M, size_t N, size_t K) {
	// ga is (K, M), gb is (K, N), gc is (M, N)
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= M || j >= N) return;
	float value = 0;
	for (size_t k = 0; k < K; k++) {
		value += ga[k * M + i] * gb[k * N + j];
	}
	gc[i * N + j] = value;
}

extern "C" void launchMatmulTN(float* ga, float* gb, float* gc, size_t M, size_t N, size_t K) {
	dim3 threads = { 16, 16 };
	dim3 blocks = {
		static_cast<unsigned int>((threads.x + N - 1) / threads.x),
		static_cast<unsigned int>((threads.y + M - 1) / threads.y)
	};
	matmulTNKernel<<<blocks, threads, 0, getStream()>>>(ga, gb, gc, M, N, K);
}

// sum rows kernel, get Column vector
__global__ void sumRowsKernel(float* __restrict__ ga, float* __restrict__ gb, size_t M, size_t N) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= M) return;
	float value = 0;
	for (size_t j = 0; j < N; j++)
		value += ga[i * N + j];
	gb[i] = value;
}

extern "C" void launchSumRows(float* ga, float* gb, size_t M, size_t N) {
	unsigned int threads = 256;
	unsigned int blocks = (threads + M - 1) / threads;
	sumRowsKernel<<<blocks, threads, 0, getStream()>>>(ga, gb, M, N);
}

// sum cols kernel, get row vector
__global__ void sumColsKernel(float* __restrict__ ga, float* __restrict__ gb, size_t M, size_t N) {
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= N) return;
	float value = 0;
	for (size_t i = 0; i < M; i++)
		value += ga[i * N + j];
	gb[j] = value;
}

extern "C" void launchSumCols(float* ga, float* gb, size_t M, size_t N) {
	unsigned int threads = 256;
	unsigned int blocks = (threads + N - 1) / threads;
	sumColsKernel<<<blocks, threads, 0, getStream()>>>(ga, gb, M, N);
}

// sum kernel
__global__ void sumKernel(float* g, size_t n, double* result) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	atomicAdd(result, g[i]);
}

extern "C" double launchSumKernel(float* g, size_t n) {
	double* result;
	cudaMallocAsync_throw(&result, sizeof(double));
	cudaMemset(result, 0.0, sizeof(double));
	unsigned int threads = 256;
	unsigned int blocks = (threads + n - 1) / threads;
	sumKernel<<<blocks, threads, 0, getStream()>>>(g, n, result);
	double ret;
	cudaMemcpy(&ret, result, sizeof(double), cudaMemcpyDeviceToHost);
	cudaFreeAsync_throw(result);
	return ret;
}

// im2col kernel
__global__ void im2colKernel(float* __restrict__ ga, float* __restrict__ gb,
	size_t N, size_t C, size_t H, size_t W, size_t OH, size_t OW,
	size_t ksize, size_t stride, size_t padding) {
	// ga is (N, (C H W)), gb is ((N OH OW), (C K K))
	unsigned int n = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= N || i >= OH * OW || c >= C) return;
	size_t oh = i / OW;
	size_t ow = i % OW;
	size_t headh = oh * stride;
	size_t headw = ow * stride;
	size_t k2 = ksize * ksize;
	size_t Ck2 = k2 * C;
	size_t OWCk2 = Ck2 * OW;
	size_t OHOWCk2 = OWCk2 * OH;
	size_t HW = H * W;
	size_t CHW = HW * C;
	for (size_t h = 0; h < ksize; h++) {
		size_t curh = headh + h;
		size_t th = curh - padding;
		for (size_t w = 0; w < ksize; w++) {
			size_t curw = headw + w;
			size_t tw = curw - padding;
			bool is_padding_area = (curh < padding || th >= H) || (curw < padding || tw >= W);
			size_t bidx = w + h * ksize + c * k2 + ow * Ck2 + oh * OWCk2 + n * OHOWCk2;
			gb[bidx] = is_padding_area ? 0.0f : ga[tw + th * W + c * HW + n * CHW];
		}
	}
}

extern "C" void launchIm2col(float* ga, float* gb, size_t N, size_t C, size_t H, size_t W, size_t OH, size_t OW,
	size_t ksize, size_t stride, size_t padding) {
	dim3 threads = { 4, 16, 4 };
	dim3 blocks = {
		static_cast<unsigned int>((threads.x + C - 1) / threads.x),
		static_cast<unsigned int>((threads.y + OH * OW - 1) / threads.y),
		static_cast<unsigned int>((threads.z + N - 1) / threads.z)
	};
	im2colKernel<<<blocks, threads, 0, getStream()>>>(ga, gb, N, C, H, W, OH, OW, ksize, stride, padding);
}

// col2im kernel: gb must be zero matrix
__global__ void col2imKernel(float* __restrict__ ga, float* __restrict__ gb,
	size_t N, size_t C, size_t H, size_t W, size_t OH, size_t OW,
	size_t ksize, size_t stride, size_t padding) {
	// ga is ((N OH OW), (C K K)), gb is (N, (C H W))
	unsigned int n = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= N || i >= OH * OW || c >= C) return;
	size_t oh = i / OW;
	size_t ow = i % OW;
	size_t headh = oh * stride;
	size_t headw = ow * stride;
	size_t k2 = ksize * ksize;
	size_t Ck2 = k2 * C;
	size_t OWCk2 = Ck2 * OW;
	size_t OHOWCk2 = OWCk2 * OH;
	size_t HW = H * W;
	size_t CHW = HW * C;
	for (size_t h = 0; h < ksize; h++) {
		size_t curh = headh + h;
		size_t th = curh - padding;
		for (size_t w = 0; w < ksize; w++) {
			size_t curw = headw + w;
			size_t tw = curw - padding;
			bool is_padding_area = (curh < padding || th >= H) || (curw < padding || tw >= W);
			size_t aidx = w + h * ksize + c * k2 + ow * Ck2 + oh * OWCk2 + n * OHOWCk2;
			auto value = is_padding_area ? 0.0f : ga[aidx];
			auto stw = is_padding_area ? 0 : tw;
			auto sth = is_padding_area ? 0 : th;
			atomicAdd(&gb[stw + sth * W + c * HW + n * CHW], value);
		}
	}
}

extern "C" void launchCol2im(float* ga, float* gb, size_t N, size_t C, size_t H, size_t W, size_t OH, size_t OW,
	size_t ksize, size_t stride, size_t padding) {
	dim3 threads = { 4, 16, 4 };
	dim3 blocks = {
		static_cast<unsigned int>((threads.x + C - 1) / threads.x),
		static_cast<unsigned int>((threads.y + OH * OW - 1) / threads.y),
		static_cast<unsigned int>((threads.z + N - 1) / threads.z)
	};
	col2imKernel<<<blocks, threads, 0, getStream()>>>(ga, gb, N, C, H, W, OH, OW, ksize, stride, padding);
}

// slice rows by index
__global__ void sliceRowsByIndex(float* __restrict__ ga, float* __restrict__ gb, 
	size_t count, size_t N, const size_t* index) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count || j >= N) return;
	// ga is (M, N), gb is (count, N)
	gb[i * N + j] = ga[index[i] * N + j];
}

extern "C" void launchSliceRowsByIndex(float* ga, float* gb, size_t M, size_t N, size_t count, const size_t* idx) {
	dim3 threads = { 16, 16 };
	size_t tc = min(count, M);
	unsigned int bx = static_cast<unsigned int>((threads.x + N - 1) / threads.x);
	unsigned int by = static_cast<unsigned int>((threads.y + tc - 1) / threads.y);
	dim3 blocks{ bx, by };
	sliceRowsByIndex<<<blocks, threads, 0, getStream()>>>(ga, gb, tc, N, idx);
}

// random
extern "C" void launchFillNormal(float* dst, size_t n, float mean, float scale) {
	if (n == 0) return;
	auto gen = getRandnGenerator();
	if (n % 2 == 0) {
		curandGenerateNormal(gen, dst, n, mean, scale);
	}
	else {
		static float* tmp = nullptr;
		static size_t tmp_cap = 0;
		size_t need = n + 1;
		if (need > tmp_cap) {
			if (tmp) cudaFreeAsync_throw(tmp);
			cudaMallocAsync_throw(&tmp, need * sizeof(float));
			tmp_cap = need;
		}
		curandGenerateNormal(gen, tmp, need, mean, scale);
		cudaMemcpy(dst, tmp, n * sizeof(float), cudaMemcpyDeviceToDevice);
	}
}

extern "C" void launchFillUniform(float* dst, size_t n) {
	if (n == 0) return;
	auto gen = getRandnGenerator();
	curandGenerateUniform(gen, dst, n);
}

// shift kernel
__global__ void shiftImgKernel(float* __restrict__ ga, float* __restrict__ gb,
	size_t N, size_t C, size_t H, size_t W, const float* random_list, int maxShift) {
	unsigned int n = blockIdx.z * blockDim.z + threadIdx.z;   
	unsigned int h = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int w = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= N || h >= H || w >= W) return;
	const int dw = (int)(random_list[2 * n] * (2 * maxShift + 1)) - maxShift;
	const int dh = (int)(random_list[2 * n + 1] * (2 * maxShift + 1)) - maxShift;
	const int sh = (int)h - dh;
	const int sw = (int)w - dw;
	const bool valid = (sh >= 0 && sh < (int)H && sw >= 0 && sw < (int)W);
	for (size_t c = 0; c < C; c++)
		gb[n * H * W * C + c * H * W + h * W + w] = valid ? ga[n * H * W * C + c * H * W + sh * W + sw] : 0.0f;
}

extern "C" void launchShiftImg(float* ga, float* gb, 
	size_t N, size_t C, size_t H, size_t W, const float* random_list, int maxShift) {
	dim3 threads = { 16, 4, 4 };
	unsigned int dx = static_cast<unsigned int>((threads.x + W - 1) / threads.x);
	unsigned int dy = static_cast<unsigned int>((threads.y + H - 1) / threads.y);
	unsigned int dz = static_cast<unsigned int>((threads.z + N - 1) / threads.z);
	dim3 blocks = { dx, dy, dz };
	shiftImgKernel<<<blocks, threads, 0, getStream()>>>(ga, gb, N, C, H, W, random_list, maxShift);
}

// linear forward kernel
__global__ void linearForwardKernel(float* __restrict__ x, float* __restrict__ W, 
	float* __restrict__ b, float* __restrict__ result,
	size_t in_dim, size_t out_dim, size_t N) {
	// x is (N, in), W is (out, in), b is (1, out), result is (N, out)
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || j >= out_dim) return;
	float value = 0;
	for (size_t k = 0; k < in_dim; k++) {
		value += x[i * in_dim + k] * W[j * in_dim + k];
	}
	value += b[j];
	result[i * out_dim + j] = value;
}

extern "C" void launchLinearForward(float* x, float* W, float* b, float* result,
	size_t in_dim, size_t out_dim, size_t N) {
	dim3 threads = { 16, 16 };
	unsigned int bx = static_cast<unsigned int>((threads.x + out_dim - 1) / threads.x);
	unsigned int by = static_cast<unsigned int>((threads.y + N - 1) / threads.y);
	dim3 blocks{ bx, by };
	linearForwardKernel<<<blocks, threads, 0, getStream()>>>(x, W, b, result, in_dim, out_dim, N);
}

// ReLU forward kernel
__global__ void reLUForwardKernel(float* __restrict__ x, float* __restrict__ out, size_t N, size_t D) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || j >= D) return;
	out[i * D + j] = fmax(0.0f, x[i * D + j]);
}

extern "C" void launchReLUForward(float* x, float* out, size_t N, size_t D) {
	dim3 threads = { 16, 16 };
	dim3 blocks = {
		static_cast<unsigned int>((threads.x + D - 1) / threads.x),
		static_cast<unsigned int>((threads.y + N - 1) / threads.y)
	};
	reLUForwardKernel<<<blocks, threads, 0, getStream()>>>(x, out, N, D);
}

// ReLU backward kernel
__global__ void reLUBackwardKernel(float* __restrict__ grad, float* __restrict__ cache,
	float* __restrict__ out, size_t N, size_t D) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N || j >= D) return;
	out[i * D + j] = cache[i * D + j] == 0 ? 0 : grad[i * D + j];
}

extern "C" void launchReLUBackward(float* grad, float* cache, float* out, size_t N, size_t D) {
	dim3 threads = { 16, 16 };
	dim3 blocks = {
		static_cast<unsigned int>((threads.x + D - 1) / threads.x),
		static_cast<unsigned int>((threads.y + N - 1) / threads.y)
	};
	reLUBackwardKernel<<<blocks, threads, 0, getStream()>>>(grad, cache, out, N, D);
}

// conv2D forward gemm kernel
__global__ void conv2DGemmKernel(float* __restrict__ A, float* __restrict__ kernel,
	float* __restrict__ bias, float* __restrict__ out,
	size_t N, size_t OH, size_t OW, size_t I, size_t O, size_t K) {
	// A is ((N OH OW), (I K K)), kernel is (O, (I K K)), bias is (1, O), 
	// out is ((N OH OW), O) -> (N, (O OH OW))
	unsigned int o = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (o >= O || i >= OH * OW || n >= N) return;

	const size_t IKK = I * K * K;                          // 内层长度
	const float* Av = A + (n * OH * OW + i) * IKK;         // 本行的 A
	const float* Kv = kernel + o * IKK;                    // 本行的权重

	float value = 0.0f;

	if (IKK % 4 == 0) {
		const size_t IKK4 = IKK / 4;
		float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);      // 4 个独立累加器，顺便打断依赖链
		for (size_t k = 0; k < IKK4; ++k) {
			const float4 a = *reinterpret_cast<const float4*>(Av + k * 4);
			const float4 b = *reinterpret_cast<const float4*>(Kv + k * 4);
			acc.x = fmaf(a.x, b.x, acc.x);
			acc.y = fmaf(a.y, b.y, acc.y);
			acc.z = fmaf(a.z, b.z, acc.z);
			acc.w = fmaf(a.w, b.w, acc.w);
		}
		value = (acc.x + acc.y) + (acc.z + acc.w);          // 树形归约
	}
	else {
		for (size_t k = 0; k < IKK; ++k)
			value += Av[k] * Kv[k];
	}

	value += bias[o];
	out[i + o * OH * OW + n * OH * OW * O] = value;
}

extern "C" void launchConv2DGemmKernel(float* A, float* kernel, float* bias, float* out,
	size_t N, size_t OH, size_t OW, size_t I, size_t O, size_t K) {
	auto threads = dim3{ 4, 16, 4 };
	unsigned int dx = static_cast<unsigned int>((threads.x + N - 1) / threads.x);
	unsigned int dy = static_cast<unsigned int>((threads.y + OH * OW - 1) / threads.y);
	unsigned int dz = static_cast<unsigned int>((threads.z + O - 1) / threads.z);
	dim3 blocks = { dx, dy, dz };
	conv2DGemmKernel<<<blocks, threads, 0, getStream()>>>(A, kernel, bias, out, N, OH, OW, I, O, K);
}

// conv2D backward reshape kernel
__global__ void conv2DBackwardReshapeKernel(float* __restrict__ grad, float* __restrict__ dP,
	size_t N, size_t O, size_t OH, size_t OW) {
	// grad is (N (O OH OW)), dP is ((N OH OW) O)
	unsigned int n = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int o = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= N || o >= O || i >= OH * OW) return;
	dP[o + i * O + n * O * OH * OW] = grad[i + o * OH * OW + n * OH * OW * O];
}

extern "C" void launchConv2DBackwardReshape(float* grad, float* dP,
	size_t N, size_t O, size_t OH, size_t OW) {
	auto threads = dim3{ 4, 16, 4 };
	unsigned int dx = static_cast<unsigned int>((threads.x + O - 1) / threads.x);
	unsigned int dy = static_cast<unsigned int>((threads.y + OH * OW - 1) / threads.y);
	unsigned int dz = static_cast<unsigned int>((threads.z + N - 1) / threads.z);
	dim3 blocks = { dx, dy, dz };
	conv2DBackwardReshapeKernel<<<blocks, threads, 0, getStream()>>>(grad, dP, N, O, OH, OW);
}

// maxpool2D forward kernel
__global__ void maxPool2DForwardKernel(float* __restrict__ x, float* __restrict__ out, 
	size_t N, size_t C, size_t H, size_t W,
	size_t ksize, size_t stride, size_t OH, size_t OW, size_t* maxIndex) {
	// x is (N, (C, H, W)) 
	unsigned int n = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int c = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= N || c >= C || i >= OH * OW) return;
	size_t oh = i / OW;
	size_t ow = i % OW;
	size_t headh = oh * stride;
	size_t headw = ow * stride;
	size_t maxIdx = headw + headh * W + c * H * W + n * H * W * C;
	float mvalue = x[maxIdx];
	for (size_t h = 0; h < ksize; h++) {
		for (size_t w = 0; w < ksize; w++) {
			size_t curh = headh + h;
			size_t curw = headw + w;
			size_t idx = curw + curh * W + c * H * W + n * H * W * C;
			if (x[idx] > mvalue) {
				maxIdx = idx;
				mvalue = x[idx];
			}
		}
	}
	out[ow + oh * OW + c * OH * OW + n * OH * OW * C] = mvalue;
	maxIndex[ow + oh * OW + c * OH * OW + n * C * OH * OW] = maxIdx;
}

extern "C" void launchMaxPool2DForward(float* x, float* out, size_t N, size_t C, size_t H, size_t W,
	size_t ksize, size_t stride, size_t* maxIndex) {
	auto threads = dim3{ 16, 4, 4 };
	size_t OH = (H - ksize) / stride + 1;
	size_t OW = (W - ksize) / stride + 1;
	unsigned int dx = static_cast<unsigned int>((threads.x + OH * OW - 1) / threads.x);
	unsigned int dy = static_cast<unsigned int>((threads.y + C - 1) / threads.y);
	unsigned int dz = static_cast<unsigned int>((threads.z + N - 1) / threads.z);
	dim3 blocks = { dx, dy, dz };
	maxPool2DForwardKernel<<<blocks, threads, 0, getStream()>>>(x, out, N, C, H, W, ksize, stride, OH, OW, maxIndex);
}

// maxpool2D backward kernel: dx must be zero matrix
__global__ void maxPool2DBackwardKernel(float* __restrict__ grad, float* __restrict__ dx,
	size_t N, size_t C, size_t H, size_t W,
	size_t ksize, size_t stride, size_t OH, size_t OW, size_t* maxIndex) {
	// grad is (N, C, OH, OW), dx is (N, C, H, W)
	unsigned int n = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int c = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= N || c >= C || i >= OH * OW) return;
	size_t oh = i / OW;
	size_t ow = i % OW;
	size_t idx = ow + oh * OW + c * OH * OW + n * C * OH * OW;
	atomicAdd(&dx[maxIndex[idx]], grad[idx]);
}

extern "C" void launchMaxPool2DBackward(float* grad, float* dX, size_t N, size_t C, size_t H, size_t W,
	size_t ksize, size_t stride, size_t* maxIndex) {
	auto threads = dim3{ 16, 4, 4 };
	size_t OH = (H - ksize) / stride + 1;
	size_t OW = (W - ksize) / stride + 1;
	unsigned int dx = static_cast<unsigned int>((threads.x + OH * OW - 1) / threads.x);
	unsigned int dy = static_cast<unsigned int>((threads.y + C - 1) / threads.y);
	unsigned int dz = static_cast<unsigned int>((threads.z + N - 1) / threads.z);
	dim3 blocks = { dx, dy, dz };
	maxPool2DBackwardKernel<<<blocks, threads, 0, getStream()>>>(grad, dX, N, C, H, W, ksize, stride, OH, OW, maxIndex);
}

// MSE lossfunction forward kernel
__global__ void MSELossForwardKernel(float* __restrict__ pred, float* __restrict__ real,
	size_t M, size_t N, double* loss) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= M || j >= N) return;
	float value = (pred[i * N + j] - real[i * N + j]);
	value = value * value / (M * N);
	atomicAdd(loss, value);
}

extern "C" double launchMSELossForward(float* pred, float* real, size_t M, size_t N) {
	dim3 threads{ 16, 16 };
	dim3 blocks{
		static_cast<unsigned int>((threads.x + N - 1) / threads.x),
		static_cast<unsigned int>((threads.y + M - 1) / threads.y)
	};
	double* r;
	cudaMallocAsync_throw(&r, sizeof(double));
	cudaMemset(r, 0, sizeof(double));
	MSELossForwardKernel<<<blocks, threads, 0, getStream()>>>(pred, real, M, N, r);
	double result;
	cudaMemcpy(&result, r, sizeof(double), cudaMemcpyDeviceToHost);
	cudaFreeAsync_throw(r);
	return result;
}

// MSE lossFunction backward kernel
__global__ void MSELossBackwardKernel(float* __restrict__ pred, float* __restrict__ real,
	float* __restrict__ out, size_t M, size_t N) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= M || j >= N) return;
	float value = (pred[i * N + j] - real[i * N + j]);
	out[i * N + j] = value * 2 / (M * N);
}

extern "C" void launchMSELossBackwardKernel(float* pred, float* real, float* out, size_t M, size_t N) {
	dim3 threads{ 16, 16 };
	dim3 blocks{
		static_cast<unsigned int>((threads.x + N - 1) / threads.x),
		static_cast<unsigned int>((threads.y + M - 1) / threads.y)
	};
	MSELossBackwardKernel<<<blocks, threads, 0, getStream()>>>(pred, real, out, M, N);
}

// log-softmax kernel
__global__ void logSoftmaxKernel(float* __restrict__ input, float* __restrict__ cache,
	float* __restrict__ out, size_t M, size_t N) {
	unsigned int l = blockIdx.x * blockDim.x + threadIdx.x;
	if (l >= M) return;
	float maxValue = input[l * N];
	for (size_t i = 0; i < N; i++) {
		maxValue = max(maxValue, input[l * N + i]);
	}
	float e_s = 0;
	for (size_t i = 0; i < N; i++) {
		e_s += exp(input[l * N + i] - maxValue);
	}
	auto log_e_s = log(e_s);
	for (size_t i = 0; i < N; i++) {
		out[l * N + i] = (input[l * N + i] - maxValue - log_e_s);
		cache[l * N + i] = exp(input[l * N + i] - maxValue) / e_s;
	}
}

extern "C" void launchLogSoftmax(float* input, float* cache, float* out, size_t M, size_t N) {
	unsigned int threads = 256;
	unsigned int blocks = (threads + M - 1) / threads;
	logSoftmaxKernel<<<blocks, threads, 0, getStream()>>>(input, cache, out, M, N);
}

// General update logic with momentum
__global__ void generalUpdateLogicWithMomentum(float* __restrict__ value, 
	float* __restrict__ grad, float* __restrict__ velocity,
	size_t n, float learning_rate, float momentum) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	velocity[i] = velocity[i] * momentum + grad[i];
	value[i] -= velocity[i] * learning_rate;
}

extern "C" void launchGeneralUpdateWithMomentum(float* value, float* grad, float* velocity,
	size_t n, float learning_rate, float momentum) {
	unsigned int threads = 256;
	unsigned int blocks = (threads + n - 1) / threads;
	generalUpdateLogicWithMomentum<<<blocks, threads, 0, getStream()>>>(value, grad, velocity, n, learning_rate, momentum);
}

// Adam update kernel
__global__ void adamUpdateKernel(float* w, float* g, float* m, float* v, 
	size_t n, float lr, float b1, float b2, float y, int t, float inv_bc1, float inv_bc2) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	// 每个参数
	m[i] = b1 * m[i] + (1 - b1) * g[i];
	v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
	auto m_hat = m[i] * inv_bc1;
	auto v_hat = v[i] * inv_bc2;
	w[i] -= lr * m_hat / (sqrtf(v_hat) + y);
}

extern "C" void launchAdamUpdateKernel(float* w, float* g, float* m, float* v, size_t n,
	float lr, float b1, float b2, float y, int t, float inv_bc1, float inv_bc2) {
	unsigned int threads = 256;
	unsigned int blocks = (n + threads - 1) / threads;
	adamUpdateKernel<<<blocks, threads, 0, getStream()>>>(w, g, m, v, n, lr, b1, b2, y, t, inv_bc1, inv_bc2);
}

// one hot and 
__global__ void oneHotAndCompareKernel(float* __restrict__ result, float* __restrict__ real,
	size_t M, size_t N, bool* retVector) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= M) return;
	float cache = result[i * N];
	size_t pred_idx = 0;
	size_t real_idx = 0;
	for (size_t j = 0; j < N; j++) {
		bool ib = result[i * N + j] > cache;
		cache = max(cache, result[i * N + j]);
		pred_idx = ib ? j : pred_idx;
		real_idx = real[i * N + j] == 1 ? j : real_idx;
	}
	retVector[i] = real_idx == pred_idx;
}

extern "C" bool* launchOneHotAndCompare(float* result, float* real, size_t M, size_t N) {
	unsigned int threads = 256;
	unsigned int blocks = (threads + M - 1) / threads;
	bool* gret;
	cudaMallocAsync_throw(&gret, sizeof(bool) * M);
	oneHotAndCompareKernel<<<blocks, threads>>>(result, real, M, N, gret);
	bool* retVector = new bool[M];
	cudaMemcpy(retVector, gret, sizeof(bool) * M, cudaMemcpyDeviceToHost);
	cudaFreeAsync_throw(gret);
	return retVector;
}

