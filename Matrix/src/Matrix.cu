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
	fillInPlaceKernel<<<blocks, threads>>>(g, n, value);
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
	addKernel<<<blocks, threads>>>(ga, gb, gc, n);
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
	subKernel<<<blocks, threads>>>(ga, gb, gc, n);
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
	subInPlaceKernel<<<blocks, threads>>>(ga, gb, n);
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
	scaleMulKernel<<<blocks, threads>>>(ga, gb, n, value);
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
	scaleMulInPlaceKernel<<<blocks, threads>>>(g, n, value);
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
	elementwiseMulKernel<<<blocks, threads>>>(ga, gb, gc, n);
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
	elementwiseMulInPlaceKernel<<<blocks, threads>>>(ga, gb, n);
}

// matmul_nn kernel
__global__ void matmulNNKernel(float* __restrict__ ga, float* __restrict__ gb, float* __restrict__ gc,
	size_t M, size_t N, size_t K) {
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= M || j >= N) return;
	float value = 0;
	for (size_t k = 0; k < K; k++) {
		value += ga[i * K + k] * gb[k * N + j];
	}
	gc[i * N + j] = value;
}

extern "C" void launchMatmulNN(float* ga, float* gb, float* gc, size_t M, size_t N, size_t K) {
	dim3 threads = { 16, 16 };
	unsigned int bx = static_cast<unsigned int>((threads.x + N - 1) / threads.x);
	unsigned int by = static_cast<unsigned int>((threads.y + M - 1) / threads.y);
	dim3 blocks{ bx, by };
	matmulNNKernel<<<blocks, threads>>>(ga, gb, gc, M, N, K);
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
	matmulNTKernel<<<blocks, threads>>>(ga, gb, gc, M, N, K);
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
	matmulTNKernel<<<blocks, threads>>>(ga, gb, gc, M, N, K);
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
	sumRowsKernel<<<blocks, threads>>>(ga, gb, M, N);
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
	sumColsKernel<<<blocks, threads>>>(ga, gb, M, N);
}

// sum kernel
__global__ void sumKernel(float* g, size_t n, double* result) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	atomicAdd(result, g[i]);
}

extern "C" double launchSumKernel(float* g, size_t n) {
	double* result;
	cudaMalloc_throw(&result, sizeof(double));
	cudaMemset(result, 0.0, sizeof(double));
	unsigned int threads = 256;
	unsigned int blocks = (threads + n - 1) / threads;
	sumKernel<<<blocks, threads>>>(g, n, result);
	double ret;
	cudaMemcpy(&ret, result, sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(result);
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
	im2colKernel<<<blocks, threads>>>(ga, gb, N, C, H, W, OH, OW, ksize, stride, padding);
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
	col2imKernel<<<blocks, threads>>>(ga, gb, N, C, H, W, OH, OW, ksize, stride, padding);
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
	sliceRowsByIndex<<<blocks, threads>>>(ga, gb, tc, N, idx);
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
			if (tmp) cudaFree(tmp);
			cudaMalloc_throw(&tmp, need * sizeof(float));
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
	size_t N, size_t H, size_t W, const float* random_list, int maxShift) {
	unsigned int n = blockIdx.z * blockDim.z + threadIdx.z;   
	unsigned int h = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int w = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= N || h >= H || w >= W) return;
	const int dw = (int)(random_list[2 * n] * (2 * maxShift + 1)) - maxShift;
	const int dh = (int)(random_list[2 * n + 1] * (2 * maxShift + 1)) - maxShift;
	const int sh = (int)h - dh;
	const int sw = (int)w - dw;
	const bool valid = (sh >= 0 && sh < (int)H && sw >= 0 && sw < (int)W);
	gb[n * H * W + h * W + w] = valid ? ga[n * H * W + sh * W + sw] : 0.0f;
}

extern "C" void launchShiftImg(float* ga, float* gb, 
	size_t N, size_t H, size_t W, const float* random_list, int maxShift) {
	dim3 threads = { 16, 4, 4 };
	unsigned int dx = static_cast<unsigned int>((threads.x + W - 1) / threads.x);
	unsigned int dy = static_cast<unsigned int>((threads.y + H - 1) / threads.y);
	unsigned int dz = static_cast<unsigned int>((threads.z + N - 1) / threads.z);
	dim3 blocks = { dx, dy, dz };
	shiftImgKernel<<<blocks, threads>>>(ga, gb, N, H, W, random_list, maxShift);
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
	linearForwardKernel<<<blocks, threads>>>(x, W, b, result, in_dim, out_dim, N);
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
	reLUForwardKernel<<<blocks, threads>>>(x, out, N, D);
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
	reLUBackwardKernel<<<blocks, threads>>>(grad, cache, out, N, D);
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
	float value = 0;
	for (size_t k = 0; k < I * K * K; k++) {
		value += A[(n * OH * OW + i) * (I * K * K) + k] * kernel[o * (I * K * K) + k];
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
	conv2DGemmKernel<<<blocks, threads>>>(A, kernel, bias, out, N, OH, OW, I, O, K);
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
	conv2DBackwardReshapeKernel<<<blocks, threads>>>(grad, dP, N, O, OH, OW);
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
	maxPool2DForwardKernel<<<blocks, threads>>>(x, out, N, C, H, W, ksize, stride, OH, OW, maxIndex);
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
	maxPool2DBackwardKernel<<<blocks, threads>>>(grad, dX, N, C, H, W, ksize, stride, OH, OW, maxIndex);
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
	cudaMalloc_throw(&r, sizeof(double));
	cudaMemset(r, 0, sizeof(double));
	MSELossForwardKernel<<<blocks, threads>>>(pred, real, M, N, r);
	double result;
	cudaMemcpy(&result, r, sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(r);
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
	MSELossBackwardKernel<<<blocks, threads>>>(pred, real, out, M, N);
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
	logSoftmaxKernel<<<blocks, threads >>>(input, cache, out, M, N);
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
	generalUpdateLogicWithMomentum<<<blocks, threads>>>(value, grad, velocity, n, learning_rate, momentum);
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
	adamUpdateKernel<<<blocks, threads>>>(w, g, m, v, n, lr, b1, b2, y, t, inv_bc1, inv_bc2);
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
	cudaMalloc_throw(&gret, sizeof(bool) * M);
	oneHotAndCompareKernel<<<blocks, threads>>>(result, real, M, N, gret);
	bool* retVector = new bool[M];
	cudaMemcpy(retVector, gret, sizeof(bool) * M, cudaMemcpyDeviceToHost);
	cudaFree(gret);
	return retVector;
}

