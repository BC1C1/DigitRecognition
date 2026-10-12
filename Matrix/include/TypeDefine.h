#pragma once
#include <cmath>
#include <random>
#include <cassert>
#include <vector>
#include <immintrin.h>
#include <omp.h>
#include <curand.h>
#include <exception>

#include "cuda_bridge.h"
#include "cuda_utils.h"

inline float randn()
{
	static std::mt19937 rng(42);
	static std::normal_distribution<float> dist(0.0, 1.0);
	return dist(rng);
}

// 默认gpu，提供toHost
struct MatrixData
{
	float* data;
	float* hostData;
	size_t row;
	size_t col;
	size_t ref_cnt;
	MatrixData(size_t row, size_t col) : row(row), col(col), hostData(nullptr), data(nullptr), ref_cnt(1) {
		if (row == 0 || col == 0) data = nullptr;
		else
		{
			size_t bsize = sizeof(float) * row * col;
			cudaMalloc_throw(&data, bsize);
		}
	}
	MatrixData(const MatrixData&) = delete;
	MatrixData& operator=(const MatrixData&) = delete;
	~MatrixData() { 
		if (data) { 
			cudaFree(data); 
			// data = nullptr; 
		}
		if (hostData) { 
			delete[] hostData; 
			// hostData = nullptr; 
		}
	} 
	//MatrixData* copy() const {
	//	float* newData = nullptr;
	//	size_t bsize = sizeof(float) * row * col;
	//	if (bsize != 0) {
	//		auto err = cudaMalloc(&newData, bsize);
	//		if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
	//		cudaMemcpy(newData, data, bsize, cudaMemcpyDeviceToDevice);
	//	}
	//	return new MatrixData(newData, row, col);
	//}
	void toHost() {
		size_t bsize = sizeof(float) * row * col;
		if (!data) { hostData = nullptr; return; }
		if (!hostData) hostData = new float[row * col];
		cudaMemcpy(hostData, data, bsize, cudaMemcpyDeviceToHost);
	}

	float* data_gpu() const { return data; }
	float* data_cpu() const { return hostData; }
	bool isHost() const { return hostData != nullptr; }
	bool isValid() const { return data != nullptr; }
//private:
//	MatrixData(float* data, size_t row, size_t col) : row(row), col(col), data(data), hostData(nullptr),
//	ref_cnt(1) {}
};
class Matrix {
public:
	Matrix() : data(new MatrixData(0, 0)) {}
	Matrix(size_t row, size_t col) : data(new MatrixData(row, col)) {}
	~Matrix() { deref(); }
	Matrix(const Matrix& other) : data(other.data) {
		ref();
	}
	Matrix(Matrix&& other) noexcept : data(other.data) {
		other.data = nullptr;
	}
	Matrix& operator=(const Matrix& other) {
		if (&other == this)
			return *this;
		this->deref();
		data = other.data;
		this->ref();
		return *this;
	}
	Matrix& operator=(Matrix&& other) noexcept {
		if (&other == this)
			return *this;
		this->deref();
		data = other.data;
		other.data = nullptr;
		return *this;
	}
private:
	inline void ref() const {
		if (data)
			data->ref_cnt++;
	}

	inline void deref() {
		if (!data) return;
		data->ref_cnt--;
		if (data->ref_cnt == 0)
		{
			delete data;
		}
	}
public:
	// meta
	size_t rows() const {
		return data->row;
	}
	size_t cols() const {
		return data->col;
	}
	size_t size() const {
		return data->row * data->col;
	}
	float* data_gpu() const {
		return data->data_gpu();
	}
	float* data_cpu() const {
		data->toHost();
		return data->data_cpu();
	}

	// factory
	static Matrix zeroMatrix(size_t row, size_t col) {
		Matrix ret(row, col);
		launchFillInplace(ret.data_gpu(), ret.size(), 0.0f);
		return ret;
	}
	static Matrix randnMatrix(size_t row, size_t col, float scale = 1.0) {
		Matrix ret(row, col);
		launchFillNormal(ret.data_gpu(), ret.size(), 0.0f, scale);
		return ret;
	}
	
	// 以下所有方法暂时没有调用者，暂时弃用。

	float* raw_data() const {
		return data->data;
	}
	// index--只操作host的，是一个很重的操作，而且修改的都是副本，所以第一个重载无效
	const float& operator()(size_t i, size_t j) const {
		return data->data[index(i, j)];
	}
	float& operator()(size_t i, size_t j) {
		return data->data[index(i, j)];
	}
	// trans
	// 通过算子实现，弃用
	Matrix transpose() const
	{
		int64_t M = rows();
		int64_t N = cols();
		Matrix ret(N, M);

		const float* src = data->data;
		float* dst = ret.data->data;

		const int64_t src_stride = cols();
		const int64_t dst_stride = ret.cols();

		constexpr int64_t tile = 16;

#pragma omp parallel for num_threads(8) schedule(dynamic)
		for (int64_t i0 = 0; i0 < M; i0 += tile) {
			int64_t i1 = std::min(i0 + tile, M);
			for (int64_t j0 = 0; j0 < N; j0 += tile) {
				int64_t j1 = std::min(j0 + tile, N);
				for (int64_t i = i0; i < i1; i++) {
					const float* src_row = src + i * src_stride;
					for (int64_t j = j0; j < j1; j++) {
						dst[j * dst_stride + i] = src_row[j];
					}
				}
			}
		}
		return ret;
	}

	Matrix sliceRowsByIndex(size_t* indexes_gpu, size_t n) const {
		if (n == 0) return Matrix();
		for (size_t i = 0; i < n; i++)
			assert(indexes_gpu[i] < rows());
		size_t N = std::min(n, rows());
		size_t c = cols();
		Matrix ret(N, c);
		launchSliceRowsByIndex(data_gpu(), ret.data_gpu(), rows(), cols(), n, indexes_gpu);
		return ret;
	}
	template <typename F>
	Matrix foreach_do(F&& func) const {
		Matrix ret(rows(), cols());
		for (size_t i = 0; i < rows(); i++)
			for (size_t j = 0; j < cols(); j++)
				ret.data->data[ret.index(i, j)] = func((*this)(i, j));
		return ret;
	}
	// 弃用
	Matrix fill(float value) const {
		Matrix ret(rows(), cols());
		for (size_t i = 0; i < rows(); i++)
			for (size_t j = 0; j < cols(); j++)
				ret.data->data[ret.index(i, j)] = value;
		return ret;
	}

	void fill(float value) {
		launchFillInplace(data->data_gpu(), rows() * cols(), value);
	}

	// 查清所有调用后弃用
	void fill(const Matrix& other) {
		assert(other.data != nullptr && "fill: 源为空");
		assert(other.data != data && "fill: 源与目标共用同一块缓冲");
		assert(rows() == other.rows() && cols() == other.cols() && "fill: 尺寸不一致");
		// data->fill(other.data);
	}

	// calculate
	static void add(const Matrix& A, const Matrix& B, Matrix& out) {
		// 暂时不广播
		assert(A.rows() == B.rows());
		assert(A.rows() == out.rows());
		assert(A.cols() == B.cols());
		assert(A.cols() == out.cols());
		launchAdd(A.data_gpu(), B.data_gpu(), out.data_gpu(), A.rows() * A.cols());
	}
	static void sub(const Matrix& A, const Matrix& B, Matrix& out) {
		assert(A.rows() == B.rows());
		assert(A.rows() == out.rows());
		assert(A.cols() == B.cols());
		assert(A.cols() == out.cols());
		launchSub(A.data_gpu(), B.data_gpu(), out.data_gpu(), A.rows() * A.cols());
	}
	static void scaleMul(const Matrix& A, float value, Matrix& out) {
		assert(A.rows() == out.rows());
		assert(A.cols() == out.cols());
		launchScaleMul(A.data_gpu(), out.data_gpu(), A.rows() * A.cols(), value);
	}
	static void elementwiseMul(const Matrix& A, const Matrix& B, Matrix& out) {
		assert(A.rows() == B.rows());
		assert(A.rows() == out.rows());
		assert(A.cols() == B.cols());
		assert(A.cols() == out.cols());
		launchElementwiseMul(A.data_gpu(), B.data_gpu(), out.data_gpu(), A.rows() * A.cols());
	}

	// 所有matmul_*弃用，下面这个改成launch版本
	static Matrix matmul(const Matrix& A, const Matrix& B, Matrix& out)
	{
		assert(A.rows() == out.rows());
		assert(A.cols() == B.rows());
		assert(B.cols() == out.cols());
		launchMatmulNN(A.data_gpu(), B.data_gpu(), out.data_gpu(), A.rows(), B.cols(), A.cols());
	}

	// 4 个 float 加起来
	static inline float hsum_ps(__m256 v) {
		__m128 lo = _mm256_castps256_ps128(v);
		__m128 hi = _mm256_extractf128_ps(v, 1); 
		__m128 s = _mm_add_ps(lo, hi);  

		// 将 4 个 float 水平相加
		__m128 shuf = _mm_movehdup_ps(s);
		__m128 sums = _mm_add_ps(s, shuf);
		shuf = _mm_movehl_ps(shuf, sums);
		sums = _mm_add_ss(sums, shuf); 

		return _mm_cvtss_f32(sums);
	}

	// out = A * B, A(M,K), B(K,N), out(M,N)
	static void matmul_nn(const Matrix& A, const Matrix& B, Matrix& out)
	{
		int64_t M = (int64_t)A.rows();
		int64_t K = (int64_t)A.cols();
		int64_t N = (int64_t)B.cols();
		assert((int64_t)B.rows() == K);
		assert((int64_t)out.rows() == M && (int64_t)out.cols() == N);

		const float* A_data = A.data->data;
		const float* B_data = B.data->data;
		float* out_data = out.data->data;
		const int64_t A_cols = (int64_t)A.cols();
		const int64_t B_cols = (int64_t)B.cols();
		const int64_t out_cols = (int64_t)out.cols();

		//std::memset(out_data, 0, (size_t)M * N * sizeof(float));

#pragma omp parallel if (M > 512) num_threads(8)
		{
#pragma omp for schedule(dynamic)
			for (int64_t bi = 0; bi < M; bi += blocksize) {
				int64_t Mend = std::min(M, bi + blocksize);
				for (int64_t bj = 0; bj < N; bj += blocksize) {
					int64_t Nend = std::min(N, bj + blocksize);
					for (int64_t bk = 0; bk < K; bk += blocksize) {
						int64_t Kend = std::min(K, bk + blocksize);
						for (int64_t i = bi; i < Mend; ++i) {
							const float* A_row_i = A_data + i * A_cols;
							float* out_row_i = out_data + i * out_cols;
							int64_t j = bj;
							for (; j + 32 <= Nend; j += 32) {
								__m256 vc0 = _mm256_setzero_ps();
								__m256 vc1 = _mm256_setzero_ps();
								__m256 vc2 = _mm256_setzero_ps();
								__m256 vc3 = _mm256_setzero_ps();
								for (int64_t k = bk; k < Kend; ++k) {
									__m256 va0 = _mm256_set1_ps(A_row_i[k]);
									const float* base = B_data + k * B_cols + j;
									vc0 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(base), vc0);
									vc1 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(base + 8), vc1);
									vc2 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(base + 16), vc2);
									vc3 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(base + 24), vc3);
								}
								float* o = out_row_i + j;
								_mm256_storeu_ps(o, _mm256_add_ps(_mm256_loadu_ps(o), vc0));
								_mm256_storeu_ps(o + 8, _mm256_add_ps(_mm256_loadu_ps(o + 8), vc1));
								_mm256_storeu_ps(o + 16, _mm256_add_ps(_mm256_loadu_ps(o + 16), vc2));
								_mm256_storeu_ps(o + 24, _mm256_add_ps(_mm256_loadu_ps(o + 24), vc3));
							}
							for (; j + 8 <= Nend; j += 8) {
								__m256 vc = _mm256_setzero_ps();
								for (int64_t k = bk; k < Kend; ++k) {
									__m256 va = _mm256_set1_ps(A_row_i[k]);
									vc = _mm256_fmadd_ps(va, _mm256_loadu_ps(B_data + k * B_cols + j), vc);
								}
								float* o = out_row_i + j;
								_mm256_storeu_ps(o, _mm256_add_ps(_mm256_loadu_ps(o), vc));
							}
							for (; j < Nend; ++j) {
								float sum = 0.0f;
								for (int64_t k = bk; k < Kend; ++k)
									sum += A_row_i[k] * B_data[k * B_cols + j];
								out_row_i[j] += sum;
							}
						}
					}
				}
			}
		}
	}

	// out = A * B^T, A(M,K), B(N,K), out(M,N)
	static void matmul_nt(const Matrix& A, const Matrix& B, Matrix& out)
	{
		int64_t M = (int64_t)A.rows();
		int64_t K = (int64_t)A.cols();
		int64_t N = (int64_t)B.rows();
		assert((int64_t)B.cols() == K);
		assert((int64_t)out.rows() == M && (int64_t)out.cols() == N);

		const float* A_data = A.data->data;
		const float* B_data = B.data->data;
		float* out_data = out.data->data;
		const int64_t A_cols = (int64_t)A.cols();
		const int64_t B_cols = (int64_t)B.cols();
		const int64_t out_cols = (int64_t)out.cols();

		int64_t blocksize = 128; // 临时覆盖

		//std::memset(out_data, 0, (size_t)M * N * sizeof(float));

#pragma omp parallel if (M > 512) num_threads(8)
		{
#pragma omp for schedule(dynamic)
			for (int64_t bi = 0; bi < M; bi += blocksize) {
				int64_t Mend = std::min(M, bi + blocksize);
				for (int64_t bj = 0; bj < N; bj += blocksize) {
					int64_t Nend = std::min(N, bj + blocksize);
					for (int64_t bk = 0; bk < K; bk += blocksize) {
						int64_t Kend = std::min(K, bk + blocksize);
						for (int64_t i = bi; i < Mend; ++i) {
							const float* A_row_i = A_data + i * A_cols;
							float* out_row_i = out_data + i * out_cols;
							for (int64_t j = bj; j < Nend; ++j) {
								const float* B_row_j = B_data + j * B_cols;
								__m256 vc = _mm256_setzero_ps();
								int64_t k = bk;
								for (; k + 8 <= Kend; k += 8) {
									__m256 va = _mm256_loadu_ps(A_row_i + k);
									__m256 vb = _mm256_loadu_ps(B_row_j + k);
									vc = _mm256_fmadd_ps(va, vb, vc);
								}
								float sum = hsum_ps(vc);
								for (; k < Kend; ++k) sum += A_row_i[k] * B_row_j[k];
								out_row_i[j] += sum;
							}
						}
					}
				}
			}
		}
	}

	// out = A^T * B, A(K,M), B(K,N), out(M,N)
	static void matmul_tn(const Matrix& A, const Matrix& B, Matrix& out)
	{
		int64_t K = (int64_t)A.rows();
		int64_t M = (int64_t)A.cols();
		int64_t N = (int64_t)B.cols();
		assert((int64_t)B.rows() == K);
		assert((int64_t)out.rows() == M && (int64_t)out.cols() == N);

		const float* A_data = A.data->data;
		const float* B_data = B.data->data;
		float* out_data = out.data->data;
		const int64_t A_cols = (int64_t)A.cols();   // = M
		const int64_t B_cols = (int64_t)B.cols();   // = N
		const int64_t out_cols = (int64_t)out.cols();

		int64_t blocksize = 32; // 临时覆盖

		//std::memset(out_data, 0, (size_t)M * N * sizeof(float));

#pragma omp parallel if (M > 512) num_threads(8)
		{
#pragma omp for schedule(dynamic)
			for (int64_t bi = 0; bi < M; bi += blocksize) {
				int64_t Mend = std::min(M, bi + blocksize);
				for (int64_t bj = 0; bj < N; bj += blocksize) {
					int64_t Nend = std::min(N, bj + blocksize);
					for (int64_t bk = 0; bk < K; bk += blocksize) {
						int64_t Kend = std::min(K, bk + blocksize);
						for (int64_t i = bi; i < Mend; ++i) {
							float* out_row_i = out_data + i * out_cols;
							int64_t j = bj;
							for (; j + 32 <= Nend; j += 32) {
								__m256 vc0 = _mm256_setzero_ps();
								__m256 vc1 = _mm256_setzero_ps();
								__m256 vc2 = _mm256_setzero_ps();
								__m256 vc3 = _mm256_setzero_ps();
								for (int64_t k = bk; k < Kend; ++k) {
									__m256 va0 = _mm256_set1_ps(A_data[k * A_cols + i]);
									const float* base = B_data + k * B_cols + j;
									vc0 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(base), vc0);
									vc1 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(base + 8), vc1);
									vc2 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(base + 16), vc2);
									vc3 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(base + 24), vc3);
								}
								float* o = out_row_i + j;
								_mm256_storeu_ps(o, _mm256_add_ps(_mm256_loadu_ps(o), vc0));
								_mm256_storeu_ps(o + 8, _mm256_add_ps(_mm256_loadu_ps(o + 8), vc1));
								_mm256_storeu_ps(o + 16, _mm256_add_ps(_mm256_loadu_ps(o + 16), vc2));
								_mm256_storeu_ps(o + 24, _mm256_add_ps(_mm256_loadu_ps(o + 24), vc3));
							}
							for (; j + 8 <= Nend; j += 8) {
								__m256 vc = _mm256_setzero_ps();
								for (int64_t k = bk; k < Kend; ++k) {
									__m256 va = _mm256_set1_ps(A_data[k * A_cols + i]);
									vc = _mm256_fmadd_ps(va, _mm256_loadu_ps(B_data + k * B_cols + j), vc);
								}
								float* o = out_row_i + j;
								_mm256_storeu_ps(o, _mm256_add_ps(_mm256_loadu_ps(o), vc));
							}
							for (; j < Nend; ++j) {
								float sum = 0.0f;
								for (int64_t k = bk; k < Kend; ++k)
									sum += A_data[k * A_cols + i] * B_data[k * B_cols + j];
								out_row_i[j] += sum;
							}
						}
					}
				}
			}
		}
	}

	void sum_rows(Matrix& out) const {
		assert(out.rows() == rows());
		assert(out.cols() == 1);
		launchSumRows(data_gpu(), out.data_gpu(), rows(), cols());
	}
	void sum_cols(Matrix& out) const {
		assert(out.rows() == 1);
		assert(out.cols() == cols());
		launchSumCols(data_gpu(), out.data_gpu(), rows(), cols());
	}
	float sum() const {
		float ret = 0;
		for (size_t i = 0; i < rows(); i++) {
			for (size_t j = 0; j < cols(); j++) {
				ret += (*this)(i, j);
			}
		}
		return ret;
	}
	
	Matrix im2col(size_t N, size_t C, size_t H, size_t W,
		size_t ksize, size_t stride = 1, size_t padding = 0) const {
		assert(stride > 0);
		assert(W + 2 * padding >= ksize && H + 2 * padding >= ksize);
		size_t OW = (W + 2 * padding - ksize) / stride + 1;
		size_t OH = (H + 2 * padding - ksize) / stride + 1;
		size_t imSize = H * W;
		size_t out_size = OH * OW;
		size_t k_2 = ksize * ksize;
		Matrix ret(N * out_size, C * k_2); // 这里就决定了切分方式
		for (size_t n = 0; n < N; n++) {
			for (size_t oh = 0; oh < OH; oh++) {
				size_t head_y = oh * stride;
				for (size_t ow = 0; ow < OW; ow++) {
					size_t head_x = ow * stride;
					size_t rr_idx = n * out_size + oh * OW + ow;
					for (size_t c = 0; c < C; c++) {
						for (size_t h = 0; h < ksize; h++) {
							size_t cur_h = head_y + h;
							size_t rc_idx_p = c * k_2 + h * ksize;
							if (cur_h < padding || cur_h >= H + padding) {
								for (size_t w = 0; w < ksize; w++) {
									size_t rc_idx = rc_idx_p + w;
									ret(rr_idx, rc_idx) = 0;
								}
							}
							else {
								size_t tc_idx_p = c * imSize + (cur_h - padding) * W;
								for (size_t w = 0; w < ksize; w++) {
									size_t cur_w = head_x + w;
									size_t rc_idx = rc_idx_p + w;
									size_t tc_idx = tc_idx_p + cur_w - padding;
									if (cur_w < padding || cur_w >= W + padding)
										ret(rr_idx, rc_idx) = 0;
									else
										ret(rr_idx, rc_idx) = (*this)(n, tc_idx);
								}
							}
						}
					}
				}
			}
		}
		return ret;
	}
	Matrix col2im(size_t N, size_t C, size_t H, size_t W,
		size_t ksize, size_t stride = 1, size_t padding = 0) const {
		// from ((N OH OW),(C K K)) to (N, (C H W))
		assert(stride > 0);
		assert(W + 2 * padding >= ksize && H + 2 * padding >= ksize);
		size_t OW = (W + 2 * padding - ksize) / stride + 1;
		size_t OH = (H + 2 * padding - ksize) / stride + 1;
		size_t out_size = OH * OW;
		size_t k_2 = ksize * ksize;
		assert(rows() % out_size == 0);
		assert(cols() % k_2 == 0);
		assert(N == rows() / out_size);
		assert(C == cols() / k_2);
		size_t imSize = H * W;
		Matrix ret = Matrix::zeroMatrix(N, C * H * W);
		for (size_t n = 0; n < N; n++) {
			for (size_t oh = 0; oh < OH; oh++) {
				size_t head_y = oh * stride;
				for (size_t ow = 0; ow < OW; ow++) {
					size_t head_x = ow * stride;
					size_t rr_idx = n * out_size + oh * OW + ow;
					for (size_t c = 0; c < C; c++) {
						for (size_t h = 0; h < ksize; h++) {
							size_t cur_h = head_y + h;
							size_t rc_idx_p = c * k_2 + h * ksize;
							if (cur_h < padding || cur_h >= H + padding)
								continue; // 整行都跳过
							else {
								size_t tc_idx_p = c * imSize + (cur_h - padding) * W;
								for (size_t w = 0; w < ksize; w++) {
									size_t cur_w = head_x + w;
									size_t rc_idx = rc_idx_p + w;
									size_t tc_idx = tc_idx_p + cur_w - padding;
									if (cur_w < padding || cur_w >= W + padding)
										continue;
									else
										ret(n, tc_idx) += (*this)(rr_idx, rc_idx);
								}
							}
						}
					}
				}
			}
		}
		return ret;
	}
	//// 下面所有函数查清调用后弃用，或者重写
	//Matrix operator+(const Matrix& other) const {
	//	return Matrix::add(*this, other);
	//}
	//Matrix operator-(const Matrix& other) const {
	//	return Matrix::sub(*this, other);
	//}
	//Matrix operator*(double scale) const {
	//	return Matrix::scaleMul(*this, scale);
	//}
	//Matrix elementwiseMul(const Matrix& other) const {
	//	return Matrix::elementwiseMul(*this, other);
	//}
	//Matrix matmul(const Matrix& other) const {
	//	return Matrix::matmul(*this, other);
	//}
private:
	MatrixData* data;
	static int64_t blocksize;
	friend struct MatrixData;
private:
	size_t index(size_t i, size_t j) const {
		assert(i < rows() && j < cols() && "index out of range");
		return i * data->col + j;
	}
};


