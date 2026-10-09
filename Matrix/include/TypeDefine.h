#pragma once
#include <cmath>
#include <random>
#include <cassert>
#include <vector>
#include <immintrin.h>
#include <omp.h>

inline float randn()
{
	static std::mt19937 rng(42);
	static std::normal_distribution<float> dist(0.0, 1.0);
	return dist(rng);
}

struct MatrixData
{
	float* data;
	size_t row;
	size_t col;
	MatrixData(size_t row, size_t col) : row(row), col(col) {
		data = new float[row * col]();
	}
	MatrixData(const MatrixData&) = delete;
	MatrixData& operator=(const MatrixData&) = delete;
	~MatrixData() { if (data) delete[] data; }
	MatrixData* copy() const {
		float* newData = new float[row * col]();
		for (size_t i = 0; i < row * col; i++)
			newData[i] = data[i];
		return new MatrixData(newData, row, col);
	}
	void fill(const MatrixData* other) {
		// 不检查，仅被Matrix调用
		size_t size = row * col;
		for (size_t i = 0; i < size; i++) {
			data[i] = other->data[i];
		}
	}
private:
	MatrixData(float* data, size_t row, size_t col) : row(row), col(col), data(data) {}
};
class Matrix {
public:
	Matrix() : data(new MatrixData(0, 0)) {}
	Matrix(size_t row, size_t col) : data(new MatrixData(row, col)) {}
	~Matrix() { delete data; }
	Matrix(const Matrix& other) : data(other.data->copy()) {
	}
	Matrix(Matrix&& other) noexcept : data(other.data) {
		other.data = nullptr;
	}
	Matrix& operator=(const Matrix& other) {
		if (&other == this)
			return *this;
		delete data;
		data = other.data->copy();
		return *this;
	}
	Matrix& operator=(Matrix&& other) noexcept {
		if (&other == this)
			return *this;
		delete data;
		data = other.data;
		other.data = nullptr;
		return *this;
	}

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
	float* raw_data() const {
		return data->data;
	}
	// index
	const float& operator()(size_t i, size_t j) const {
		return data->data[index(i, j)];
	}
	float& operator()(size_t i, size_t j) {
		return data->data[index(i, j)];
	}
	// trans
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

	Matrix sliceRowsByIndex(const std::vector<size_t>& indexes) const {
		size_t N = std::min(indexes.size(), rows());
		size_t c = cols();
		Matrix ret(N, c);
		for (size_t i = 0; i < N; i++) {
			assert(indexes[i] < rows() && "index out of range");
			for (size_t j = 0; j < c; j++)
				ret.data->data[ret.index(i, j)] = (*this)(indexes[i], j);
		}
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

	Matrix fill(float value) const {
		Matrix ret(rows(), cols());
		for (size_t i = 0; i < rows(); i++)
			for (size_t j = 0; j < cols(); j++)
				ret.data->data[ret.index(i, j)] = value;
		return ret;
	}

	void fill(float value) {
		const size_t r = rows(), c = cols();
		float* p = data->data;
		for (size_t i = 0; i < r; i++) {
			float* rowp = p + i * c;
			for (size_t j = 0; j < c; j++)
				rowp[j] = value;
		}
	}

	void fill(const Matrix& other) {
		assert(other.data != nullptr && "fill: 源为空");
		assert(other.data != data && "fill: 源与目标共用同一块缓冲");
		assert(rows() == other.rows() && cols() == other.cols() && "fill: 尺寸不一致");
		data->fill(other.data);
	}

	// factory
	static Matrix zeroMatrix(size_t row, size_t col) {
		Matrix ret(row, col);
		std::memset(ret.data->data, 0, row * col * sizeof(float));
		return ret;
	}
	static Matrix randnMatrix(size_t row, size_t col, float scale = 1.0) {
		Matrix ret(row, col);
		return ret.foreach_do([=](float) { return randn() * scale; });
	}
	// calculate
	static Matrix add(const Matrix& A, const Matrix& B) {
		// 不考虑某一矩阵行数为0
		size_t M = A.rows();
		size_t N = A.cols();
		assert(N == B.cols() && "Dimension mismatch for matrix add");
		if (M == B.rows()) {
			Matrix ret(M, N);
			for (size_t i = 0; i < M; i++)
				for (size_t j = 0; j < N; j++)
					ret.data->data[ret.index(i, j)] = A(i, j) + B(i, j);
			return ret;
		}
		else if (M == 1) {
			Matrix ret(B.rows(), N);
			for (size_t i = 0; i < B.rows(); i++)
				for (size_t j = 0; j < N; j++)
					ret.data->data[ret.index(i, j)] = A(0, j) + B(i, j);
			return ret;
		}
		else if (B.rows() == 1) {
			Matrix ret(M, N);
			for (size_t i = 0; i < M; i++)
				for (size_t j = 0; j < N; j++)
					ret.data->data[ret.index(i, j)] = A(i, j) + B(0, j);
			return ret;
		}
		else
			assert(false && "Dimension mismatch for matrix add");
		return Matrix();
	}
	static Matrix sub(const Matrix& A, const Matrix& B) {
		return Matrix::add(A, Matrix::scaleMul(B, -1));
	}
	static Matrix scaleMul(const Matrix& A, float value) {
		size_t M = A.rows();
		size_t N = A.cols();
		Matrix ret(M, N);
		for (size_t i = 0; i < M; i++)
			for (size_t j = 0; j < N; j++)
				ret.data->data[ret.index(i, j)] = A(i, j) * value;
		return ret;
	}
	static Matrix elementwiseMul(const Matrix& A, const Matrix& B) {
		size_t M = A.rows();
		size_t N = A.cols();
		assert(M == B.rows() && N == B.cols() && "Dimension mismatch for elementwise multiply");
		Matrix ret(M, N);
		for (size_t i = 0; i < M; i++)
			for (size_t j = 0; j < N; j++)
				ret.data->data[ret.index(i, j)] = A(i, j) * B(i, j);
		return ret;
	}

	static Matrix matmul(const Matrix& A, const Matrix& B)
	{
		int64_t M = A.rows();
		int64_t K = A.cols();
		assert(K == B.rows() && "Dimension mismatch for matmul");
		int64_t N = B.cols();
		Matrix ret = Matrix::zeroMatrix(M, N);

		const float* A_data = A.data->data;
		const float* B_data = B.data->data;
		float* ret_data = ret.data->data;

		int64_t A_cols = A.cols();
		int64_t B_cols = B.cols();
		int64_t ret_cols = ret.cols();

#pragma omp parallel num_threads(8)
		{
#pragma omp for schedule(dynamic)
			for (int64_t bi = 0; bi < M; bi += blocksize)
			{
				int64_t Mend = std::min(M, bi + blocksize);
				for (int64_t bj = 0; bj < N; bj += blocksize)
				{
					int64_t Nend = std::min(N, bj + blocksize);
					for (int64_t bk = 0; bk < K; bk += blocksize)
					{
						int64_t Kend = std::min(K, bk + blocksize);

						for (int64_t i = bi; i < Mend; i++)
						{
							// A第i行起始指针，i行固定，提到k循环外面
							const float* A_row_i = A_data + i * A_cols;
							float* ret_row_i = ret_data + i * ret_cols;

							int64_t j = bj;
							// j += 32，4组ymm，每组8个float
							for (; j + 32 <= Nend; j += 32)
							{
								auto vc0 = _mm256_setzero_ps();
								auto vc1 = _mm256_setzero_ps();
								auto vc2 = _mm256_setzero_ps();
								auto vc3 = _mm256_setzero_ps();

								for (int64_t k = bk; k < Kend; k++)
								{
									float a_ik = A_row_i[k];
									__m256 va0 = _mm256_set1_ps(a_ik);

									// B的k行指针，k循环内只算一次k*B_cols
									const float* B_row_k = B_data + k * B_cols;
									const float* base = B_row_k + j;

									__m256 vb0 = _mm256_loadu_ps(base);
									__m256 vb1 = _mm256_loadu_ps(base + 8);
									__m256 vb2 = _mm256_loadu_ps(base + 16);
									__m256 vb3 = _mm256_loadu_ps(base + 24);

									vc0 = _mm256_fmadd_ps(va0, vb0, vc0);
									vc1 = _mm256_fmadd_ps(va0, vb1, vc1);
									vc2 = _mm256_fmadd_ps(va0, vb2, vc2);
									vc3 = _mm256_fmadd_ps(va0, vb3, vc3);
								}

								float* out_base = ret_row_i + j;
								__m256 v0 = _mm256_loadu_ps(out_base);
								__m256 v1 = _mm256_loadu_ps(out_base + 8);
								__m256 v2 = _mm256_loadu_ps(out_base + 16);
								__m256 v3 = _mm256_loadu_ps(out_base + 24);

								v0 = _mm256_add_ps(v0, vc0);
								v1 = _mm256_add_ps(v1, vc1);
								v2 = _mm256_add_ps(v2, vc2);
								v3 = _mm256_add_ps(v3, vc3);

								_mm256_storeu_ps(out_base, v0);
								_mm256_storeu_ps(out_base + 8, v1);
								_mm256_storeu_ps(out_base + 16, v2);
								_mm256_storeu_ps(out_base + 24, v3);
							}

							for (; j + 8 <= Nend; j += 8)
							{
								auto vc = _mm256_setzero_ps();
								for (int64_t k = bk; k < Kend; k++)
								{
									float a_ik = A_row_i[k];
									__m256 va = _mm256_set1_ps(a_ik);
									const float* B_row_k = B_data + k * B_cols;
									__m256 vb = _mm256_loadu_ps(B_row_k + j);
									vc = _mm256_fmadd_ps(va, vb, vc);
								}
								float* out_ptr = ret_row_i + j;
								__m256 vret = _mm256_loadu_ps(out_ptr);
								vret = _mm256_add_ps(vret, vc);
								_mm256_storeu_ps(out_ptr, vret);
							}

							for (; j < Nend; j++)
							{
								float sum = 0.0f;
								for (int64_t k = bk; k < Kend; k++)
								{
									float a_ik = A_row_i[k];
									const float* B_row_k = B_data + k * B_cols;
									sum += a_ik * B_row_k[j];
								}
								ret_row_i[j] += sum;
							}
						}
					}
				}
			}
		}
		return ret;
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

	Matrix sum_rows() const {
		Matrix ret = Matrix::zeroMatrix(rows(), 1);
		for (size_t i = 0; i < rows(); i++) {
			for (size_t j = 0; j < cols(); j++) {
				ret.data->data[ret.index(i, 0)] += (*this)(i, j);
			}
		}
		return ret;
	}
	Matrix sum_cols() const {
		Matrix ret = Matrix::zeroMatrix(1, cols());
		for (size_t i = 0; i < rows(); i++) {
			for (size_t j = 0; j < cols(); j++) {
				ret.data->data[ret.index(0, j)] += (*this)(i, j);
			}
		}
		return ret;
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
	Matrix operator+(const Matrix& other) const {
		return Matrix::add(*this, other);
	}
	Matrix operator-(const Matrix& other) const {
		return Matrix::sub(*this, other);
	}
	Matrix operator*(double scale) const {
		return Matrix::scaleMul(*this, scale);
	}
	Matrix elementwiseMul(const Matrix& other) const {
		return Matrix::elementwiseMul(*this, other);
	}
	Matrix matmul(const Matrix& other) const {
		return Matrix::matmul(*this, other);
	}
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


