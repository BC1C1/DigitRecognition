#pragma once
#include <cmath>
#include <cassert>
#include <vector>
#include <immintrin.h>
#include <omp.h>

#include "Utils.h"

struct MatrixData
{
	double* data;
	size_t row;
	size_t col;
	MatrixData(size_t row, size_t col) : row(row), col(col) {
		data = new double[row * col];
	}
	MatrixData(const MatrixData&) = delete;
	MatrixData& operator=(const MatrixData&) = delete;
	~MatrixData() { if (data) delete[] data; }
	MatrixData* copy() const {
		double* newData = new double[row * col]();
		for (size_t i = 0; i < row * col; i++)
			newData[i] = data[i];
		return new MatrixData(newData, row, col);
	}
private:
	MatrixData(double* data, size_t row, size_t col) : row(row), col(col), data(data) {}
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
	// index
	const double& operator()(size_t i, size_t j) const {
		return data->data[index(i, j)];
	}
	double& operator()(size_t i, size_t j) {
		return data->data[index(i, j)];
	}
	// trans
	Matrix transpose() const
	{
		size_t M = rows();
		size_t N = cols();
		Matrix ret(N, M);

		const double* src = data->data;
		double* dst = ret.data->data;

		const size_t src_stride = cols();
		const size_t dst_stride = ret.cols();

		constexpr size_t tile = 16;

#pragma omp parallel for num_threads(16) schedule(dynamic)
		for (size_t i0 = 0; i0 < M; i0 += tile) {
			size_t i1 = std::min(i0 + tile, M);
			for (size_t j0 = 0; j0 < N; j0 += tile) {
				size_t j1 = std::min(j0 + tile, N);
				for (size_t i = i0; i < i1; i++) {
					const double* src_row = src + i * src_stride;
					for (size_t j = j0; j < j1; j++) {
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
	Matrix fill(double value) const {
		Matrix ret(rows(), cols());
		for (size_t i = 0; i < rows(); i++)
			for (size_t j = 0; j < cols(); j++)
				ret.data->data[ret.index(i, j)] = value;
		return ret;
	}
	// factory
	static Matrix zeroMatrix(size_t row, size_t col) {
		Matrix ret(row, col);
		std::memset(ret.data->data, 0, row * col * sizeof(double));
		return ret;
	}
	static Matrix randnMatrix(size_t row, size_t col, double scale = 1.0) {
		Matrix ret(row, col);
		return ret.foreach_do([=](double) { return randn() * scale; });
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
	static Matrix scaleMul(const Matrix& A, double value) {
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

		const double* A_data = A.data->data;
		const double* B_data = B.data->data;
		double* ret_data = ret.data->data;

		int64_t A_cols = A.cols();
		int64_t B_cols = B.cols();
		int64_t ret_cols = ret.cols();

#pragma omp parallel num_threads(16)
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
							const double* A_row_i = A_data + i * A_cols;
							double* ret_row_i = ret_data + i * ret_cols;

							int64_t j = bj;
							// j +=16，4组ymm，每组4个double
							for (; j + 16 <= Nend; j += 16)
							{
								auto vc0 = _mm256_setzero_pd();
								auto vc1 = _mm256_setzero_pd();
								auto vc2 = _mm256_setzero_pd();
								auto vc3 = _mm256_setzero_pd();

								for (int64_t k = bk; k < Kend; k++)
								{
									double a_ik = A_row_i[k];
									__m256d va0 = _mm256_set1_pd(a_ik);

									// B的k行指针，k循环内只算一次k*B_cols
									const double* B_row_k = B_data + k * B_cols;
									const double* base = B_row_k + j;

									__m256d vb0 = _mm256_loadu_pd(base);
									__m256d vb1 = _mm256_loadu_pd(base + 4);
									__m256d vb2 = _mm256_loadu_pd(base + 8);
									__m256d vb3 = _mm256_loadu_pd(base + 12);

									vc0 = _mm256_fmadd_pd(va0, vb0, vc0);
									vc1 = _mm256_fmadd_pd(va0, vb1, vc1);
									vc2 = _mm256_fmadd_pd(va0, vb2, vc2);
									vc3 = _mm256_fmadd_pd(va0, vb3, vc3);
								}

								double* out_base = ret_row_i + j;
								__m256d v0 = _mm256_loadu_pd(out_base);
								__m256d v1 = _mm256_loadu_pd(out_base + 4);
								__m256d v2 = _mm256_loadu_pd(out_base + 8);
								__m256d v3 = _mm256_loadu_pd(out_base + 12);

								v0 = _mm256_add_pd(v0, vc0);
								v1 = _mm256_add_pd(v1, vc1);
								v2 = _mm256_add_pd(v2, vc2);
								v3 = _mm256_add_pd(v3, vc3);

								_mm256_storeu_pd(out_base, v0);
								_mm256_storeu_pd(out_base + 4, v1);
								_mm256_storeu_pd(out_base + 8, v2);
								_mm256_storeu_pd(out_base + 12, v3);
							}

							for (; j + 4 <= Nend; j += 4)
							{
								auto vc = _mm256_setzero_pd();
								for (int64_t k = bk; k < Kend; k++)
								{
									double a_ik = A_row_i[k];
									__m256d va = _mm256_set1_pd(a_ik);
									const double* B_row_k = B_data + k * B_cols;
									__m256d vb = _mm256_loadu_pd(B_row_k + j);
									vc = _mm256_fmadd_pd(va, vb, vc);
								}
								double* out_ptr = ret_row_i + j;
								__m256d vret = _mm256_loadu_pd(out_ptr);
								vret = _mm256_add_pd(vret, vc);
								_mm256_storeu_pd(out_ptr, vret);
							}

							for (; j < Nend; j++)
							{
								double sum = 0.0;
								for (int64_t k = bk; k < Kend; k++)
								{
									double a_ik = A_row_i[k];
									const double* B_row_k = B_data + k * B_cols;
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

	// 水平归约：把 __m256d 的 4 个 double 加起来
	static inline double hsum_pd(__m256d v) {
		__m128d lo = _mm256_castpd256_pd128(v);
		__m128d hi = _mm256_extractf128_pd(v, 1);
		__m128d s = _mm_add_pd(lo, hi);
		return _mm_cvtsd_f64(s) + _mm_cvtsd_f64(_mm_unpackhi_pd(s, s));
	}

	// out = A * B, A(M,K), B(K,N), out(M,N)
	static void matmul_nn(const Matrix& A, const Matrix& B, Matrix& out)
	{
		int64_t M = (int64_t)A.rows();
		int64_t K = (int64_t)A.cols();
		int64_t N = (int64_t)B.cols();
		assert((int64_t)B.rows() == K);
		assert((int64_t)out.rows() == M && (int64_t)out.cols() == N);

		const double* A_data = A.data->data;
		const double* B_data = B.data->data;
		double* out_data = out.data->data;
		const int64_t A_cols = (int64_t)A.cols();
		const int64_t B_cols = (int64_t)B.cols();
		const int64_t out_cols = (int64_t)out.cols();

		//std::memset(out_data, 0, (size_t)M * N * sizeof(double));

#pragma omp parallel
		{
#pragma omp for schedule(dynamic)
			for (int64_t bi = 0; bi < M; bi += blocksize) {
				int64_t Mend = std::min(M, bi + blocksize);
				for (int64_t bj = 0; bj < N; bj += blocksize) {
					int64_t Nend = std::min(N, bj + blocksize);
					for (int64_t bk = 0; bk < K; bk += blocksize) {
						int64_t Kend = std::min(K, bk + blocksize);
						for (int64_t i = bi; i < Mend; ++i) {
							const double* A_row_i = A_data + i * A_cols;
							double* out_row_i = out_data + i * out_cols;
							int64_t j = bj;
							for (; j + 16 <= Nend; j += 16) {
								__m256d vc0 = _mm256_setzero_pd();
								__m256d vc1 = _mm256_setzero_pd();
								__m256d vc2 = _mm256_setzero_pd();
								__m256d vc3 = _mm256_setzero_pd();
								for (int64_t k = bk; k < Kend; ++k) {
									__m256d va0 = _mm256_set1_pd(A_row_i[k]);
									const double* base = B_data + k * B_cols + j;
									vc0 = _mm256_fmadd_pd(va0, _mm256_loadu_pd(base), vc0);
									vc1 = _mm256_fmadd_pd(va0, _mm256_loadu_pd(base + 4), vc1);
									vc2 = _mm256_fmadd_pd(va0, _mm256_loadu_pd(base + 8), vc2);
									vc3 = _mm256_fmadd_pd(va0, _mm256_loadu_pd(base + 12), vc3);
								}
								double* o = out_row_i + j;
								_mm256_storeu_pd(o, _mm256_add_pd(_mm256_loadu_pd(o), vc0));
								_mm256_storeu_pd(o + 4, _mm256_add_pd(_mm256_loadu_pd(o + 4), vc1));
								_mm256_storeu_pd(o + 8, _mm256_add_pd(_mm256_loadu_pd(o + 8), vc2));
								_mm256_storeu_pd(o + 12, _mm256_add_pd(_mm256_loadu_pd(o + 12), vc3));
							}
							for (; j + 4 <= Nend; j += 4) {
								__m256d vc = _mm256_setzero_pd();
								for (int64_t k = bk; k < Kend; ++k) {
									__m256d va = _mm256_set1_pd(A_row_i[k]);
									vc = _mm256_fmadd_pd(va, _mm256_loadu_pd(B_data + k * B_cols + j), vc);
								}
								double* o = out_row_i + j;
								_mm256_storeu_pd(o, _mm256_add_pd(_mm256_loadu_pd(o), vc));
							}
							for (; j < Nend; ++j) {
								double sum = 0.0;
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

		const double* A_data = A.data->data;
		const double* B_data = B.data->data;
		double* out_data = out.data->data;
		const int64_t A_cols = (int64_t)A.cols();
		const int64_t B_cols = (int64_t)B.cols();
		const int64_t out_cols = (int64_t)out.cols();

		//std::memset(out_data, 0, (size_t)M * N * sizeof(double));

#pragma omp parallel
		{
#pragma omp for schedule(dynamic)
			for (int64_t bi = 0; bi < M; bi += blocksize) {
				int64_t Mend = std::min(M, bi + blocksize);
				for (int64_t bj = 0; bj < N; bj += blocksize) {
					int64_t Nend = std::min(N, bj + blocksize);
					for (int64_t bk = 0; bk < K; bk += blocksize) {
						int64_t Kend = std::min(K, bk + blocksize);
						for (int64_t i = bi; i < Mend; ++i) {
							const double* A_row_i = A_data + i * A_cols;
							double* out_row_i = out_data + i * out_cols;
							for (int64_t j = bj; j < Nend; ++j) {
								const double* B_row_j = B_data + j * B_cols;
								__m256d vc = _mm256_setzero_pd();
								int64_t k = bk;
								for (; k + 4 <= Kend; k += 4) {
									__m256d va = _mm256_loadu_pd(A_row_i + k);
									__m256d vb = _mm256_loadu_pd(B_row_j + k);
									vc = _mm256_fmadd_pd(va, vb, vc);
								}
								double sum = hsum_pd(vc);
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

		const double* A_data = A.data->data;
		const double* B_data = B.data->data;
		double* out_data = out.data->data;
		const int64_t A_cols = (int64_t)A.cols();   // = M
		const int64_t B_cols = (int64_t)B.cols();   // = N
		const int64_t out_cols = (int64_t)out.cols();

		//std::memset(out_data, 0, (size_t)M * N * sizeof(double));

#pragma omp parallel
		{
#pragma omp for schedule(dynamic)
			for (int64_t bi = 0; bi < M; bi += blocksize) {
				int64_t Mend = std::min(M, bi + blocksize);
				for (int64_t bj = 0; bj < N; bj += blocksize) {
					int64_t Nend = std::min(N, bj + blocksize);
					for (int64_t bk = 0; bk < K; bk += blocksize) {
						int64_t Kend = std::min(K, bk + blocksize);
						for (int64_t i = bi; i < Mend; ++i) {
							double* out_row_i = out_data + i * out_cols;
							int64_t j = bj;
							for (; j + 16 <= Nend; j += 16) {
								__m256d vc0 = _mm256_setzero_pd();
								__m256d vc1 = _mm256_setzero_pd();
								__m256d vc2 = _mm256_setzero_pd();
								__m256d vc3 = _mm256_setzero_pd();
								for (int64_t k = bk; k < Kend; ++k) {
									__m256d va0 = _mm256_set1_pd(A_data[k * A_cols + i]);
									const double* base = B_data + k * B_cols + j;
									vc0 = _mm256_fmadd_pd(va0, _mm256_loadu_pd(base), vc0);
									vc1 = _mm256_fmadd_pd(va0, _mm256_loadu_pd(base + 4), vc1);
									vc2 = _mm256_fmadd_pd(va0, _mm256_loadu_pd(base + 8), vc2);
									vc3 = _mm256_fmadd_pd(va0, _mm256_loadu_pd(base + 12), vc3);
								}
								double* o = out_row_i + j;
								_mm256_storeu_pd(o, _mm256_add_pd(_mm256_loadu_pd(o), vc0));
								_mm256_storeu_pd(o + 4, _mm256_add_pd(_mm256_loadu_pd(o + 4), vc1));
								_mm256_storeu_pd(o + 8, _mm256_add_pd(_mm256_loadu_pd(o + 8), vc2));
								_mm256_storeu_pd(o + 12, _mm256_add_pd(_mm256_loadu_pd(o + 12), vc3));
							}
							for (; j + 4 <= Nend; j += 4) {
								__m256d vc = _mm256_setzero_pd();
								for (int64_t k = bk; k < Kend; ++k) {
									__m256d va = _mm256_set1_pd(A_data[k * A_cols + i]);
									vc = _mm256_fmadd_pd(va, _mm256_loadu_pd(B_data + k * B_cols + j), vc);
								}
								double* o = out_row_i + j;
								_mm256_storeu_pd(o, _mm256_add_pd(_mm256_loadu_pd(o), vc));
							}
							for (; j < Nend; ++j) {
								double sum = 0.0;
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
	double sum() const {
		double ret = 0;
		for (size_t i = 0; i < rows(); i++) {
			for (size_t j = 0; j < cols(); j++) {
				ret += (*this)(i, j);
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
private:
	size_t index(size_t i, size_t j) const {
		assert(i < rows() && j < cols() && "index out of range");
		return i * data->col + j;
	}
};


