#pragma once

#include "Utils.h"
#include <cassert>

struct OldMatrixData
{
	double* data;
	size_t ref_cnt;
	size_t row;
	size_t col;
	OldMatrixData(size_t row = 0, size_t col = 0) : ref_cnt(1), row(row), col(col) {
		data = new double[row * col]();
	}

	OldMatrixData* copy() {
		OldMatrixData* ret = new OldMatrixData(row, col);
		for (size_t i = 0; i < row * col; i++) {
			ret->data[i] = data[i];
		}
		return ret;
	}
};

class OldMatrix
{
public:
	using Func = std::function<double(double)>;
public:
	OldMatrix(size_t row = 0, size_t col = 0) : is_transposed(false) { matrix = new OldMatrixData(row, col); }

	OldMatrix(const OldMatrix& other) : matrix(other.matrix), is_transposed(other.is_transposed) { ref(); }

	OldMatrix(OldMatrix&& other) : matrix(other.matrix), is_transposed(other.is_transposed) { other.matrix = nullptr; }

	OldMatrix collectRowsByIndex(const std::vector<size_t>& indexes) const {
		size_t M = indexes.size();	// 要取多少条
		M = std::min(rows(), M);	// rows()是实际能取多少条，多的截断
		OldMatrix ret(M, cols());
		for (size_t i = 0; i < M; i++) {
			auto idx = indexes[i];
			assert(idx < rows(), "index out of range when collect row from matrix");
			for (size_t j = 0; j < cols(); j++) {
				ret(i, j) = (*this)(idx, j);
			}
		}
		return ret;
	}

	OldMatrix sliceRows(size_t beginIdx, size_t size) {
		assert(beginIdx < rows(), "slice beginIdx out of range");
		size_t endIdx = beginIdx + size;
		if (endIdx > this->rows())
		{
			size = this->rows() - beginIdx;
		}
		OldMatrix ret(size, cols());
		for (size_t i = 0; i < size; i++) {
			for (size_t j = 0; j < cols(); j++) {
				ret(i, j) = (*this)(i + beginIdx, j);
			}
		}
		return ret;
	}

	void foreach_do(Func func) {
		for (size_t i = 0; i < rows(); i++) {
			for (size_t j = 0; j < cols(); j++) {
				(*this)(i, j) = func((*this)(i, j));
			}
		}
	}

	OldMatrix operator*(const OldMatrix& other) const {
		return mul(*this, other);
	}

	OldMatrix operator+(const OldMatrix& other) const {
		return add(*this, other);
	}

	OldMatrix operator-(const OldMatrix& other) const {
		return sub(*this, other);
	}

	static OldMatrix mul(const OldMatrix& A, const OldMatrix& B) {
		// A -> M*K; B -> K*N; C -> M*N
		auto M = A.rows();
		auto K = A.cols();
		assert(B.rows() == K && "Dimension mismatch for matrix multiply");
		auto N = B.cols();
		OldMatrix C = OldMatrix::generateZeroMatrix(M, N);
		const OldMatrix Bt = B.transpose_view();
		for (size_t i = 0; i < M; i++)
		{
			for (size_t j = 0; j < N; j++)
			{
				double sum = 0.0;
				for (size_t k = 0; k < K; k++)
				{
					sum += A(i, k) * Bt(j, k);
				}
				C(i, j) = sum;
			}
		}
		return C;
	}

	static OldMatrix add(const OldMatrix& A, const OldMatrix& B) {
		// 给每一个行加一个行向量
		if (B.rows() == 1 && A.rows() > 1) {
			OldMatrix C = generateZeroMatrix(A.rows(), A.cols());
			for (size_t i = 0; i < A.rows(); i++)
				for (size_t j = 0; j < A.cols(); j++)
					C(i, j) = A(i, j) + B(0, j);
			return C;
		}
		auto M = A.rows();
		auto N = A.cols();
		assert(M == B.rows() && "Dimension mismatch for matrix add");
		assert(N == B.cols() && "Dimension mismatch for matrix add");
		OldMatrix C = generateZeroMatrix(M, N);
		for (size_t i = 0; i < M; i++) {
			for (size_t j = 0; j < N; j++) {
				C(i, j) = A(i, j) + B(i, j);
			}
		}
		return C;
	}

	static OldMatrix sub(const OldMatrix& A, const OldMatrix& B) {
		auto M = A.rows();
		auto N = A.cols();
		assert(M == B.rows() && "Dimension mismatch for matrix sub");
		assert(N == B.cols() && "Dimension mismatch for matrix sub");
		OldMatrix C = generateZeroMatrix(M, N);
		for (size_t i = 0; i < M; i++) {
			for (size_t j = 0; j < N; j++) {
				C(i, j) = A(i, j) - B(i, j);
			}
		}
		return C;
	}

	static OldMatrix elementwiseMul(const OldMatrix& A, const OldMatrix& B) {
		auto M = A.rows();
		auto N = A.cols();
		assert(M == B.rows() && "Dimension mismatch for matrix elementwise multiply");
		assert(N == B.cols() && "Dimension mismatch for matrix elementwise multiply");
		OldMatrix C = generateZeroMatrix(M, N);
		for (size_t i = 0; i < M; i++) {
			for (size_t j = 0; j < N; j++) {
				C(i, j) = A(i, j) * B(i, j);
			}
		}
		return C;
	}

	// 把自己变成转置后的矩阵(内存行为上不复制)(但是不要认为共享的一起变，就当COW不存在)
	void transpose() {
		is_transposed = !is_transposed;
	}

	// 生成得到转置后的矩阵(内存行为上不复制)
	OldMatrix transpose_view() const {
		OldMatrix ret(*this);
		ret.is_transposed = !this->is_transposed;
		return ret;
	}

	static OldMatrix generateRandomMatrix(size_t row, size_t col, double scale = 1) {
		return fillWith(row, col, [=](double)->double { return randn() * scale; });
	}

	static OldMatrix generateZeroMatrix(size_t row, size_t col) {
		return fillWith(row, col, [](double)->double { return 0; });
	}

	static OldMatrix fillWith(size_t row, size_t col, Func func) {
		auto body = OldMatrix(row, col);
		body.foreach_do(func);
		return body;
	}

	double operator()(size_t i, size_t j) const {
		return matrix->data[index(i, j)];
	}
	double& operator()(size_t i, size_t j) {
		detach();
		return matrix->data[index(i, j)];
	}

	OldMatrix& operator=(const OldMatrix& other) {
		if (&other == this)
			return *this;
		this->deref();
		matrix = other.matrix;
		is_transposed = other.is_transposed;
		other.ref();
		return *this;
	}

	OldMatrix& operator=(OldMatrix&& other) noexcept {
		if (this == &other) return *this;
		deref();
		matrix = other.matrix;
		is_transposed = other.is_transposed;
		other.matrix = nullptr;
		return *this;
	}

	size_t rows() const { return is_transposed ? matrix->col : matrix->row; }
	size_t cols() const { return is_transposed ? matrix->row : matrix->col; }

	~OldMatrix() { deref(); }

private:
	OldMatrixData* matrix;
	bool is_transposed;

	inline size_t index(size_t i, size_t j) const {
		return is_transposed ? j * matrix->col + i : i * matrix->col + j;
	}

	inline void ref() const {
		if (matrix)
			matrix->ref_cnt++;
	}

	inline void deref() {
		if (!matrix) return;
		matrix->ref_cnt--;
		if (matrix->ref_cnt == 0)
		{
			delete[] matrix->data;
			delete matrix;
		}
	}

	void detach() {
		if (!matrix) return;
		if (matrix->ref_cnt > 1) {
			auto newData = matrix->copy();
			deref();
			matrix = newData;
		}
	}

};

