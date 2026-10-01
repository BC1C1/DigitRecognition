#include "Linear.h"

Linear::Linear(size_t input_dim, size_t output_dim)
{
	W = Matrix::randnMatrix(output_dim, input_dim) * std::sqrt(2.0 / input_dim);
	//W_T = W.transpose();
	b = Matrix::randnMatrix(1, output_dim);
}

Matrix Linear::forward(const Matrix& x)
{
	cache = x;
	Matrix t = Matrix::zeroMatrix(x.rows(), W.rows());
	Matrix::matmul_nt(x, W, t);
	return t + b;
}

Matrix Linear::backward(const Matrix& gard)
{
	size_t N = gard.rows();
	size_t out_dim = gard.cols();
	// gard
	auto ret = gard.matmul(W);
	dw = Matrix::zeroMatrix(gard.cols(), cache.cols());
	Matrix::matmul_tn(gard, cache, dw);

	db = Matrix::zeroMatrix(1, out_dim);
	for (size_t n = 0; n < N; n++)
		for (size_t o = 0; o < out_dim; o++)
			db(0, o) += gard(n, o);

	return ret;
}

void Linear::update(double learning_rate) {
	for (size_t i = 0; i < W.rows(); i++)
		for (size_t j = 0; j < W.cols(); j++)
			W(i, j) -= learning_rate * dw(i, j);
	for (size_t i = 0; i < b.rows(); i++)
		for (size_t j = 0; j < b.cols(); j++)
			b(i, j) -= learning_rate * db(i, j);
	//W_T = W.transpose();
}
