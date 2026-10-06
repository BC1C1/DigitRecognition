#include "Linear.h"

Linear::Linear(size_t input_dim, size_t output_dim)
{
	W = Matrix::randnMatrix(output_dim, input_dim) * std::sqrt(2.0 / input_dim);
	//W_T = W.transpose();
	b = Matrix::zeroMatrix(1, output_dim);

	dw = Matrix(output_dim, input_dim);
	db = Matrix(1, output_dim);
}

Matrix Linear::forward(const Matrix& x)
{
	if (cache.rows() != x.rows() || cache.cols() != x.cols()) {
		cache = Matrix(x.rows(), x.cols()); 
		forwardPartCache = Matrix(x.rows(), W.rows()); // (B, O) 
		backwardRetCache = Matrix(x.rows(), W.cols()); // (B, I) 
	}

	cache.fill(x);                  

	forwardPartCache.fill(0);
	Matrix::matmul_nt(x, W, forwardPartCache);
	return forwardPartCache + b;
}

Matrix Linear::backward(const Matrix& gard)
{
	// 同理
	if (backwardRetCache.rows() != gard.rows() || backwardRetCache.cols() != W.cols()) {
		backwardRetCache = Matrix(gard.rows(), W.cols());   // (B, I)
	}

	const size_t N = gard.rows();
	const size_t out_dim = gard.cols();

	backwardRetCache.fill(0.0);
	Matrix::matmul_nn(gard, W, backwardRetCache);

	dw.fill(0.0);
	Matrix::matmul_tn(gard, cache, dw);

	db.fill(0.0);
	for (size_t n = 0; n < N; n++)
		for (size_t o = 0; o < out_dim; o++)
			db(0, o) += gard(n, o);

	return backwardRetCache;
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
