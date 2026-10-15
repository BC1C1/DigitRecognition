#include "Linear.h"

Linear::Linear(size_t input_dim, size_t output_dim) : I(input_dim), O(output_dim)
{
	W = Matrix::randnMatrix(output_dim, input_dim, std::sqrt(2.0 / input_dim));
	b = Matrix::zeroMatrix(1, output_dim);

	dw = Matrix(output_dim, input_dim);
	db = Matrix(1, output_dim);
	mW = Matrix::zeroMatrix(output_dim, input_dim);
	mb = Matrix::zeroMatrix(1, output_dim);
	vW = Matrix::zeroMatrix(output_dim, input_dim);
	vb = Matrix::zeroMatrix(1, output_dim);
}

Matrix Linear::forward(const Matrix& x)
{
	reshape(x.rows());
	launchForward(x);
	return forwardCache;
}

Matrix Linear::backward(const Matrix& gard)
{
	launchBackward(gard);
	return backwardRetCache;
}

void Linear::launchForward(const Matrix& x)
{
	assert(x.cols() == I);
	/*cudaMemcpy(cache.data_gpu(), x.data_gpu(), sizeof(float) * x.size(), cudaMemcpyDeviceToDevice);*/
	cache = x; // cache 如果后续明确只读，那么这样就足够，由Matrix的浅拷贝担保
	launchLinearForward(x.data_gpu(), W.data_gpu(), b.data_gpu(), forwardCache.data_gpu(),
		I, O, x.rows());
}

void Linear::launchBackward(const Matrix& grad)
{
	assert(grad.cols() == O);
	launchMatmulNN(grad.data_gpu(), W.data_gpu(), backwardRetCache.data_gpu(), 
		grad.rows(), I, O, 64, 64, 16);
	launchMatmulTN(grad.data_gpu(), cache.data_gpu(), dw.data_gpu(), O, I, grad.rows());
	launchSumCols(grad.data_gpu(), db.data_gpu(), grad.rows(), O);
}

void Linear::reshape(size_t N)
{
	if (cache.rows() != N) {
		cache = Matrix(N, I);
		forwardCache = Matrix(N, O);
		backwardRetCache = Matrix(N, I);
	}
}
