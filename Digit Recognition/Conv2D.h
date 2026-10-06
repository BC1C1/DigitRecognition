#pragma once

#include "Layer.h"

class Conv2D : public Layer
{
public:
	Conv2D(size_t H, size_t W, 
		size_t inputChannel, size_t outputChannel,
		size_t ksize, size_t stride = 1, size_t padding = 0) :
		stride(stride), padding(padding), H(H), W(W), ksize(ksize),
		inputChannel(inputChannel), outputChannel(outputChannel)
	{
		kernel = Matrix::randnMatrix(outputChannel, inputChannel * ksize * ksize,
			std::sqrt(2.0 / (inputChannel * ksize * ksize)));
		bias = Matrix::zeroMatrix(1, outputChannel);
		dW = Matrix(outputChannel, inputChannel * ksize * ksize);
		// db = Matrix(1, outputChannel); sum_cols 返回新矩阵，缓存等跑通再说
	}
	virtual Matrix forward(const Matrix& x) override {
		assert(x.cols() == inputChannel * H * W);
		size_t OW = (W + 2 * padding - ksize) / stride + 1;
		size_t OH = (H + 2 * padding - ksize) / stride + 1;
		A = x.im2col(x.rows(), inputChannel, H, W, ksize, stride, padding);
		Matrix out = Matrix::zeroMatrix(x.rows() * OH * OW, outputChannel);
		Matrix::matmul_nt(A, kernel, out);
		out = out + bias;
		// reshape out
		Matrix result(x.rows(), outputChannel * OH * OW);
		for (size_t n = 0; n < x.rows(); n++) {
			for (size_t oh = 0; oh < OH; oh++) {
				for (size_t ow = 0; ow < OW; ow++) {
					for (size_t oc = 0; oc < outputChannel; oc++) {
						result(n, oc * OW * OH + oh * OW + ow) =
							out(n * OH * OW + oh * OW + ow, oc);
					}
				}
			}
		}// ohow顺序没变，我觉得是可以写的更高效的，这里就这样
		return result;
	}
	virtual Matrix backward(const Matrix& grad) override {
		size_t OW = (W + 2 * padding - ksize) / stride + 1;
		size_t OH = (H + 2 * padding - ksize) / stride + 1;
		size_t N = grad.rows();
		// grad is (N, (O OH OW)), dP is ((N OH OW), O)
		Matrix dP(N * OH * OW, outputChannel);
		for (size_t n = 0; n < N; n++) {
			for (size_t oh = 0; oh < OH; oh++) {
				for (size_t ow = 0; ow < OW; ow++) {
					for (size_t oc = 0; oc < outputChannel; oc++) {
						dP(n * OH * OW + oh * OW + ow, oc) =
							grad(n, oc * OW * OH + oh * OW + ow);
					}
				}
			}
		}
		// dw
		dW.fill(0);
		Matrix::matmul_tn(dP, A, dW);
		// db
		db = dP.sum_cols();
		// dA ((N OH OW),(C K K))
		dA = Matrix::zeroMatrix(N * OH * OW, ksize * ksize * inputChannel);
		Matrix::matmul_nn(dP, kernel, dA);
		return dA.col2im(N, inputChannel, H, W, ksize, stride, padding);
	}
	virtual void update(double learning_rate) override {
		for (size_t i = 0; i < kernel.rows(); i++)
			for (size_t j = 0; j < kernel.cols(); j++)
				kernel(i, j) -= learning_rate * dW(i, j);
		for (size_t i = 0; i < bias.rows(); i++)
			for (size_t j = 0; j < bias.cols(); j++)
				bias(i, j) -= learning_rate * db(i, j);
	}
private:
	// param
	Matrix kernel;
	Matrix bias;
	Matrix dW;
	Matrix db;
	// cache
	Matrix A;
	Matrix dA;

	size_t stride;
	size_t padding;
	size_t inputChannel;
	size_t outputChannel;
	size_t ksize;
	size_t H;
	size_t W;
};

