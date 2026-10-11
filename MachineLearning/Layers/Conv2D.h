#pragma once

#include "Layer.h"

class Conv2D : public Layer
{
public:
	// 参数初始化函数由Matrix类提供，此处暂时不改
	Conv2D(size_t H, size_t W, 
		size_t inputChannel, size_t outputChannel,
		size_t ksize, size_t stride = 1, size_t padding = 0) :
		stride(stride), padding(padding), H(H), W(W), ksize(ksize),
		inputChannel(inputChannel), outputChannel(outputChannel)
	{
		OW = (W + 2 * padding - ksize) / stride + 1;
		OH = (H + 2 * padding - ksize) / stride + 1;
		kernel = Matrix::randnMatrix(outputChannel, inputChannel * ksize * ksize,
			std::sqrt(2.0 / (inputChannel * ksize * ksize)));
		bias = Matrix::zeroMatrix(1, outputChannel);
		dW = Matrix(outputChannel, inputChannel * ksize * ksize);
		db = Matrix(1, outputChannel); 
		vW = Matrix::zeroMatrix(outputChannel, inputChannel * ksize * ksize);
		vb = Matrix::zeroMatrix(1, outputChannel);
	}
	virtual Matrix forward(const Matrix& x) override {
		reshape(x.rows());
		launchForward(x);
		return forwardCache;
	}
	virtual Matrix backward(const Matrix& grad) override {
		launchBackward(grad);
		return backwardCache;
	}
	virtual std::vector<ParamPtr> getParams() {
		return std::vector<ParamPtr>{ { &kernel, &dW, &vW }, { &bias, &db, &vb }};
	}
private:
	void launchForward(const Matrix& x) {
		launchIm2col(x.data_gpu(), A.data_gpu(), x.rows(), inputChannel, H, W, OH, OW, ksize, stride, padding);
		launchConv2DGemmKernel(A.data_gpu(), kernel.data_gpu(), bias.data_gpu(), forwardCache.data_gpu(),
			x.rows(), OH, OW, inputChannel, outputChannel, ksize);
	}
	void launchBackward(const Matrix& grad) {
		// grad is (N, (O OH OW)), dP is ((N OH OW), O)
		// dW is (O (I K K)), A is ((N OH OW), (I K K))
		launchConv2DBackwardReshape(grad.data_gpu(), dP.data_gpu(), grad.rows(), outputChannel, OH, OW);
		launchMatmulTN(dP.data_gpu(), A.data_gpu(), dW.data_gpu(), dW.rows(), dW.cols(), A.rows());
		launchSumCols(dP.data_gpu(), db.data_gpu(), dP.rows(), outputChannel);
		launchMatmulNN(dP.data_gpu(), kernel.data_gpu(), dA.data_gpu(), dA.rows(), dA.cols(), kernel.rows());
		launchFillInplace(backwardCache.data_gpu(), backwardCache.size(), 0.0f);
		launchCol2im(dA.data_gpu(), backwardCache.data_gpu(), grad.rows(), 
			inputChannel, H, W, OH, OW, ksize, stride, padding);
	}
	void reshape(size_t N) {
		if (N * OH * OW != A.rows()) {
			A = Matrix(N * OH * OW, inputChannel * ksize * ksize);
			dA = Matrix(N * OH * OW, inputChannel * ksize * ksize);
			dP = Matrix(N * OH * OW, outputChannel);
			forwardCache = Matrix(N, outputChannel * OH * OW);
			backwardCache = Matrix(N, inputChannel * H * W);
		}
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
	Matrix dP;
	Matrix forwardCache;
	Matrix backwardCache;
	// momentum
	Matrix vW;
	Matrix vb;

	size_t stride;
	size_t padding;
	size_t inputChannel;
	size_t outputChannel;
	size_t ksize;
	size_t H;
	size_t W;
	size_t OH;
	size_t OW;
};

