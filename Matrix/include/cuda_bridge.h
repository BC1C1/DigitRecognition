#pragma once
#include <cstddef>

// ============================================================================
//  cuda_bridge.h 启动函数声明
// ============================================================================

extern "C" {

	void launchFillInplace(float* g, size_t n, float value);
	void launchAdd(float* ga, float* gb, float* gc, size_t n);
	void launchSub(float* ga, float* gb, float* gc, size_t n);
	void launchSubInPlace(float* ga, float* gb, size_t n);
	void launchScaleMul(float* ga, float* gb, size_t n, float value);
	void launchScaleMulInPlace(float* g, size_t n, float value);
	void launchElementwiseMul(float* ga, float* gb, float* gc, size_t n);
	void launchElementwiseMulInPlace(float* ga, float* gb, size_t n);
	void launchMatmulNN(float* ga, float* gb, float* gc, size_t M, size_t N, size_t K);
	void launchMatmulNT(float* ga, float* gb, float* gc, size_t M, size_t N, size_t K);
	void launchMatmulTN(float* ga, float* gb, float* gc, size_t M, size_t N, size_t K);
	void launchSumRows(float* ga, float* gb, size_t M, size_t N);
	void launchSumCols(float* ga, float* gb, size_t M, size_t N);
	double launchSumKernel(float* g, size_t n);
	void launchIm2col(float* ga, float* gb, size_t N, size_t C, size_t H, size_t W, size_t OH, size_t OW,
		size_t ksize, size_t stride, size_t padding);
	void launchCol2im(float* ga, float* gb, size_t N, size_t C, size_t H, size_t W, size_t OH, size_t OW,
		size_t ksize, size_t stride, size_t padding);
	// kernel for layers
	void launchLinearForward(float* x, float* W, float* b, float* result,
		size_t in_dim, size_t out_dim, size_t N);
	void launchReLUForward(float* x, float* out, size_t N, size_t D);
	void launchReLUBackward(float* grad, float* cache, float* out, size_t N, size_t D);
	void launchConv2DGemmKernel(float* A, float* kernel, float* bias, float* out,
		size_t N, size_t OH, size_t OW, size_t I, size_t O, size_t K);
	void launchConv2DBackwardReshape(float* grad, float* dP,
		size_t N, size_t O, size_t OH, size_t OW);
	void launchMaxPool2DForward(float* x, float* out, size_t N, size_t C, size_t H, size_t W,
		size_t ksize, size_t stride, size_t* maxIndex);
	void launchMaxPool2DBackward(float* grad, float* dX, size_t N, size_t C, size_t H, size_t W,
		size_t ksize, size_t stride, size_t* maxIndex);
	// kernel for lossfunction
	double launchMSELossForward(float* pred, float* real, size_t M, size_t N);
	void launchMSELossBackwardKernel(float* pred, float* real, float* out, size_t M, size_t N);
	void launchLogSoftmax(float* input, float* cache, float* out, size_t M, size_t N);
	// kernel for optimizer
	void launchGeneralUpdateWithMomentum(float* value, float* grad, float* velocity,
		size_t n, float learning_rate, float momentum);
}
