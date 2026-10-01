#pragma once

#include <vector>
#include <cassert>
#include <algorithm>

#include "Utils.h"

// 创建后会立刻被Tensor持有，而且一定是new出来的，tensor不需要管它的释放时机
struct TensorData {
	double* data;
	size_t ref_cnt;
	size_t length;
	TensorData(size_t length);
	TensorData(double* data, size_t length);
	TensorData* copy() const;
	void ref();
	void deref();
	~TensorData();
};

class Tensor
{
private:
	using sizeVector = std::vector<size_t>;
	using constSizeVectorRef = const std::vector<size_t>&;
public:
	// 构造、析构、赋值
	Tensor();
	Tensor(constSizeVectorRef shape);

	Tensor(TensorData* data, constSizeVectorRef shape, constSizeVectorRef stride, size_t offset);

	Tensor(const Tensor& other);

	Tensor(Tensor&& other);

	~Tensor() { if (data) data->deref(); }

	Tensor& operator=(const Tensor& other);

	Tensor& operator=(Tensor&& other);

	// 访问
	double operator()(std::initializer_list<size_t> idx) const;

	double& operator()(std::initializer_list<size_t> idx);

	double operator()(constSizeVectorRef idx) const;

	double& operator()(constSizeVectorRef idx);

	double operator()(size_t i) const { return (*this)({ i }); }
	double& operator()(size_t i) { return (*this)({ i }); }

	double& operator()(size_t i, size_t j);

	double operator()(size_t i, size_t j) const;

	// 基础参数
	size_t dim() const;

	size_t size(size_t dim_idx) const;

	constSizeVectorRef sizes() const;

	size_t numel() const;

	// 操作
	Tensor broadcastTo(constSizeVectorRef targetShape) const;

	static std::pair<bool, sizeVector> broadcastShape(constSizeVectorRef A, constSizeVectorRef B);

	Tensor transpose(size_t dim1, size_t dim2) const;

	struct Slice
	{
		size_t dim; size_t start; size_t len;
	};

	Tensor slice_multi(std::initializer_list<Slice> slices) const;

	Tensor slice(size_t dim, size_t start, size_t len) const;

	Tensor window(size_t dim1, size_t dim2, size_t start1, size_t start2, size_t len1, size_t len2) const;

	bool is_contiguous() const;

	static sizeVector compute_strides(constSizeVectorRef shape);

	// invalid when is not contiguous
	Tensor view(constSizeVectorRef new_shape) const;

	void fill(double val);

	static Tensor zeroTensor(const std::vector<size_t>& shape);

	static Tensor randomTensor(const std::vector<size_t>& shape);

	// op
	static Tensor add(const Tensor& A, const Tensor& B);

	static Tensor sub(const Tensor& A, const Tensor& B);

	static Tensor mm(const Tensor& A, const Tensor& B); // only matrix

	static Tensor scalarMul(const Tensor& A, double value);

	static Tensor elementwiseMul(const Tensor& A, const Tensor& B);

	Tensor operator+(const Tensor& other) const;

	Tensor operator-(const Tensor& other) const;

	Tensor operator*(double value) const;

	Tensor elementwiseMul(const Tensor& other) const;

	Tensor mm(const Tensor& other) const;

	Tensor collectRowsByIndex(const std::vector<size_t>& indexes) const;

	double sum() const;

	template <typename F>
	void foreach_do(F func) {
		if (numel() == 0) return;

		if (is_contiguous()) {
			for (size_t i = 0; i < numel(); i++)
				data->data[offset + i] = func(data->data[offset + i]);
			return;
		}

		size_t dim_num = shape.size();
		std::vector<size_t> cnt(dim_num, 0);

		while (true) {
			size_t phys = index(cnt);
			data->data[phys] = func(data->data[phys]);

			size_t c = 1;
			bool overflow = true;
			for (size_t i = dim_num; i-- > 0;) {
				cnt[i] += c;
				c = cnt[i] / shape[i];
				if (c == 0) { overflow = false; break; }
				cnt[i] = cnt[i] % shape[i];
			}
			if (overflow) break;
		}
	}

private:
	sizeVector shape;
	sizeVector stride;
	size_t offset;
	TensorData* data;

	// 索引方法
	size_t index(constSizeVectorRef idx) const;

	//CoW
	void detach();
};

