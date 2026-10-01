#include "Tensor.h"

inline TensorData::TensorData(size_t length) : ref_cnt(1), length(length) {
	data = new double[length]();
}

inline TensorData::TensorData(double* data, size_t length) : ref_cnt(1), length(length), data(data) {}

inline TensorData* TensorData::copy() const {
	double* newData = new double[length];
	for (size_t i = 0; i < length; i++) {
		newData[i] = data[i];
	}
	return new TensorData(newData, length);
}

inline void TensorData::ref() {
	ref_cnt++;
}

inline void TensorData::deref() {
	ref_cnt--;
	if (ref_cnt == 0) {
		delete this;
	}
}

inline TensorData::~TensorData() {
	delete[] data;
	data = nullptr;
}

// 构造、析构、赋值

Tensor::Tensor() : data(nullptr) {}

Tensor::Tensor(constSizeVectorRef shape) :
	shape(shape), stride(compute_strides(shape)), offset(0) {
	assert(shape.size() == stride.size() && "shape and stride dimension count mismatch");
	size_t length = 1;
	for (auto s : shape)
		length *= s;
	data = new TensorData(length);
}

Tensor::Tensor(TensorData* data, constSizeVectorRef shape, constSizeVectorRef stride, size_t offset) :
	shape(shape), stride(stride), offset(offset), data(data) {
	assert(data != nullptr);
	assert([&]() {
		size_t max_idx = offset;
		for (size_t i = 0; i < shape.size(); i++) {
			if (shape[i] == 0) return false;
			max_idx += (shape[i] - 1) * stride[i];
		}
		return max_idx < data->length;
		}() && "view exceeds data bounds");
	data->ref();
}

Tensor::Tensor(const Tensor& other) :
	data(other.data), shape(other.shape), stride(other.stride), offset(other.offset)
{
	if (data) data->ref();
}

Tensor::Tensor(Tensor&& other) :
	data(other.data), shape(other.shape), stride(other.stride), offset(other.offset)
{
	other.data = nullptr;
}

Tensor& Tensor::operator=(const Tensor& other) {
	if (&other == this)
		return *this;
	if (other.data) other.data->ref();
	if (this->data) this->data->deref();
	this->data = other.data;
	this->shape = other.shape;
	this->stride = other.stride;
	this->offset = other.offset;
	return *this;
}

Tensor& Tensor::operator=(Tensor&& other) {
	if (&other == this)
		return *this;
	if (this->data) this->data->deref();
	this->data = other.data;
	this->shape = other.shape;			// 不移动拉倒
	this->stride = other.stride;
	this->offset = other.offset;
	other.data = nullptr;
	return *this;
}

double Tensor::operator()(std::initializer_list<size_t> idx) const
{
	assert(idx.size() == shape.size());
	size_t result = offset;
	size_t i = 0;
	for (auto v : idx) {
		assert(v < shape[i]);
		result += v * stride[i];
		i++;
	}
	return data->data[result];
}

double& Tensor::operator()(std::initializer_list<size_t> idx)
{
	assert(idx.size() == shape.size());
	detach();
	size_t result = offset;
	size_t i = 0;
	for (auto v : idx) {
		assert(v < shape[i]);
		result += v * stride[i];
		i++;
	}
	return data->data[result];
}

double Tensor::operator()(constSizeVectorRef idx) const
{
	return data->data[index(idx)];
}

double& Tensor::operator()(constSizeVectorRef idx)
{
	detach();
	return data->data[index(idx)];
}

double& Tensor::operator()(size_t i, size_t j)
{
	detach();
	return data->data[offset + i * stride[0] + j * stride[1]];
}

double Tensor::operator()(size_t i, size_t j) const
{
	return data->data[offset + i * stride[0] + j * stride[1]];
}


// 基础参数

inline size_t Tensor::dim() const {
	return shape.size();
}

inline size_t Tensor::size(size_t dim_idx) const {
	assert(dim_idx < shape.size(), "Dimension index out of bounds");
	return shape[dim_idx];
}

inline Tensor::constSizeVectorRef Tensor::sizes() const { return shape; }

inline size_t Tensor::numel() const {
	size_t r = 1;
	for (auto i : shape)
		r *= i;
	return r;
}

// 操作
Tensor Tensor::broadcastTo(constSizeVectorRef targetShape) const
{
	assert(!targetShape.empty(), "broadcast failed");
	size_t ds1 = shape.size();
	size_t ds2 = targetShape.size();
	assert(ds1 <= ds2, "cannot broadcastTo smaller Tensor");
	Tensor ret(data, targetShape, sizeVector(ds2), offset);
	size_t j = ds2 - 1;
	for (size_t i = ds1; i-- > 0;) {
		auto cond1 = shape[i] == targetShape[j];
		auto cond2 = shape[i] == 1;
		assert(cond1 || cond2, "broadcast failed");
		if (cond1)
			ret.stride[j] = stride[i];
		else
			ret.stride[j] = 0;
		j--;
	}
	/* for (; j-- > 0;) {
		ret.stride[j] = 0;
	} */ // 默认成立
	return ret;
}

std::pair<bool, Tensor::sizeVector> Tensor::broadcastShape(constSizeVectorRef A, constSizeVectorRef B)
{
	size_t i = A.size();
	size_t j = B.size();
	sizeVector ret;
	while (i > 0 || j > 0) {
		auto ai = i > 0 ? A[--i] : 1;
		auto bi = j > 0 ? B[--j] : 1;
		if (ai != bi && ai != 1 && bi != 1)
			return { false, {} };
		ret.push_back(std::max(ai, bi));
	}
	std::reverse(ret.begin(), ret.end());
	return { true, ret };
}

Tensor Tensor::transpose(size_t dim1, size_t dim2) const {
	Tensor t(*this);
	assert(dim1 < shape.size(), "Dimension index out of bounds");
	assert(dim2 < shape.size(), "Dimension index out of bounds");
	std::swap(t.shape[dim1], t.shape[dim2]);
	std::swap(t.stride[dim1], t.stride[dim2]);
	return t;
}

inline Tensor Tensor::slice_multi(std::initializer_list<Slice> slices) const {
	// read-only check, only in debug
	assert([&]() -> bool {
		bool* flags = new bool[shape.size()];
		for (size_t i = 0; i < shape.size(); i++)
			flags[i] = false;
		for (const auto& s : slices) {
			// also read-only check
			assert(s.dim < shape.size(), "Dimension index out of bounds");
			if (flags[s.dim]) {
				delete[] flags;
				return false;
			}
			flags[s.dim] = true;
		}
		delete[] flags;
		return true;
		}(), "Repeated slicing in the same dimension");
	Tensor t(*this);
	for (const auto& s : slices) {
		assert(s.start < shape[s.dim], "start index out of range");
		t.offset += s.start * t.stride[s.dim];
		t.shape[s.dim] = std::min(s.len, t.shape[s.dim] - s.start); // 如果步长太长那么后面的丢弃而不报错
	}
	return t;
}

inline Tensor Tensor::slice(size_t dim, size_t start, size_t len) const {
	// 切条
	return slice_multi({ { dim, start, len } });
}

inline Tensor Tensor::window(size_t dim1, size_t dim2, size_t start1, size_t start2, size_t len1, size_t len2) const {
	// 切片
	return slice_multi({ { dim1, start1, len1 },{ dim2, start2, len2 } });
}

inline bool Tensor::is_contiguous() const {
	size_t N = shape.size();
	if (N == 0) return true;
	size_t e = 1;
	for (size_t i = stride.size(); i-- > 0; ) {
		if (e != stride[i]) return false;
		e *= shape[i];
	}
	return true;
}

inline Tensor::sizeVector Tensor::compute_strides(constSizeVectorRef shape) {
	sizeVector s(shape.size());
	size_t acc = 1;
	for (size_t i = shape.size(); i-- > 0; ) {
		s[i] = acc;
		acc *= shape[i];
	}
	return s;
}

// invalid when is not contiguous

inline Tensor Tensor::view(constSizeVectorRef new_shape) const {
	assert(is_contiguous() && "view requires contiguous tensor");
	size_t new_numel = 1;
	for (auto s : new_shape) new_numel *= s;
	assert(new_numel == numel() && "view must preserve numel");

	Tensor v(*this);
	v.shape = new_shape;
	v.stride = compute_strides(new_shape);
	v.offset = offset;
	return v;
}

void Tensor::fill(double val)
{
	if (numel() == 0) return;
	if (is_contiguous()) {
		for (size_t i = 0; i < numel(); i++) {
			data->data[offset + i] = val;
		}
		return;
	}
	size_t dim_num = shape.size();
	std::vector<size_t> cnt(dim_num, 0);

	while (true) {
		bool flag = true;
		(*this)(cnt) = val;
		size_t c = 1;
		for (size_t i = dim_num; i-- > 0;) {
			cnt[i] += c;
			c = cnt[i] / shape[i];
			if (c == 0) {
				flag = false;
				break;
			}
			cnt[i] = cnt[i] % shape[i];
		}
		if (flag)
			break;
	}
}

Tensor Tensor::zeroTensor(const std::vector<size_t>& shape)
{
	Tensor ret(shape);
	if (ret.numel() == 0) return ret;
	for (size_t i = 0; i < ret.numel(); i++) {
		ret.data->data[i] = 0;
	}
	return ret;
}

Tensor Tensor::randomTensor(const std::vector<size_t>& shape)
{
	Tensor ret(shape);
	if (ret.numel() == 0) return ret;
	for (size_t i = 0; i < ret.numel(); i++) {
		ret.data->data[i] = randn();
	}
	return ret;
}

Tensor Tensor::add(const Tensor& A, const Tensor& B)
{
	auto common = Tensor::broadcastShape(A.shape, B.shape);
	assert(common.first, "broadcast failed");
	auto A_ = A.shape == common.second ? A : A.broadcastTo(common.second);
	auto B_ = B.shape == common.second ? B : B.broadcastTo(common.second);
	Tensor C(common.second);
	if (C.numel() == 0) return C;
	if (A_.is_contiguous() && B_.is_contiguous()) { 
		for (size_t i = 0; i < A_.numel(); i++) {
			C.data->data[i] = A_.data->data[A_.offset + i] + B_.data->data[B_.offset + i];
		}
		return C;
	}// 如果A和B本身就可以加，那么广播相当于什么也没做
	size_t dim_num = A_.shape.size();
	std::vector<size_t> cnt(dim_num, 0);

	while (true) {
		bool overflow = true;
		C(cnt) = A_(cnt) + B_(cnt);
		size_t c = 1;
		for (size_t i = dim_num; i-- > 0;) {
			cnt[i] += c;
			c = cnt[i] / A_.shape[i];
			if (c == 0) {
				overflow = false;
				break;
			}
			cnt[i] = cnt[i] % A_.shape[i];
		}
		if (overflow)
			break;
	}
	return C;
}

Tensor Tensor::sub(const Tensor& A, const Tensor& B)
{
	auto common = Tensor::broadcastShape(A.shape, B.shape);
	assert(common.first, "broadcast failed");
	auto A_ = A.shape == common.second ? A : A.broadcastTo(common.second);
	auto B_ = B.shape == common.second ? B : B.broadcastTo(common.second);
	Tensor C(common.second);
	if (C.numel() == 0) return C;
	if (A_.is_contiguous() && B_.is_contiguous()) {
		for (size_t i = 0; i < A_.numel(); i++) {
			C.data->data[i] = A_.data->data[A_.offset + i] - B_.data->data[B_.offset + i];
		}
		return C;
	}// 如果A和B本身就可以加，那么广播相当于什么也没做
	size_t dim_num = A_.shape.size();
	std::vector<size_t> cnt(dim_num, 0);

	while (true) {
		bool overflow = true;
		C(cnt) = A_(cnt) - B_(cnt);
		size_t c = 1;
		for (size_t i = dim_num; i-- > 0;) {
			cnt[i] += c;
			c = cnt[i] / A_.shape[i];
			if (c == 0) {
				overflow = false;
				break;
			}
			cnt[i] = cnt[i] % A_.shape[i];
		}
		if (overflow)
			break;
	}
	return C;
}

Tensor Tensor::mm(const Tensor& A, const Tensor& B) // only matrix
{
	assert(A.shape.size() == B.shape.size(), "Dimension mismatch for tensor mm");
	assert(A.shape.size() == 2, "mm only for matrix");
	size_t M = A.shape[0];
	size_t K = A.shape[1];
	assert(B.shape[0] == K, "Dimension mismatch for tensor mm");
	size_t N = B.shape[1];
	Tensor C = Tensor::zeroTensor({ M, N });
	for (size_t i = 0; i < M; i++) {
		for (size_t k = 0; k < K; k++) {
			double aval = A(i, k);
			for (size_t j = 0; j < N; j++) {
				C(i, j) += aval * B(k, j);
			}
		}
	}
	return C;
}

Tensor Tensor::scalarMul(const Tensor& A, double value)
{
	Tensor C(A.sizes());
	if (A.numel() == 0) return C;
	if (A.is_contiguous()) {
		for (size_t i = 0; i < A.numel(); i++) {
			C.data->data[i] = A.data->data[A.offset + i] * value;
		}
		return C;
	}
	size_t dim_num = A.shape.size();
	std::vector<size_t> cnt(dim_num, 0);

	while (true) {
		bool overflow = true;
		C(cnt) = A(cnt) * value;
		size_t c = 1;
		for (size_t i = dim_num; i-- > 0;) {
			cnt[i] += c;
			c = cnt[i] / A.shape[i];
			if (c == 0) {
				overflow = false;
				break;
			}
			cnt[i] = cnt[i] % A.shape[i];
		}
		if (overflow)
			break;
	}
	return C;
}

Tensor Tensor::elementwiseMul(const Tensor& A, const Tensor& B)
{
	auto common = Tensor::broadcastShape(A.shape, B.shape);
	assert(common.first, "broadcast failed");
	auto A_ = A.shape == common.second ? A : A.broadcastTo(common.second);
	auto B_ = B.shape == common.second ? B : B.broadcastTo(common.second);
	Tensor C(common.second);
	if (C.numel() == 0) return C;
	if (A_.is_contiguous() && B_.is_contiguous()) {
		for (size_t i = 0; i < A_.numel(); i++) {
			C.data->data[i] = A_.data->data[A_.offset + i] * B_.data->data[B_.offset + i];
		}
		return C;
	}// 如果A和B本身就可以加，那么广播相当于什么也没做
	size_t dim_num = A_.shape.size();
	std::vector<size_t> cnt(dim_num, 0);

	while (true) {
		bool overflow = true;
		C(cnt) = A_(cnt) * B_(cnt);
		size_t c = 1;
		for (size_t i = dim_num; i-- > 0;) {
			cnt[i] += c;
			c = cnt[i] / A_.shape[i];
			if (c == 0) {
				overflow = false;
				break;
			}
			cnt[i] = cnt[i] % A_.shape[i];
		}
		if (overflow)
			break;
	}
	return C;
}

Tensor Tensor::operator+(const Tensor& other) const
{
	return Tensor::add(*this, other);
}

Tensor Tensor::operator-(const Tensor& other) const
{
	return Tensor::sub(*this, other);
}

Tensor Tensor::operator*(double value) const
{
	return Tensor::scalarMul(*this, value);
}

Tensor Tensor::elementwiseMul(const Tensor& other) const
{
	return Tensor::elementwiseMul(*this, other);
}

Tensor Tensor::mm(const Tensor& other) const
{
	return Tensor::mm(*this, other);
}

Tensor Tensor::collectRowsByIndex(const std::vector<size_t>& indexes) const {
	assert(dim() == 2 && "collectRowsByIndex requires 2D tensor");
	size_t M = std::min(indexes.size(), size(0));
	size_t N = size(1);

	Tensor ret({ M, N });
	for (size_t i = 0; i < M; i++) {
		size_t idx = indexes[i];
		assert(idx < size(0));
		for (size_t j = 0; j < N; j++)
			ret(i, j) = (*this)(idx, j);
	}
	return ret;
}

double Tensor::sum() const
{
	double ret = 0;
	if (is_contiguous()) {
		for (size_t i = 0; i < numel(); i++) {
			ret += data->data[offset + i];
		}
		return ret;
	}
	size_t dim_num = shape.size();
	std::vector<size_t> cnt(dim_num, 0);

	while (true) {
		bool overflow = true;
		ret += (*this)(cnt);
		size_t c = 1;
		for (size_t i = dim_num; i-- > 0;) {
			cnt[i] += c;
			c = cnt[i] / shape[i];
			if (c == 0) {
				overflow = false;
				break;
			}
			cnt[i] = cnt[i] % shape[i];
		}
		if (overflow)
			break;
	}
	return ret;
}

// 索引方法
inline size_t Tensor::index(constSizeVectorRef idx) const {
	assert(idx.size() == shape.size(), "indexes are invalid");
	for (size_t i = 0; i < idx.size(); i++) {
		assert(idx[i] < shape[i], "index out of range");
	}
	size_t result = offset;
	for (size_t i = 0; i < idx.size(); i++) {
		result += idx[i] * stride[i];
	}
	return result;
}

//CoW

inline void Tensor::detach() {
	if (!data) return;
	if (data->ref_cnt > 1) {
		auto newData = data->copy();
		data->deref();
		data = newData;
	}
}
