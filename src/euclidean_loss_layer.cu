#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

// Normalized with a value (inter-ocular distance)
template <typename Dtype>
void EuclideanLoss2Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();
	caffe_gpu_sub(
		count,
		bottom[0]->gpu_data(),
		bottom[1]->gpu_data(),
		diff_.mutable_gpu_data());
	Dtype dot;
	caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
	Dtype loss = dot / bottom[0]->num() / Dtype(2) / bottom[2]->cpu_data()[0];
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLoss2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num() / bottom[2]->cpu_data()[0];
			caffe_gpu_axpby(
				bottom[i]->count(),              // count
				alpha,                              // alpha
				diff_.gpu_data(),                   // a
				Dtype(0),                           // beta
				bottom[i]->mutable_gpu_diff());  // b
		}
	}
}

// Normalized with a value (inter-ocular distance) and weighted
template <typename Dtype>
void EuclideanLoss3Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();
	caffe_gpu_sub(
		count,
		bottom[0]->gpu_data(),
		bottom[1]->gpu_data(),
		diff_.mutable_gpu_data());
	for (int j = 0; j < count; j++)
	{
		diff_.mutable_cpu_data()[j] = diff_.cpu_data()[j] * bottom[3]->cpu_data()[j];
	}
	Dtype dot;
	caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
	Dtype loss = dot / bottom[0]->num() / Dtype(2) / bottom[2]->cpu_data()[0];
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLoss3Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num() / bottom[2]->cpu_data()[0];
			caffe_gpu_axpby(
				bottom[i]->count(),              // count
				alpha,                              // alpha
				diff_.gpu_data(),                   // a
				Dtype(0),                           // beta
				bottom[i]->mutable_gpu_diff());  // b
		}
	}
}

// Weighted
template <typename Dtype>
void EuclideanLoss4Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();
	caffe_gpu_sub(
		count,
		bottom[0]->gpu_data(),
		bottom[1]->gpu_data(),
		diff_.mutable_gpu_data());
	for (int j = 0; j < count; j++)
	{
		//if (abs(bottom[1]->cpu_data()[j] - 1.8)<1e-4) //label is 9*0.2 (missing)  
		//	diff_.mutable_cpu_data()[j] = 0;
		//else
			diff_.mutable_cpu_data()[j] = diff_.cpu_data()[j] * bottom[2]->cpu_data()[j];
	}
	Dtype dot;
	caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
	Dtype loss = dot / bottom[0]->num() / Dtype(2);
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLoss4Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
			caffe_gpu_axpby(
				bottom[i]->count(),              // count
				alpha,                              // alpha
				diff_.gpu_data(),                   // a
				Dtype(0),                           // beta
				bottom[i]->mutable_gpu_diff());  // b
		}
	}
}

// Weighted; According to the AU intensity to select weights
template <typename Dtype>
void EuclideanLoss5Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();
	caffe_gpu_sub(
		count,
		bottom[0]->gpu_data(),
		bottom[1]->gpu_data(),
		diff_.mutable_gpu_data());
	for (int j = 0; j < count; j++)
	{
		if (abs(bottom[1]->cpu_data()[j] - 1.8)<1e-4) //label is 9*0.2 (missing)  
			diff_.mutable_cpu_data()[j] = 0;
		else
    {
      int ind = j*6 + int(round(bottom[1]->cpu_data()[j]*5));
			diff_.mutable_cpu_data()[j] = diff_.cpu_data()[j] * bottom[2]->cpu_data()[ind];
    }
	}
	Dtype dot;
	caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
	Dtype loss = dot / bottom[0]->num() / Dtype(2);
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLoss5Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
			caffe_gpu_axpby(
				bottom[i]->count(),              // count
				alpha,                              // alpha
				diff_.gpu_data(),                   // a
				Dtype(0),                           // beta
				bottom[i]->mutable_gpu_diff());  // b
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);
INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLoss2Layer);
INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLoss3Layer);
INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLoss4Layer);
INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLoss5Layer);

}  // namespace caffe
