#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

// Normalized with a value (inter-ocular distance)
template <typename Dtype>
void EuclideanLoss2Layer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::Reshape(bottom, top);
	CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
		<< "Inputs must have the same dimension.";
	diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLoss2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();
	caffe_sub(
		count,
		bottom[0]->cpu_data(),
		bottom[1]->cpu_data(),
		diff_.mutable_cpu_data());
	Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
	Dtype loss = dot / bottom[0]->num() / Dtype(2) / bottom[2]->cpu_data()[0];
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLoss2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num() / bottom[2]->cpu_data()[0];
			caffe_cpu_axpby(
				bottom[i]->count(),              // count
				alpha,                              // alpha
				diff_.cpu_data(),                   // a
				Dtype(0),                           // beta
				bottom[i]->mutable_cpu_diff());  // b
		}
	}
}

// Normalized with a value (inter-ocular distance) and weighted
template <typename Dtype>
void EuclideanLoss3Layer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::Reshape(bottom, top);
	CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
		<< "Inputs must have the same dimension.";
	diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLoss3Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();
	caffe_sub(
		count,
		bottom[0]->cpu_data(),
		bottom[1]->cpu_data(),
		diff_.mutable_cpu_data());
	for (int j = 0; j < count; j++)
	{
		diff_.mutable_cpu_data()[j] = diff_.cpu_data()[j] * bottom[3]->cpu_data()[j];
	}
	Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
	Dtype loss = dot / bottom[0]->num() / Dtype(2) / bottom[2]->cpu_data()[0];
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLoss3Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num() / bottom[2]->cpu_data()[0];
			caffe_cpu_axpby(
				bottom[i]->count(),              // count
				alpha,                              // alpha
				diff_.cpu_data(),                   // a
				Dtype(0),                           // beta
				bottom[i]->mutable_cpu_diff());  // b
		}
	}
}

// Weighted
template <typename Dtype>
void EuclideanLoss4Layer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::Reshape(bottom, top);
	CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
		<< "Inputs must have the same dimension.";
	diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLoss4Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();
	caffe_sub(
		count,
		bottom[0]->cpu_data(),
		bottom[1]->cpu_data(),
		diff_.mutable_cpu_data());
	for (int j = 0; j < count; j++)
	{
		//if (abs(bottom[1]->cpu_data()[j] - 1.8)<1e-4) //label is 9*0.2 (missing)  
		//	diff_.mutable_cpu_data()[j] = 0;
		//else
			diff_.mutable_cpu_data()[j] = diff_.cpu_data()[j] * bottom[2]->cpu_data()[j];
	}
	Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
	Dtype loss = dot / bottom[0]->num() / Dtype(2);
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLoss4Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
			caffe_cpu_axpby(
				bottom[i]->count(),              // count
				alpha,                              // alpha
				diff_.cpu_data(),                   // a
				Dtype(0),                           // beta
				bottom[i]->mutable_cpu_diff());  // b
		}
	}
}

// Weighted; According to the AU intensity to select weights
template <typename Dtype>
void EuclideanLoss5Layer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::Reshape(bottom, top);
	CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
		<< "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->count(1)*6, bottom[2]->count(1))
		<< "Weights must have the dimension of all intensities."; 
	diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLoss5Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();
	caffe_sub(
		count,
		bottom[0]->cpu_data(),
		bottom[1]->cpu_data(),
		diff_.mutable_cpu_data());
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
	Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
	Dtype loss = dot / bottom[0]->num() / Dtype(2);
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLoss5Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
			caffe_cpu_axpby(
				bottom[i]->count(),              // count
				alpha,                              // alpha
				diff_.cpu_data(),                   // a
				Dtype(0),                           // beta
				bottom[i]->mutable_cpu_diff());  // b
		}
	}
}


#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
STUB_GPU(EuclideanLoss2Layer);
STUB_GPU(EuclideanLoss3Layer);
STUB_GPU(EuclideanLoss4Layer);
STUB_GPU(EuclideanLoss5Layer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);
INSTANTIATE_CLASS(EuclideanLoss2Layer);
REGISTER_LAYER_CLASS(EuclideanLoss2);
INSTANTIATE_CLASS(EuclideanLoss3Layer);
REGISTER_LAYER_CLASS(EuclideanLoss3);
INSTANTIATE_CLASS(EuclideanLoss4Layer);
REGISTER_LAYER_CLASS(EuclideanLoss4);
INSTANTIATE_CLASS(EuclideanLoss5Layer);
REGISTER_LAYER_CLASS(EuclideanLoss5);


}  // namespace caffe
