#include <vector>

#include "caffe/layers/align_data_transform_layer.hpp"

namespace caffe {

template <typename Dtype>
void AlignDataTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	num_ = bottom[0]->num();
	dim_ = bottom[0]->count() / num_;
}


template <typename Dtype>
void AlignDataTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
		bottom[0]->height(), bottom[0]->width());

}

template <typename Dtype>
void AlignDataTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* mirror_crop_para = bottom[1]->cpu_data();
  const Dtype* index = bottom[2]->cpu_data();

  Dtype* top_data = top[0]->mutable_cpu_data();
  vector<Dtype> swap_data(dim_);

  for (int i = 0; i < num_; ++i) {
	  for (int j = 0; j < dim_; j += 2) {
		  top_data[i * dim_ + j] = bottom_data[i * dim_ + j] - mirror_crop_para[i * 4 + 3];
		  top_data[i * dim_ + j + 1] = bottom_data[i * dim_ + j + 1] - mirror_crop_para[i * 4 + 2];
		  if (mirror_crop_para[i * 4]) {
			  top_data[i * dim_ + j] = mirror_crop_para[i * 4 + 1] - 1 - top_data[i * dim_ + j];
		  }		  
	  }
	  if (mirror_crop_para[i * 4]) {
		  for (int j = 0; j < dim_; j += 2) {
			  swap_data[j] = top_data[i * dim_ + (2 * int(index[j / 2]) - 2)];
			  swap_data[j + 1] = top_data[i * dim_ + (2 * int(index[j / 2]) - 1)];
		  }
		  for (int j = 0; j < dim_; j += 2) {
			  top_data[i * dim_ + j] = swap_data[j];
			  top_data[i * dim_ + j + 1] = swap_data[j + 1];
		  }
	  }
  }
}

template <typename Dtype>
void AlignDataTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
	const Dtype* mirror_crop_para = bottom[1]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* index = bottom[2]->cpu_data();

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	vector<Dtype> swap_data(dim_);

	for (int i = 0; i < num_; ++i) {
		for (int j = 0; j < dim_; j += 2) {
			bottom_diff[i * dim_ + j] = top_diff[i * dim_ + j];
			bottom_diff[i * dim_ + j + 1] = top_diff[i * dim_ + j + 1];
			if (mirror_crop_para[i * 4]) {
				bottom_diff[i * dim_ + j] = -bottom_diff[i * dim_ + j];
			}
		}
		if (mirror_crop_para[i * 4]) {
			for (int j = 0; j < dim_; j += 2) {
				swap_data[j] = bottom_diff[i * dim_ + (2 * int(index[j / 2]) - 2)];
				swap_data[j + 1] = bottom_diff[i * dim_ + (2 * int(index[j / 2]) - 1)];
			}
			for (int j = 0; j < dim_; j += 2) {
				bottom_diff[i * dim_ + j] = swap_data[j];
				bottom_diff[i * dim_ + j + 1] = swap_data[j + 1];
			}
		}
	}
  }
}

#ifdef CPU_ONLY
STUB_GPU(AlignDataTransformLayer);
#endif
INSTANTIATE_CLASS(AlignDataTransformLayer);
REGISTER_LAYER_CLASS(AlignDataTransform);

}  // namespace caffe
