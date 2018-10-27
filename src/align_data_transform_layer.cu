#include <vector>

#include "caffe/layers/align_data_transform_layer.hpp"

namespace caffe {

template <typename Dtype>
void AlignDataTransformLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	Forward_cpu(bottom, top);
}


template <typename Dtype>
void AlignDataTransformLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(AlignDataTransformLayer);


}  // namespace caffe
