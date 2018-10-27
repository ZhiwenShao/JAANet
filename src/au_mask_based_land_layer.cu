#include <vector>

#include "caffe/layers/au_mask_based_land_layer.hpp"

namespace caffe {

template <typename Dtype>
void AUMaskBasedLandLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	Forward_cpu(bottom, top);
}


template <typename Dtype>
void AUMaskBasedLandLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(AUMaskBasedLandLayer);


}  // namespace caffe
