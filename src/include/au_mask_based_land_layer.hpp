#ifndef CAFFE_AUMaskBasedLand_LAYER_HPP_
#define CAFFE_AUMaskBasedLand_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	template <typename Dtype>
	class AUMaskBasedLandLayer : public Layer<Dtype> {
	public:
		explicit AUMaskBasedLandLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "AUMaskBasedLand"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 13; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int img_size_, mask_size_, AU_num_, num_, dim_, top_dim_;
		Dtype spatial_ratio_, spatial_scale_, half_AU_size_;
		Dtype fill_coeff_, fill_value_;
	};

}  // namespace caffe

#endif  // CAFFE_AUMaskBasedLand_LAYER_HPP_
