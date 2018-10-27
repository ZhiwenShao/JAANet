#include <vector>

#include "caffe/layers/au_mask_based_land_layer.hpp"

namespace caffe {

template <typename Dtype>
void AUMaskBasedLandLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	AUMaskBasedLandParameter au_mask_based_land_param = this->layer_param_.au_mask_based_land_param();

	img_size_ = au_mask_based_land_param.img_size();
	spatial_ratio_ = au_mask_based_land_param.spatial_ratio();
	spatial_scale_ = au_mask_based_land_param.spatial_scale();
	fill_coeff_ = au_mask_based_land_param.fill_coeff();
	fill_value_ = au_mask_based_land_param.fill_value();

	AU_num_ = 12;
	half_AU_size_ = (img_size_ - 1) / 2.0 * spatial_ratio_;

	num_ = bottom[0]->num();
	dim_ = bottom[0]->count() / num_;
	mask_size_ = static_cast<int>(round(img_size_ * spatial_scale_));

}


template <typename Dtype>
void AUMaskBasedLandLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	for (int j = 0; j < AU_num_ + 1; ++j)
	{
		top[j]->Reshape(num_, 1,
			mask_size_, mask_size_);
		int count = top[j]->count();
		caffe_set(count, fill_value_, top[j]->mutable_cpu_data());
	}
	top_dim_ = top[0]->count() / num_;
}

template <typename Dtype>
void AUMaskBasedLandLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* all_au_mask = top[AU_num_]->mutable_cpu_data();

	Dtype AU_center_x, AU_center_y;
	int start_w, start_h, end_w, end_h;

	for (int i = 0; i < num_; ++i) {
		Dtype ruler = abs(bottom_data[i * dim_ + 22 * 2] - bottom_data[i * dim_ + 25 * 2]);
		for (int j = 0; j < AU_num_ * 2; ++j) {
			Dtype* top_data = top[j / 2]->mutable_cpu_data();

			if (j == 0) {
				AU_center_x = bottom_data[i * dim_ + 4 * 2];
				AU_center_y = bottom_data[i * dim_ + 4 * 2 + 1] - ruler / 2;
			}
			if (j == 1) {
				AU_center_x = bottom_data[i * dim_ + 5 * 2];
				AU_center_y = bottom_data[i * dim_ + 5 * 2 + 1] - ruler / 2;
			}

			if (j == 2) {
				AU_center_x = bottom_data[i * dim_ + 1 * 2];
				AU_center_y = bottom_data[i * dim_ + 1 * 2 + 1] - ruler / 3;
			}
			if (j == 3) {
				AU_center_x = bottom_data[i * dim_ + 8 * 2];
				AU_center_y = bottom_data[i * dim_ + 8 * 2 + 1] - ruler / 3;			
			}

			if (j == 4) {
				AU_center_x = bottom_data[i * dim_ + 2 * 2];
				AU_center_y = bottom_data[i * dim_ + 2 * 2 + 1] + ruler / 3;
			}
			if (j == 5) {
				AU_center_x = bottom_data[i * dim_ + 7 * 2];
				AU_center_y = bottom_data[i * dim_ + 7 * 2 + 1] + ruler / 3;
			}

			if (j == 6) {
				AU_center_x = bottom_data[i * dim_ + 24 * 2];
				AU_center_y = bottom_data[i * dim_ + 24 * 2 + 1] + ruler;
			}
			if (j == 7) {
				AU_center_x = bottom_data[i * dim_ + 29 * 2];
				AU_center_y = bottom_data[i * dim_ + 29 * 2 + 1] + ruler;
			}

			if (j == 8) {
				AU_center_x = bottom_data[i * dim_ + 21 * 2];
				AU_center_y = bottom_data[i * dim_ + 21 * 2 + 1];
			}
			if (j == 9) {
				AU_center_x = bottom_data[i * dim_ + 26 * 2];
				AU_center_y = bottom_data[i * dim_ + 26 * 2 + 1];
			}

			if (j == 10) {
				AU_center_x = bottom_data[i * dim_ + 43 * 2];
				AU_center_y = bottom_data[i * dim_ + 43 * 2 + 1];
			}
			if (j == 11) {
				AU_center_x = bottom_data[i * dim_ + 45 * 2];
				AU_center_y = bottom_data[i * dim_ + 45 * 2 + 1];
			}

			if (j == 12 || j == 14 || j == 16) {
				AU_center_x = bottom_data[i * dim_ + 31 * 2];
				AU_center_y = bottom_data[i * dim_ + 31 * 2 + 1];
			}
			if (j == 13 || j == 15 || j == 17) {
				AU_center_x = bottom_data[i * dim_ + 37 * 2];
				AU_center_y = bottom_data[i * dim_ + 37 * 2 + 1];
			}

			if (j == 18) {
				AU_center_x = bottom_data[i * dim_ + 39 * 2];
				AU_center_y = bottom_data[i * dim_ + 39 * 2 + 1] + ruler / 2;
			}
			if (j == 19) {
				AU_center_x = bottom_data[i * dim_ + 41 * 2];
				AU_center_y = bottom_data[i * dim_ + 41 * 2 + 1] + ruler / 2;
			}

			if (j == 20 || j == 22) {
				AU_center_x = bottom_data[i * dim_ + 34 * 2];
				AU_center_y = bottom_data[i * dim_ + 34 * 2 + 1];
			}
			if (j == 21 || j == 23) {
				AU_center_x = bottom_data[i * dim_ + 40 * 2];
				AU_center_y = bottom_data[i * dim_ + 40 * 2 + 1];
			}

			start_w = round((AU_center_x - half_AU_size_) * spatial_scale_);
			start_h = round((AU_center_y - half_AU_size_) * spatial_scale_);
			end_w = round((AU_center_x + half_AU_size_) * spatial_scale_);
			end_h = round((AU_center_y + half_AU_size_) * spatial_scale_);

			start_h = fmax(start_h, 0);
			start_h = fmin(start_h, mask_size_ - 1);
			start_w = fmax(start_w, 0);
			start_w = fmin(start_w, mask_size_ - 1);
			end_h = fmax(end_h, 0);
			end_h = fmin(end_h, mask_size_ - 1);
			end_w = fmax(end_w, 0);
			end_w = fmin(end_w, mask_size_ - 1);
	
			AU_center_x = AU_center_x * spatial_scale_;
			AU_center_y = AU_center_y * spatial_scale_;

			for (int h = start_h; h <= end_h; h++) {
				for (int w = start_w; w <= end_w; w++) {
					if (this->layer_param_.au_mask_based_land_param().fill_type() == "manhattan") {
						top_data[i*top_dim_ + h*mask_size_ + w] = fmax(1 - (abs(h - AU_center_y) + abs(w - AU_center_x))*fill_coeff_*(100.0 / (img_size_*spatial_scale_)), fill_value_);
						all_au_mask[i*top_dim_ + h*mask_size_ + w] = fmax(all_au_mask[i*top_dim_ + h*mask_size_ + w], top_data[i*top_dim_ + h*mask_size_ + w]);
					}				
				}
			}
		}
	}
}

template <typename Dtype>
void AUMaskBasedLandLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	return;
}

#ifdef CPU_ONLY
STUB_GPU(AUMaskBasedLandLayer);
#endif
INSTANTIATE_CLASS(AUMaskBasedLandLayer);
REGISTER_LAYER_CLASS(AUMaskBasedLand);

}  // namespace caffe
