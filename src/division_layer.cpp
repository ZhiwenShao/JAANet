#include <algorithm>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/division_layer.hpp"


namespace caffe {


template <typename Dtype>
void DivisionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{

    LOG(INFO)<< "top.size is  " <<top.size();
    count_ = bottom[0]->count();

	width_ = static_cast<int>(this->layer_param_.division_param().width());
	height_ = static_cast<int>(this->layer_param_.division_param().height());

    num_ = top.size();
    channels_ = bottom[0]->channels();   

    const DivisionParameter& division_param = this->layer_param_.division_param();
    xcoord_.clear();
    std::copy(division_param.xcoord().begin(),
    division_param.xcoord().end(),
      std::back_inserter(xcoord_));

    ycoord_.clear();
    std::copy(division_param.ycoord().begin(),
    division_param.ycoord().end(),
      std::back_inserter(ycoord_));
    
}


template <typename Dtype>
void DivisionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
   for (int i = 0; i < num_; ++i) {
      top[i]->Reshape(bottom[0]->num(), bottom[0]->channels(),
                       height_, width_);
    }
}


/* copy only clipped region */
template <typename Dtype>
void DivisionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data = bottom[0]->cpu_data(); 

	int height_bottom = bottom[0]->height();
	int width_bottom = bottom[0]->width();

	for (int n = 0; n < num_; n++) {

		Dtype* top_data = top[n]->mutable_cpu_data();
		int v = ycoord_[n];
		int w = xcoord_[n];
		for(int b = 0; b < bottom[0]->num(); b++){

			for(int c = 0; c< channels_; c++) {
        
				for(int h = v; h<v+height_; h++) {
          
					int index_bottom =  b*channels_*height_bottom*width_bottom+c*height_bottom*width_bottom + h*width_bottom + w;
					int index_top = b*channels_*height_*width_+c*height_*width_ + (h-v)*width_ + 0;
          
					for (int i = 0; i <width_; i++) {
						top_data[index_top + i] = bottom_data[index_bottom + i];
					}
				}
			}
		}
	}
  
}
/* copy only clipped region */

template <typename Dtype>
void DivisionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    if (!propagate_down[0]) { return; }
      
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    int height_bottom = bottom[0]->height();
    int width_bottom = bottom[0]->width();

    for (int n = 0; n < num_; n++) {

		const Dtype* top_diff = top[n]->cpu_diff();
		int v = ycoord_[n];
		int w = xcoord_[n];
		for (int b = 0; b < bottom[0]->num(); b++){
          
			for(int c = 0; c< channels_; c++) {
            
				for(int h = v; h<v+height_; h++) {
            
					int index_bottom = b*channels_*height_bottom*width_bottom+c*height_bottom*width_bottom + h*width_bottom + w;
					int index_top = b*channels_*height_*width_+c*height_*width_ + (h-v)*width_ + 0;
					for (int i = 0; i < width_; i++) {
						bottom_diff[index_bottom + i] = top_diff[index_top + i];
					}
				}
			}
		}
    }
}

#ifdef CPU_ONLY
STUB_GPU(DivisionLayer);
#endif
INSTANTIATE_CLASS(DivisionLayer);
REGISTER_LAYER_CLASS(Division);
}