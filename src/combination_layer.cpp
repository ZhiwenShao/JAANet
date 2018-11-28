#include <vector>

#include "caffe/layers/combination_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



template <typename Dtype>
void CombinationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    num_ = bottom.size();
    LOG(INFO)<<" the bottom size is "<<num_;
      
    const CombinationParameter& combination_param = this->layer_param_.combination_param();
    xcoord_.clear();
    std::copy(combination_param.xcoord().begin(),
    combination_param.xcoord().end(),
      std::back_inserter(xcoord_));

    ycoord_.clear();
    std::copy(combination_param.ycoord().begin(),
    combination_param.ycoord().end(),
      std::back_inserter(ycoord_));
    
	height_ = ycoord_[num_ - 1] + bottom[num_ - 1]->height();
	width_ = xcoord_[num_ - 1] + bottom[num_ - 1]->width();
}


template <typename Dtype>
void CombinationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

      top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
		  height_, width_);

}

/* copy only clipped region */
template <typename Dtype>
void CombinationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	Dtype* top_data = top[0]->mutable_cpu_data();
	int height_bottom, width_bottom, v, w;

	for (int n = 0; n < num_; n++) {//The nth bottom to be concatenated.
		const Dtype* bottom_data = bottom[n]->cpu_data();    
		
		v = ycoord_[n];
		w = xcoord_[n];
		height_bottom = bottom[n]->height();
		width_bottom = bottom[n]->width();

		for(int b = 0; b < bottom[0]->num(); b++){

			for(int c = 0; c< bottom[0]->channels(); c++) {
        
				for(int h = v; h<v+height_bottom; h++) {

					int index_bottom = b*bottom[0]->channels()*height_bottom*width_bottom+c*height_bottom*width_bottom + (h-v)*width_bottom;
					int index_top = b*bottom[0]->channels()*height_*width_+c*height_*width_+ h*width_ + w;
          
					for (int i = 0; i <width_bottom; i++) {
						top_data[index_top + i] = bottom_data[index_bottom + i];
					}
				}
			}
		}
	}
}
/* copy only clipped region */

template <typename Dtype>
void CombinationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    if (!propagate_down[0]) { return; }

    const Dtype* top_diff = top[0]->cpu_diff();
	int height_bottom, width_bottom, v, w;

    for (int n = 0; n < num_; n++) {

		Dtype* bottom_diff = bottom[n]->mutable_cpu_diff(); 
         
		v = ycoord_[n];
		w = xcoord_[n];
		height_bottom = bottom[n]->height();
		width_bottom = bottom[n]->width();

		for (int b = 0; b < bottom[0]->num(); b++){
          
			for(int c = 0; c< bottom[0]->channels(); c++) {
            
				for(int h = v; h<v+height_bottom; h++) {

					int index_bottom = b*bottom[0]->channels()*height_bottom*width_bottom+c*height_bottom*width_bottom + (h-v)*width_bottom;
					int index_top = b*bottom[0]->channels()*height_*width_+c*height_*width_+ h*width_ + w;
               
					for (int i = 0; i < width_bottom; i++) {
						bottom_diff[index_bottom + i] = top_diff[index_top + i];
					}       
				}// for height and width
			}
		}
    }
}

#ifdef CPU_ONLY
STUB_GPU(CombinationLayer);
#endif
INSTANTIATE_CLASS(CombinationLayer);
REGISTER_LAYER_CLASS(Combination);
}