# JAANet
This repository implements the training and testing of JAA-Net for "[Deep Adaptive Attention for Joint Facial Action Unit Detection and Face Alignment](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhiwen_Shao_Deep_Adaptive_Attention_ECCV_2018_paper.pdf)". The repository offers the original implementation of the paper in Caffe.

# Getting Started
## Dependencies
- Dependencies for Caffe (http://caffe.berkeleyvision.org/install_apt.html) are required.

- The new implementations in the folders "src" and "include" should be merged into the official Caffe:
  - add the .cpp, .cu files into "src/caffe/layers"
  - add the .hpp files into "include/caffe/layers"
  - add the content of "caffe.proto" into "src/caffe/proto"

- New implementations used in our paper:
  - au_mask_based_land_layer: generate attention maps given the locations of landmarks
  - division_layer: divide a feature map into multiple identical subparts
  - combination_layer: combine mutiple sub feature maps
  - data_layer: the processing of landmarks in the case of mirroring faces is added
  - align_data_transform_layer: reset the order and change the coordinates for landmarks in the cases of mirroring and cropping
  - dice_coef_loss_layer: Dice coefficient loss
  - softmax_loss_layer: the weighting for the loss of each element is added
  - euclidean_loss_layer: the weighting for the loss of each element and the normalizing with inter-ocular distance are added
  
# Waitting to be updated
