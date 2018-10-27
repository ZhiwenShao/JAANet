# JAANet
This repository implements the training and testing of JAA-Net for "[Deep Adaptive Attention for Joint Facial Action Unit Detection and Face Alignment](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhiwen_Shao_Deep_Adaptive_Attention_ECCV_2018_paper.pdf)". The repository offers the original implementation of the paper in Caffe.

# Getting Started
## Dependencies
Dependencies for Caffe (http://caffe.berkeleyvision.org/install_apt.html) are required.
-The new implementations in the folders "src" and "include" should be merged into the official Caffe:
(1) add the .cpp, .cu files into "src/caffe/layers"
(2) add the .hpp files into "include/caffe/layers"
(3) add the content of "caffe.proto" into "src/caffe/proto"

-New implementations used in our paper:
(1) au_mask_based_land_layer: generate attention maps given the locations of landmarks
(2) division_layer: divide a feature map into multiple identical subparts
(3) combination_layer: combine mutiple sub feature maps
(4) data_layer: the processing of landmarks in the case of mirroring faces is added
(5) align_data_transform_layer: reset the order and change the coordinates for landmarks in the cases of mirroring and cropping
(6) dice_coef_loss_layer: Dice coefficient loss
(7) softmax_loss_layer: the weighting for the loss of each element is added

