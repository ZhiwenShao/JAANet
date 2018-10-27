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
  
## Datasets
[BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) and [DISFA](http://www.engr.du.edu/mmahoor/DISFA.htm)

The 3-fold partitions of both BP4D and DISFA are provided in the folder "data".

## Preprocessing
- Run "face_transform.cpp" to conduct similarity transformation for face images.
- Run "convert_imageset" of Caffe to convert the images to leveldb or lmdb
- Prepare the training data and modify the paths in the "model/BP4D_train_val.prototxt"
  - A recommended training strategy is that selecting a small set of training data for validation to choose a proper maximum iterations and then using all the training data to retrain the model
  - The loss_weight for DiceCoefLoss of each AU is the normalized weight computed from the training data
  - The lr_mult for au_mask_conv corresponds to the enhancement coefficient \lambda_3, and the loss_weight of au_mask_loss is related to the reconstruction constraint E_r and \lambda_3
   - \lambda_3 = 1:
    param {
      lr_mult: 1
      decay_mult: 1
    }
    param {
      lr_mult: 2
      decay_mult: 0
    }
    loss_weight: 1e-7
    
   - \lambda_3 = 2:
    param {
      lr_mult: 2
      decay_mult: 1
    }
    param {
      lr_mult: 4
      decay_mult: 0
    }
    loss_weight: 5e-8
  
## Training
```
cd model
sh train_net.sh
```



# Waitting to be updated
