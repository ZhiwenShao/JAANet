# JAANet
This repository implements the training and testing of JAA-Net for "[Deep Adaptive Attention for Joint Facial Action Unit Detection and Face Alignment](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhiwen_Shao_Deep_Adaptive_Attention_ECCV_2018_paper.pdf)". The repository offers the original implementation of the paper in Caffe, and the [PyTorch implementation](https://github.com/ZhiwenShao/PyTorch-JAANet) is also released

# Getting Started
## Dependencies
- Dependencies for [Caffe](http://caffe.berkeleyvision.org/install_apt.html) are required

- The new implementations in the folders "src" and "include" should be merged into the official [Caffe](https://github.com/BVLC/caffe):
  - Add the .cpp, .cu files into "src/caffe/layers"
  - Add the .hpp files into "include/caffe/layers"
  - Add the content of "caffe.proto" into "src/caffe/proto"
  - Add "tools/convert_data.cpp" into "tools"
- New implementations used in our paper:
  - au_mask_based_land_layer: generate attention maps given the locations of landmarks
  - division_layer: divide a feature map into multiple identical subparts
  - combination_layer: combine mutiple sub feature maps
  - data_layer and data_transform_layer: the processing of landmarks in the case of mirroring faces is added
  - align_data_transform_layer: reset the order and change the coordinates for landmarks in the cases of mirroring and cropping
  - dice_coef_loss_layer: Dice coefficient loss
  - softmax_loss_layer: the weighting for the loss of each element is added
  - euclidean_loss_layer: the weighting for the loss of each element and the normalizing with inter-ocular distance are added
  - convert_data: convert the AU and landmark labels, weights, and reflect_49 to leveldb or lmdb
- Build Caffe

## Datasets
[BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) and [DISFA](http://www.engr.du.edu/mmahoor/DISFA.htm)

The 3-fold partitions of both BP4D and DISFA are provided in the folder "data", in which the path file of DISFA contains each frame of videos

## Preprocessing
- Prepare the training data
  - Run "prep/face_transform.cpp" to conduct similarity transformation for face images
  - Run "prep/write_biocular.m" to compute the inter-ocular distance of each face image
  - Run "prep/combine2parts.m" to combine two partitions as a training set, respectively
  - Run "prep/write_AU_weight.m" to compute the weight of each AU for the training set
  - Run "tools/convert_imageset" of Caffe to convert the images to leveldb or lmdb
  - Run "tools/convert_data" to convert the AU and landmark labels, inter-ocular distances, weights and reflect_49 to leveldb or lmdb: the example format of files for AU and landmark labels, inter-ocular distances and weights are in "data/examples"; reflect_49.txt is in "data"; the weights are shared by all the training samples (only one line needed); reflect_49 is used to reset the order and change the coordinates for landmarks in the case of face mirroring
  - Our method is evaluated by 3-fold cross validation. For example, “BP4D_combine_1_2” denotes the combination of partition 1 and partition 2
- Modify the "model/BP4D_train_val.prototxt":
  - Modify the paths of data
  - A recommended training strategy is that selecting a small set of training data for validation to choose a proper maximum iterations and then using all the training data to retrain the model
  - The loss_weight for DiceCoefLoss of each AU is the normalized weight computed from the training data
  - The lr_mult for "au*_mask_conv3*" corresponds to the enhancement coefficient "\lambda_3", and the loss_weight of "au*_mask_loss" is related to the reconstruction constraint "E_r" and "\lambda_3"
```
   When \lambda_3 = 1:
   
    param {
      lr_mult: 1
      decay_mult: 1
    }
    param {
      lr_mult: 2
      decay_mult: 0
    }
    
    loss_weight: 1e-7
```
```
   When \lambda_3 = 2:
   
    param {
      lr_mult: 2
      decay_mult: 1
    }
    param {
      lr_mult: 4
      decay_mult: 0
    }
    
    loss_weight: 5e-8
```
- There are two minor differences from the original paper without performance degradation:
  - The redundant cropping of attention maps is removed
  - The first convolution of the third block uses the stride of 2 instead of 1 for smaller model complexity

## Training
```
cd model
sh train_net.sh
```
- Trained models on BP4D with 3-fold cross-validation can be downloaded [here](https://sjtueducn-my.sharepoint.com/:f:/g/personal/shaozhiwen_sjtu_edu_cn/EhVWf3EgvnNLj-o_fWT3InEBCcI9vlnxkrOkSOUxRAzkAg?e=XN9v67)

## Testing
- Compute evaluation metrics
```
python test.py
```
- Visualize attention maps
```
python visualize_attention_map.py
```

## Citation
If you use this code for your research, please cite our paper
```
@inproceedings{shao2018deep,
  title={Deep Adaptive Attention for Joint Facial Action Unit Detection and Face Alignment},
  author={Shao, Zhiwen and Liu, Zhilei and Cai, Jianfei and Ma, Lizhuang},
  booktitle={European Conference on Computer Vision},
  year={2018},
  pages={725--740},
  organization={Springer}
}
```

# Acknowledgments
Code is partially inspired by [DRML](https://github.com/zkl20061823/DRML) and [A-Variation-of-Dice-coefficient-Loss-Caffe-Layer](https://github.com/HolmesShuan/A-Variation-of-Dice-coefficient-Loss-Caffe-Layer)
