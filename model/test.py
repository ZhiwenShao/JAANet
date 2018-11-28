import os
import sys
import numpy as np
import cv2
import sklearn
from sklearn.metrics import accuracy_score, f1_score

#0 - debug, 1 - info (still a LOT of outputs), 2 - warnings, 3 - errors
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0,"/code/caffe/python")
import caffe


def load_image(path, crop_offset, crop_size):
    img = cv2.imread(path,cv2.IMREAD_COLOR) # BGR
    img = img[crop_offset:crop_offset+crop_size, crop_offset:crop_offset+crop_size]
    im = (img-128.0)*0.0078125        
    return im

gpu = 0
caffe.set_mode_gpu()
caffe.set_device(gpu)

model_path = "./"
img_path_prefix = "../imgs/BP4D_aligned/"
au_net_model = model_path + "deploy.prototxt"
start_iter = 1
n_iters = 8
batch_size = 8
crop_size = 176
crop_offset = 12

data_name = "BP4D"
part_ind = 3
test_img_list = open("../"+data_name+"_part"+str(part_ind)+"_path.txt").readlines()
img_num = len(test_img_list)

test_chunk_list = [test_img_list[i:i + batch_size] for i in xrange(0, len(test_img_list), batch_size)]
AUoccur_actual = np.loadtxt("../"+data_name+"_part"+str(part_ind)+"_AUoccur.txt")
AUoccur_actual=AUoccur_actual.transpose((1,0))
res_file = open(model_path+data_name+"_part"+str(part_ind)+"_res_all_"+str(start_iter)+".txt","w")
    
for _iter in range(start_iter, n_iters+1):
    au_net_weights = model_path + "AUNet_iter_"+str(_iter)+"0000.caffemodel"
    au_net = caffe.Net(au_net_model, au_net_weights, caffe.TEST)
    transformer = caffe.io.Transformer({"data": au_net.blobs["data"].data.shape})
    transformer.set_transpose("data", (2,0,1))
    
    start = True
    for batch_ind, img_paths in enumerate(test_chunk_list):
        for ind, img_path in enumerate(img_paths):
            im = load_image(img_path_prefix+img_path.strip(), crop_offset, crop_size)       
            au_net.blobs["data"].data[ind] = transformer.preprocess("data", im)
        au_net.forward()    
        au_probAU = au_net.blobs["au_probAU"].data
        au_probAU = np.array(au_probAU)
        if start:
            AUoccur_pred_prob = au_probAU
            start = False
        else:
            AUoccur_pred_prob = np.concatenate((AUoccur_pred_prob, au_probAU))
        print _iter, batch_ind    
    AUoccur_pred_prob = AUoccur_pred_prob[0:img_num,:]
    np.savetxt(model_path+data_name+"_part"+str(part_ind)+"_predAUprob-"+str(_iter)+"_all_.txt", AUoccur_pred_prob, fmt="%f", delimiter="\t")
    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob<0.5] =0
    AUoccur_pred[AUoccur_pred_prob>=0.5] =1

    AUoccur_pred=AUoccur_pred.transpose((1,0))

    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])
    for i in range(AUoccur_actual.shape[0]):          
        curr_actual=AUoccur_actual[i]
        curr_pred=AUoccur_pred[i]
        f1score_arr[i] = f1_score(curr_actual, curr_pred)
        acc_arr[i] = accuracy_score(curr_actual, curr_pred)
        
    print>> res_file, _iter, f1score_arr.mean(), acc_arr.mean()    
res_file.close()
