import os
import sys
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# 0 - debug, 1 - info (still a LOT of outputs), 2 - warnings, 3 - errors
os.environ["GLOG_minloglevel"] = "2"
sys.path.insert(0, "/code/caffe/python")
import caffe

gpu = 0
caffe.set_mode_gpu()
caffe.set_device(gpu)

crop_size = 176
crop_offset = 12
au_num = 12
alpha = 0.5

model_path = "./"
img_path_prefix = "../data/examples/"

# change input_param { shape: { dim: 8 dim: 3 dim: 176 dim: 176 } }
# to input_param { shape: { dim: 1 dim: 3 dim: 176 dim: 176 } } in the deploy.prototxt
au_net_model = model_path + "deploy.prototxt"
au_net_weights = model_path + "BP4D_combine_1_3.caffemodel"

if not os.path.exists(model_path + "vis_map/"):
    os.makedirs(model_path + "vis_map/")
if not os.path.exists(model_path + '/overlay_vis_map/'):
    os.makedirs(model_path + '/overlay_vis_map/')

au_net = caffe.Net(au_net_model, au_net_weights, caffe.TEST)
transformer = caffe.io.Transformer({"data": au_net.blobs["data"].data.shape})
transformer.set_transpose("data", (2, 0, 1))

# For consistency between testing and training, we use cv2 to read image
img_name = "0449"
img = cv2.imread(img_path_prefix + img_name + ".jpg", cv2.IMREAD_COLOR)  # BGR; contrarily, skimage.io.imread: RGB
img = img[crop_offset:crop_offset + crop_size, crop_offset:crop_offset + crop_size]
im = (img - 128.0) * 0.0078125
background = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

au_net.blobs["data"].data[...] = transformer.preprocess("data", im)
au_net.forward()

for i in range(au_num):
    # au_mask = au_net.blobs["au"+str(i+1)+"_mask_ori"].data[0]
    new_au_mask = au_net.blobs["au" + str(i + 1) + "_mask"].data[0]
    fig, ax = plt.subplots()
    cax = ax.imshow(new_au_mask[0, :], cmap="jet", interpolation="bicubic", vmin=0, vmax=1)

    ax.axis("off")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    # fig.savefig(model_path + "initial_vis_map/" + img_name + "_map" + str(i) + ".png", bbox_inches="tight",
    #             pad_inches=0)
    fig.savefig(model_path + "vis_map/" + img_name + "_map" + str(i) + ".png", bbox_inches="tight", pad_inches=0)

for i in range(au_num):
    overlay = Image.open(model_path + "vis_map/" + img_name + "_map" + str(i) + ".png")
    overlay = overlay.resize(background.size, Image.ANTIALIAS)
    background = background.convert('RGBA')
    overlay = overlay.convert('RGBA')
    new_img = Image.blend(background, overlay, alpha)
    new_img.save(model_path + '/overlay_vis_map/' + img_name + "_map" + str(i) + ".png", 'PNG')