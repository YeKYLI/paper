import caffe
import numpy as np
import cv2
from ../util.box import compute_iou

val_file = open("data/val-2003-720x480.txt")
prototxt = "../data/1.prototxt"
caffemodel = "../data/1.caffemodel"
ssd = caffe.Net(prototxt, caffemodel, caffe.TEST)
data_width = ssd.blobs['data'].data.shape[3]
data_height = ssd.blobs['data'].data.shape[2]

def ssd_infer(image_path):
    image = caffe.io.load_image(image_path)
    image = caffe.io.resize(image, (data_height, data_width, 3))
    image = image.transpose(2, 0 ,1)
    ssd.blobs['data'].data[...] = image
    output = ssd.forward()
    pred = np.array(ssd.blobs['detection_out3'].data[...])
    pred = np.squeeze(pred)
    #print pred.shape()
    ablation_ratio = []
    for i in range(10):
        ablation_ratio.append(pred[0][i])
    return ablation_ratio

lines = val_file.readlines()
test = 0
ablation_thresh =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ablation_count = [0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for line in lines:
    line = line.strip("\n")
    image = cv2.imread(line).shape
    image_w = image[1]
    image_h = image[0]
    test += 1
#    if test > 10:
#        break
    ablation_ratio = ssd_infer(line)
    for i in range(10):
       # print ablation_ratio[i]
        ablation_thresh[i] += ablation_ratio[i]
        if ablation_ratio[i] > 0:
            ablation_count[i] = ablation_count[i] + 1

for i in range(10):
    print str(i * 0.1) + " " + str(ablation_thresh[i] / ablation_count[i])




