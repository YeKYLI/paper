import caffe
import numpy as np

val_file = open("data/val-2003-720x480.txt")
prototxt = "data/1.prototxt"
caffemodel = "data/1.caffemodel"

#if area_1 composed of area_2 , warning #####################
def compute_iou(x1min, y1min, x1max, y1max, x2min, y2min, x2max, y2max):
    if x1min > x2max or x1max < x2min:
        return 0
    else:
        xmin = max(x1min, x2min)
        xmax = min(x1max, x2max)
    if y1min > y2max or y1max < y2min:
        return 0
    else:
        ymin = max(y1min, y2min)
        ymax = min(y1max, y2max)
    area_1 = (x1max - x1min) * (y1max - y1min)
    area_2 = (x2max - x2min) * (y2max - y2min)
    area_iou = (xmax - xmin) * (ymax - ymin)
    iou = float(area_iou) / float(area_1 + area_2 - area_iou)
    return iou

#get th pred through ssd infer
def ssd_infer(prototxt_, caffemodel_, image_path):
    ssd = caffe.Net(prototxt_, caffemodel_, caffe.TEST)
    data_width = ssd.blobs['data'].data.shape[3]
    data_height = ssd.blobs['data'].data.shape[2]
    image = caffe.io.load_image(image_path)
    image = caffe.io.resize(image, (data_height, data_width, 3))
    image = image.transpose(2, 0 ,1)
    ssd.blobs['data'].data[...] = image
    output = ssd.forward()
    pred = np.array(ssd.blobs['detection_out1'].data[...])
    pred = np.squeeze(pred)
    print pred
        

lines = val_file.readlines()
for line in lines:
    ssd_infer(prototxt, caffemodel, line.strip("\n"))
    line = line.replace("images", "original_labels")
    line = line.replace("jpg", "txt").strip("\n")
    gts = open(line)
    gt_lines = gts.readlines()
    for gt_line in gt_lines:
        gt_line = gt_line.strip("\n").split(" ")

        




#for line in lines:
#    print line
