import caffe
import numpy as np
import cv2
from util.box import compute_iou

val_file = open("data/val-2003-720x480.txt")
prototxt = "data/1.prototxt"
caffemodel = "data/1.caffemodel"
ssd = caffe.Net(prototxt, caffemodel, caffe.TEST)
data_width = ssd.blobs['data'].data.shape[3]
data_height = ssd.blobs['data'].data.shape[2]

def ssd_infer(image_path):
    image = caffe.io.load_image(image_path)
    image = caffe.io.resize(image, (data_height, data_width, 3))
    image = image.transpose(2, 0 ,1)
    ssd.blobs['data'].data[...] = image
    output = ssd.forward()
    pred = np.array(ssd.blobs['detection_out1'].data[...])
    pred = np.squeeze(pred)
    total_predict = np.array(ssd.blobs['detection_out2'].data[...])
    total_predict = np.squeeze(total_predict)
    total_predict = int(total_predict[0][0])
    pred_boxes = []
    for i in range(total_predict):
        print "test : " + str(i)
        pred_boxes.append(pred[i][3])
        pred_boxes.append(pred[i][4])
        pred_boxes.append(pred[i][5])
        pred_boxes.append(pred[i][6])
    return pred_boxes, total_predict

lines = val_file.readlines()
box_iou = []
num_all = 0
test = 0
for line in lines:
    line = line.strip("\n")
    image = cv2.imread(line).shape
    image_w = image[1]
    image_h = image[0]
    test += 1
#    if test > 10:
#        break
    pred_boxes, num = ssd_infer(line)
    num_all += num
    line = line.replace("images", "original_labels")
    line = line.replace("jpg", "txt")
    gts = open(line)
    gt_lines = gts.readlines()
    for i in range(len(pred_boxes) / 4):
        best_iou = 0
        for gt_line in gt_lines:
            gt_line = gt_line.strip("\n").split(" ")
            iou = compute_iou(pred_boxes[4 * i], pred_boxes[4 * i + 1], pred_boxes[4 * i + 2], pred_boxes[4 * i + 3], float(gt_line[1]) / float(image_w), float(gt_line[2]) / float(image_h), float(gt_line[3]) / float(image_w), float(gt_line[4]) / float(image_h))
            if best_iou < iou:
                best_iou = iou
        box_iou.append(best_iou)
        print best_iou

#compute map
accurate = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(box_iou)):
    for j in range(10):
        if box_iou[i] > (0.5 + 0.05 * j):
            accurate[j] += 1
AP = 0;
for j in range(10):
    AP_J = float(accurate[j]) / float(len(box_iou))
    print str(j) + " " + str(AP_J)
    AP += AP_J
AP /= 10
print AP

