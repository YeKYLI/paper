import caffe
import numpy as np
import cv2

val_file = open("data/val-2003-720x480.txt")
prototxt = "data/1.prototxt"
caffemodel = "data/1.caffemodel"
ssd = caffe.Net(prototxt, caffemodel, caffe.TEST)
data_width = ssd.blobs['data'].data.shape[3]
data_height = ssd.blobs['data'].data.shape[2]

#if area_1 composed of area_2 , warning #####################
#here i have somt questions

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
def ssd_infer(image_path):
    image = caffe.io.load_image(image_path)
    image = caffe.io.resize(image, (data_height, data_width, 3))
    image = image.transpose(2, 0 ,1)
    ssd.blobs['data'].data[...] = image
    output = ssd.forward()
    pred = np.array(ssd.blobs['detection_out1'].data[...])
    pred = np.squeeze(pred)
    pred_boxes = []
    for i in range(len(pred)):
        if pred[i][2] < 0.1:
            continue
        pred_boxes.append(pred[i][3])
        pred_boxes.append(pred[i][4])
        pred_boxes.append(pred[i][5])
        pred_boxes.append(pred[i][6])
    return pred_boxes
        

lines = val_file.readlines()
box_iou = []
test = 0
for line in lines:
    line = line.strip("\n")
    image = cv2.imread(line).shape
    image_w = image[1]
    image_h = image[0]
    test += 1
    if test > 1000:
        break
    pred_boxes = ssd_infer(line)
    print len(pred_boxes)
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

accurate = 0
for i in range(len(box_iou)):
    if(box_iou[i] > 0.5):
        accurate += 1
print accurate
print len(box_iou)
AP50 = float(accurate) / float(len(box_iou)) 
print AP50

