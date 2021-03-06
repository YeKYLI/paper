import caffe
import cv2
import numpy as np

#hyperparameter
caffemodel = "bdd.caffemodel"
#caffemodel = "data/1.caffemodel"
prototxt = "bdd.prototxt"
#prototxt = "data/1.prototxt"
#imagename = "/home/data4t/zhijiang/dataset/car-720x480/images/val-2003-720x480/day_normal_test_2/day_qingdao_07141619_0115.MP4_0027.jpg"
imagename = "/home/data4t/zhijiang/dataset/car-720x480/images/val-2003-720x480/day_normal_test_2/2015_0207_180105_032.MOV_0028.jpg"
#imagename = "data/5.jpg"
anch = [10, 55, 99, 144]
anch_num = 8
aspect_ratio = 1.15
ssd = caffe.Net(prototxt, caffemodel, caffe.TEST)
data_height = ssd.blobs['data'].data.shape[2]
data_width = ssd.blobs['data'].data.shape[3]
feature_w = ssd.blobs['conv4_3'].data.shape[3]
feature_h = ssd.blobs['conv4_3'].data.shape[2]
grid_w = float(data_width) / float(feature_w)
grid_h = float(data_height) / float(feature_h)

#forward the ssd network
image = caffe.io.load_image(imagename)
image = caffe.io.resize(image, (data_height, data_width, 3))
image = image.transpose(2, 0 ,1)
ssd.blobs['data'].data[...] = image
output = ssd.forward()

#paint the results
image = cv2.imread(imagename)
image = cv2.resize(image, (data_width, data_height), interpolation=cv2.INTER_CUBIC)
#paint the grid
for i in range(feature_w):
    cv2.line(image, (int((i + 1) * grid_w), 0), (int((i + 1) * grid_w), data_height), (255, 255, 255), 1)
for j in range(feature_h):
    cv2.line(image, (0, int((j + 1) * grid_h)), (data_width, int((j + 1) * grid_h)), (255, 255, 255), 1)
#paint the predicted box
pred = np.array(ssd.blobs['detection_out1'].data[...])
pred = np.squeeze(pred)
print "predict : " + str(len(pred)) + " boxes"

#test
count = 0
for i in range(feature_h):
    for j in range(feature_w):
        for k in range(anch_num):
            if pred[anch_num * (feature_w * i + j) + k][2] > 0.01:
                count = count + 1
                print str(count) + " " + str(i)+ " " + str(j)+ " " + str(k) + " "  + str(pred[anch_num * (feature_w * i + j) + k][2])

test = 0;
for i in range(len(pred)):
    if pred[i][2] > 0.01 and pred[i][1] == 1:
        test = test + 1
        print str(test) +" " +  "confidence id: " + str(i) + " " + str(pred[i][1]) + " " + str(pred[i][2]) + " " + str(pred[i][3] * float(data_width)) + " " + str(pred[i][4] * float(data_height)) + " " + str(pred[i][6] * float(data_height))
        cv2.rectangle(image, (int(pred[i][3] * float(data_width)), int(pred[i][4] * float(data_height))), (int((pred[i][5]) * float(data_width)), int(pred[i][6] * float(data_height))), (0, 0, 255), 1)
#paint the anchor
for i in anch:
    cv2.rectangle(image, (0, 0), (i, i), (0, 0, 0), 2)
    cv2.rectangle(image, (0, 0), (int(i * 1 ** aspect_ratio), int(i / (1 ** aspect_ratio))), (0, 0, 0), 2)
#paint the prior box
anchor = np.array(ssd.blobs['detection_out2'].data[...])
anchor = np.squeeze(anchor)
for j in range(len(anchor)):
    if pred[j][2] > 0.01 and pred[j][1] == 1:
        cv2.rectangle(image, (int(anchor[j][3] * float(data_width)), int(anchor[j][4] * float(data_height))), (int((anchor[j][5]) * float(data_width)), int(anchor[j][6] * float(data_height))), (0, 255, 0), 1);

cv2.imshow("out", image)
cv2.waitKey(0)
