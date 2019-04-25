import caffe
import cv2
import numpy as np

#hyperparameter
caffemodel = "data/1.caffemodel"
prototxt = "data/1.prototxt"
#imagename = "/home/huaijian/data/bdd100k/images/100k/train/76fe3a3f-5027eaa7.jpg"
imagename = "2.jpg"
#imagename = "data/3.jpg"
ssd = caffe.Net(prototxt, caffemodel, caffe.TEST)
data_height = ssd.blobs['data'].data.shape[2]
data_width = ssd.blobs['data'].data.shape[3]

#forward the ssd network
image = caffe.io.load_image(imagename)
image = caffe.io.resize(image, (data_height, data_width, 3))
image = image.transpose(2, 0 ,1)
ssd.blobs['data'].data[...] = image
output = ssd.forward()

#paint the results
image = cv2.imread(imagename)
image = cv2.resize(image, (data_width, data_height), interpolation=cv2.INTER_CUBIC)

#paint the predicted box
pred = np.array(ssd.blobs['detection_out1'].data[...])
pred = np.squeeze(pred)
print "predict : " + str(len(pred)) + " boxes"

test = 0;
for i in range(len(pred)):
    if pred[i][2] > 0.01:
        test = test + 1
        print str(test) + " " +  "confidence id: " + str(i) + " " + str(pred[i][1]) + " " + str(pred[i][2]) + " " + str(pred[i][3] * float(data_width)) + " " + str(pred[i][4] * float(data_height)) + " " + str(pred[i][6] * float(data_height))
        cv2.rectangle(image, (int(pred[i][3] * float(data_width)), int(pred[i][4] * float(data_height))), (int((pred[i][5]) * float(data_width)), int(pred[i][6] * float(data_height))), (0, 0, 255), 1)

cv2.imshow("out", image)
cv2.waitKey(0)
