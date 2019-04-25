import os

image_dir = "/home/huaijian/data/bdd100k/images/100k/test"
outfile = "100k_test.txt"

outfile = open(outfile, "w")

for image in os.listdir(image_dir):
    outfile.write(image_dir + "/" + image + "\n")

outfile.close()
