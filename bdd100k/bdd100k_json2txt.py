import json
json_path = "/home/huaijian/data/bdd100k/labels/bdd100k_labels_images_val.json"
#json_path = "/home/huaijian/data/bdd100k/labels/bdd100k_labels_images_val.json"
json_file = json.load(open(json_path, "r"))
out_dir = "/home/huaijian/data/bdd100k/original_labels"
categories = ["bike","bus","car","motor","person","rider","traffic light","traffic sign","train","truck"]

for image in json_file:
    outfile = open(out_dir + "/" + image['name'].strip("jpg") + "txt", "w")
    for label in image['labels']:
 #       print label['category']
        if 'box2d'not in label:
            continue
        box2d = label['box2d']
        if box2d['x1'] >= box2d['x2'] or box2d['y1'] >= box2d['y2']:
            continue
        x1 = int(box2d['x1'])
        x2 = int(box2d['x2'])
        y1 = int(box2d['y1'])
        y2 = int(box2d['y2'])
#        print categories.index('car')
        outfile.write(str(categories.index(label['category'])) + " " + str(x1) + " " +  str(y1) + " " +  str(x2) + " " + str(y2) + '\n')
    outfile.close()





