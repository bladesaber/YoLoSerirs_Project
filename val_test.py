import cv2
import numpy as np
from Utils import visualize
import matplotlib.pyplot as plt
import os

data_dir = 'D:\\DataSet\\coco_val2017\\val2017'
with open('D:\\coursera\\YoLoSerirs\\data\\val2017.txt', 'r') as f:
    data = f.readlines()

inds = np.random.randint(0, len(data), 20)
for ind in inds:
    img_data = data[ind].split(' ')
    path = img_data[0]
    image_path = os.path.join(data_dir, path.split('/')[-1])
    if os.path.exists(image_path)==False:
        raise ValueError

    bboxes = []
    for line in img_data[1:]:
        x1, y1, x2, y2, label = line.split(',')
        bboxes.append([int(x1), int(y1), int(x2), int(y2), 1.0, int(label)])
        labels = label
    bboxes = np.array(bboxes)

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    img = visualize.draw_bbox(original_image, bboxes)

    plt.imshow(img)
    plt.show()
