import matplotlib.pylab as plt
import numpy as np
import matplotlib.patches as patches

def draw_box(img, bboxs):
    fig, ax = plt.subplots()
    ax.imshow(img)

    for box in bboxs:
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

with open('D:\\coursera\\YoLoSerirs_Project\\data\\fddb_face_train.txt', 'r') as f:
    data = f.readlines()

for line in data:
    data_list = line.split(' ')
    path = data_list[0]

    bboxs = []
    for bbox in data_list[1:]:
        xmin, ymin, xmax, ymax, label = bbox.split(',')
        xmin, ymin, xmax, ymax, label = int(xmin), int(ymin), int(xmax), int(ymax), int(label)
        bboxs.append([xmin, ymin, xmax, ymax])

    img = plt.imread(path)
    draw_box(img, bboxs)
