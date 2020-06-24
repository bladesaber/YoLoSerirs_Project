import numpy as np
import os
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)

def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed(99)

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break
        else:
            print(np.mean(np.abs(last_clusters-nearest_clusters)))

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def read_data_wh(xml_path):
    tree = ET.parse(xml_path)
    height = int(tree.findtext("./size/height"))
    width = int(tree.findtext("./size/width"))
    return height, width

def voc_demision_cluster():
    xml_dir = 'D:/DataSet/VOCtrainval_06-Nov-2007/VOC2007/Annotations/'

    with open('D:\coursera\YoLoSerirs\data/voc2007_train.txt', 'r') as f:
        data = f.readlines()

    boxes = []
    for line in data:
        cells = line.split(' ')
        img_path = cells[0]
        regex = img_path.split('/')[-1].replace('.jpg', '')
        xml_path = os.path.join(xml_dir, regex + '.xml')
        height, width = read_data_wh(xml_path)

        for cell in cells[1:]:
            xmin, ymin, xmax, ymax, label = cell.split(',')
            xmin, ymin, xmax, ymax, label = float(xmin), float(ymin), float(xmax), float(ymax), int(label)
            assert xmax > xmin and ymax > ymin
            boxes.append([(xmax - xmin) / width, (ymax - ymin) / height])
    boxes = np.array(boxes)

    out = kmeans(boxes, k=6)
    print("Accuracy: {:.2f}%".format(avg_iou(boxes, out) * 100))
    print((out * 416).astype(np.int))
    # print("Boxes:\n {}-{}".format(out[:, 0] * 416, out[:, 1] * 416))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))

if __name__ == '__main__':
    with open('D:/coursera/YoLoSerirs/data/fddb_face_train.txt', 'r') as f:
        data = f.readlines()

    boxes = []
    for line in data:
        cells = line.split(' ')
        img_path = cells[0]
        img_data = cv2.imread(img_path)
        height, width, channel = img_data.shape

        for cell in cells[1:]:
            xmin, ymin, xmax, ymax, label = cell.split(',')
            xmin, ymin, xmax, ymax, label = float(xmin), float(ymin), float(xmax), float(ymax), int(label)
            assert xmax > xmin and ymax > ymin
            boxes.append([(xmax - xmin) / width, (ymax - ymin) / height])

    boxes = np.array(boxes)

    out = kmeans(boxes, k=9)
    print("Accuracy: {:.2f}%".format(avg_iou(boxes, out) * 100))
    print((out * 416).astype(np.int))
    # print("Boxes:\n {}-{}".format(out[:, 0] * 416, out[:, 1] * 416))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))


