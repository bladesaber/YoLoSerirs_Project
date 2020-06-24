import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def draw_box(img, boxes, BGR2RGB=True):
    plt.figure(8)

    if BGR2RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    currentAxis=plt.gca()
    for x1, y1, x2, y2 in boxes:
        rect=patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r',facecolor='none')
        currentAxis.add_patch(rect)
    plt.show()

# class_list = [
#     "aeroplane", "bicycle", "bird", "boat", "bottle",
#     "bus", "car", "cat", "chair", "cow",
#     "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# ]

class_list = ["car", "person"]

xml_dir = 'D:/DataSet/VOCtrainval_06-Nov-2007/VOC2007/Annotations/'
img_dir = 'D:/DataSet/VOCtrainval_06-Nov-2007/VOC2007/JPEGImages/'

def read_data_rpn(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall("object")
    gtbboxes = []

    for idx, obj in enumerate(objects):
        if obj.find("name").text in class_list:
            xmin = int(obj.find("bndbox").find("xmin").text)
            xmax = int(obj.find("bndbox").find("xmax").text)
            ymin = int(obj.find("bndbox").find("ymin").text)
            ymax = int(obj.find("bndbox").find("ymax").text)

            txt = ','.join([str(xmin), str(ymin), str(xmax), str(ymax), str(class_list.index(obj.find("name").text))])
            gtbboxes.append(txt)
    return gtbboxes

def get_xml_image(xml_path):
    return os.path.join(xml_dir, xml_path), os.path.join(img_dir, xml_path[:-4] + ".jpg")

def extract_special_file():
    paths = os.listdir(xml_dir)
    train_num = int(len(paths)*0.8)
    train_paths = paths[:train_num]
    test_paths = paths[train_num:]

    with open('D:\coursera\YoLoSerirs\data/voc2007_train.txt', 'w') as f:
        for path in train_paths:
            xml_path, img_path = get_xml_image(path)
            gtbboxes = read_data_rpn(xml_path)

            if len(gtbboxes)>0:
                txt = ' '.join([img_path]+gtbboxes)
                f.write(txt+'\n')

    with open('D:\coursera\YoLoSerirs\data/voc2007_test.txt', 'w') as f:
        for path in test_paths:
            xml_path, img_path = get_xml_image(path)
            gtbboxes = read_data_rpn(xml_path)

            if len(gtbboxes)>0:
                txt = ' '.join([img_path]+gtbboxes)
                f.write(txt+'\n')

if __name__ == '__main__':
    extract_special_file()

    # with open('D:\coursera\YoLoSerirs\data/voc2007.txt', 'r') as f:
    #     data = f.readlines()
    #
    # for line in data:
    #     cells = line.split(' ')
    #     image_path = cells[0]
    #     boxes = []
    #     for cell in cells[1:]:
    #         xmin, ymin, xmax, ymax, label = cell.split(',')
    #         xmin, ymin, xmax, ymax, label = int(xmin), int(ymin), int(xmax), int(ymax), int(label)
    #         boxes.append([xmin, ymin, xmax, ymax])
    #     img = cv2.imread(image_path)
    #     draw_box(img, boxes, True)