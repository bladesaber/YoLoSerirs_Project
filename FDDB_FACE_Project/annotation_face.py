import os
import shutil
import cv2

annotion_dir = 'D:/DataSet/rebuild/FDDB_folds'
img_root = 'D:/DataSet/rebuild/'

output_file = 'D:\\coursera\\YoLoSerirs_Project\\data\\fddb_face_train.txt'

annotion_list = os.listdir(annotion_dir)
annotion_files = []
for annotion in annotion_list:
    if 'ellipseList' in annotion:
        annotion_files.append(os.path.join(annotion_dir, annotion))

with open(output_file, 'w') as f:
    for annotion_file in annotion_files:
        with open(annotion_file, 'r') as r:
            data = r.readlines()

        state = 0
        img_path = ''
        bbox = []
        width, height = 1, 1

        for line in data:
            line = line.strip()
            if '/' in line:

                if len(bbox)>0:
                    if os.path.exists(img_path):
                        f.write(img_path + ' ' + ' '.join(bbox)+'\n')
                    bbox = []
                    img_path = ''

                img_path = img_root+line+'.jpg'

                if os.path.exists(img_path):
                    height_v, width_v, channel = (cv2.imread(img_path)).shape
                    width = width_v
                    height = height_v

                state = 1

            elif state==1:
                num = int(line)
                state = 2

            elif state == 2:
                ry, rw, _, xcenter, ycenter, _, _ = line.split(' ')
                ry, rw, xcenter, ycenter = int(float(ry)*1.0), int(float(rw)*1.0), int(float(xcenter)), int(float(ycenter))

                xmin = max(xcenter-rw, 0)
                ymin = max(ycenter-ry, 0)
                xmax = min(xcenter+rw, width)
                ymax = min(ycenter+ry, height)

                bbox.append(','.join([str(xmin), str(ymin), str(xmax), str(ymax), '0']))

        if os.path.exists(img_path):
            f.write(img_path + ' ' + ' '.join(bbox) + '\n')
            bbox = []
            img_path = ''
