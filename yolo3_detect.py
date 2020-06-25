import tensorflow as tf
from model.yolov3 import YOLOv3
from data.General_DataSet import General_Dataset
from Config.fddbConfig import cfg
import numpy as np
from Utils import utils
from model import ops
import cv2
from Utils import visualize
from PIL import Image

def detect(image_path, weight_path, input_size):
    STRIDES = np.array(cfg.YOLO.STRIDES)
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, tiny=False)

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))

    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv3(input_layer, NUM_CLASS)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = ops.decode(fm, NUM_CLASS)
        bbox_tensors.append(bbox_tensor)
    model = tf.keras.Model(input_layer, bbox_tensors)

    if weight_path:
        weight = np.load(weight_path, allow_pickle=True)
        model.set_weights(weight)
        print('Restoring weights from: %s ' % weight_path)

    pred_bbox = model.predict(image_data)
    pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
    bboxes = utils.nms(bboxes, 0.2, method='nms')

    image = visualize.draw_bbox(original_image, bboxes, classes=utils.read_class_names(cfg.YOLO.CLASSES))
    image = Image.fromarray(image)
    image.show()

if __name__ == '__main__':
    detect(image_path='D:\\coursera\\YoLoSerirs_Project\\dataset\\val\\6.jpg',
           weight_path='D:\\coursera\\YoLoSerirs_Project\\checkpoint\\yolo3_fddb.npy', input_size=320)
