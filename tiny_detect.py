import tensorflow as tf
from model.yolov3 import YOLOv3, YOLOv3_tiny
from Config.tinyConfig import cfg
import numpy as np
from Utils import utils
from model import ops
import os
import cv2
from Utils import visualize
from PIL import Image
import time

def detect(model_name, weight_path, input_size, image_path):
    assert model_name in ['yolov3_tiny']

    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, tiny=True)

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    if model_name == 'yolov3_tiny':
        feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = ops.decode(fm, NUM_CLASS)
            bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)
    else:
        raise ValueError

    if weight_path:
        if model_name == 'yolov3_tiny':
            weight = np.load(weight_path, allow_pickle=True)
            model.set_weights(weight)
        else:
            raise ValueError
        print('Restoring weights from: %s ' % weight_path)

    # model.summary()

    start_time = time.time()
    pred_bbox = model.predict(image_data)
    print(time.time() - start_time)

    pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.5)
    bboxes = utils.nms(bboxes, 0.3, method='nms')

    image = visualize.draw_bbox(original_image, bboxes, classes=utils.read_class_names(cfg.YOLO.CLASSES))
    image = Image.fromarray(image)
    image.show()

def detect_tflite(model_name, weight_path, input_size, image_path):
    assert model_name in ['yolov3_tiny']

    STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, True)

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=weight_path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(input_details)
    print(output_details)
    interpreter.set_tensor(input_details[0]['index'], image_data)
    start_time = time.time()
    interpreter.invoke()
    pred_bbox = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    print(time.time() - start_time)

    pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.5)
    bboxes = utils.nms(bboxes, 0.3, method='nms')

    image = visualize.draw_bbox(original_image, bboxes, classes=utils.read_class_names(cfg.YOLO.CLASSES))
    image = Image.fromarray(image)
    image.show()

if __name__ == '__main__':
    # detect('yolov3_tiny', weight_path='D:\coursera\YoLoSerirs\checkpoint\yolo3_tiny_custom_anchors.npy',
    #        input_size=416, image_path='D:\\coursera\\YoLoSerirs\\dataset\\val\\1.jpg')

    # detect('yolov3_tiny', weight_path='D:\coursera\YoLoSerirs\checkpoint\yolo3_tiny_original_anchors.npy',
    #        input_size=416, image_path='D:\\coursera\\YoLoSerirs\\dataset\\val\\1.jpg')

    detect('yolov3_tiny', weight_path='D:\coursera\YoLoSerirs\checkpoint\yolo3_tiny_prun.npy',
           input_size=416, image_path='D:\\coursera\\YoLoSerirs\\dataset\\val\\1.jpg')

    # detect_tflite('yolov3_tiny', weight_path='D:\\coursera\\YoLoSerirs\\checkpoint\\yolov3_tiny.tflite',
    #        input_size=416, image_path='D:\\coursera\\YoLoSerirs\\dataset\\val\\2.jpg')
