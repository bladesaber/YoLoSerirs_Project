import tensorflow as tf
from model.yolov4 import YOLOv4
from model.yolov3 import YOLOv3, YOLOv3_tiny
from Config.config import cfg
import numpy as np
from Utils import utils
from model import ops
import os
import cv2
from Utils import visualize
from PIL import Image
import time

def detect(model_name, weight_path, input_size, image_path, framework):
    assert model_name in ['yolov3_tiny', 'yolov3', 'yolov4']

    if model_name == 'yolov3_tiny':
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, True)
    elif model_name == 'yolov3':
        STRIDES = np.array(cfg.YOLO.STRIDES)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, False)
    elif model_name == 'yolov4':
        STRIDES = np.array(cfg.YOLO.STRIDES)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
    else:
        raise ValueError

    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    XYSCALE = cfg.YOLO.XYSCALE

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if framework == 'tf':
        input_layer = tf.keras.layers.Input([input_size, input_size, 3])
        if model_name == 'yolov3_tiny':
            feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = ops.decode(fm, NUM_CLASS)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
        elif model_name == 'yolov3':
            feature_maps = YOLOv3(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = ops.decode(fm, NUM_CLASS)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
        elif model_name == 'yolov4':
            feature_maps = YOLOv4(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = ops.decode(fm, NUM_CLASS)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
        else:
            model = None
            raise ValueError

        if weight_path.split(".")[-1] == "weights":
            if model_name == 'yolov3_tiny':
                utils.load_weights_tiny(model, weight_path)
                # utils.extract_weights_tiny(model, weight_path)
                print('load yolo tiny 3')

            elif model_name=='yolov3':
                utils.load_weights_v3(model, weight_path)
                print('load yolo 3')

            elif model_name=='yolov4':
                utils.load_weights(model, weight_path)
                print('load yolo 4')
            else:
                raise ValueError

        elif weight_path.split(".")[-1] == "npy":
            if model_name == 'yolov3_tiny':
                # utils.load_weights_tiny_npy(model, weight_path)
                print('load yolo tiny 3 npy')
        else:
            model.load_weights(weight_path)
        print('Restoring weights from: %s ' % weight_path)

        # weight = np.load('D:\\coursera\\YoLoSerirs\\checkpoint\\yolo3_tiny.npy', allow_pickle=True)
        # model.set_weights(weight)

        # model.summary()

        start_time = time.time()
        pred_bbox = model.predict(image_data)
        print(time.time()-start_time)

    else:
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
        print(time.time()-start_time)

    if model_name == 'yolov4':
        pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
    else:
        pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.5)
    bboxes = utils.nms(bboxes, 0.3, method='nms')

    image = visualize.draw_bbox(original_image, bboxes)
    image = Image.fromarray(image)
    image.show()

if __name__ == '__main__':
    # detect('yolov4', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov4.weights',
    #        input_size=608, image_path='D:\\coursera\\YoLoSerirs\\dataset\\val\\1.jpg',
    #        framework='tf')

    # detect('yolov4', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov4.weights',
    #        input_size=416, image_path='D:\\coursera\\YoLoSerirs\\dataset\\val\\1.jpg',
    #        framework='tf')

    # detect('yolov3_tiny', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov3-tiny.weights',
    #        input_size=416, image_path='D:\\coursera\\YoLoSerirs\\dataset\\val\\1.jpg',
    #        framework='tf')

    # detect('yolov3', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov3.weights',
    #        input_size=608, image_path='D:\\coursera\\YoLoSerirs\\dataset\\val\\1.jpg',
    #        framework='tf')

    # detect('yolov4', weight_path='D:\\coursera\\YoLoSerirs\\checkpoint\\yolov4.tflite',
    #        input_size=416, image_path='D:\\coursera\\YoLoSerirs\\dataset\\val\\1.jpg',
    #        framework='tflite')

    # detect('yolov3_tiny', weight_path='D:\\coursera\\YoLoSerirs\\checkpoint\\yolov3_tiny.tflite',
    #        input_size=416, image_path='D:\\coursera\\YoLoSerirs\\dataset\\val\\2.jpg',
    #        framework='tflite')

    # detect('yolov3_tiny', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov3_tiny.npy',
    #        input_size=416, image_path='D:\\coursera\\YoLoSerirs\\dataset\\val\\1.jpg',
    #        framework='tf')

    pass

