import tensorflow as tf
import numpy as np
import cv2
from model.yolov3 import YOLOv3_tiny
from model.ops import decode
from Utils import utils
import os
from Config.tinyConfig import cfg
from model import ops
from absl import logging
from tensorflow.python.framework import graph_util

def representative_data_gen():
    fimage = open('D:\\coursera\\YoLoSerirs\\data\\voc2007_train.txt').read().split()
    for input_value in range(500):
        if os.path.exists(fimage[input_value]):
            original_image = cv2.imread(fimage[input_value])
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            image_data = utils.image_preprocess(np.copy(original_image), [cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE])
            img_in = image_data[np.newaxis, ...].astype(np.float32)
            yield [img_in]
        else:
            continue

def transfer_tflite(model_name, weight_path, output, input_size):
    assert model_name in ['yolov3_tiny']

    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])

    if model_name == 'yolov3_tiny':
        feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = ops.decode(fm, NUM_CLASS)
            bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)
    else:
        model = None
        raise ValueError

    if model_name == 'yolov3_tiny':
        weight = np.load(weight_path, allow_pickle=True)
        model.set_weights(weight)
    else:
        raise ValueError
    print('Restoring weights from: %s ' % weight_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(output, 'wb').write(tflite_model)

def save_tflite(model_name, weight_path, quantize_mode, output, input_size):
    assert model_name in ['yolov3_tiny']
    assert quantize_mode in ['int8', 'float16', 'full_int8']

    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])

    if model_name == 'yolov3_tiny':
        feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = ops.decode(fm, NUM_CLASS)
            bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)
    else:
        model = None
        raise ValueError

    if weight_path:
        if model_name == 'yolov3_tiny':
            weight = np.load(weight_path, allow_pickle=True)
            model.set_weights(weight)
        else:
            raise ValueError
        print('Restoring weights from: %s ' % weight_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # if tf.__version__ >= '2.2.0':
    #     converter.experimental_new_converter = False

    if quantize_mode == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    elif quantize_mode == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]

    elif quantize_mode == 'full_int8':
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.allow_custom_ops = True
        converter.representative_dataset = representative_data_gen
    else:
        raise ValueError

    tflite_model = converter.convert()
    open(output, 'wb').write(tflite_model)

    logging.info("model saved to: {}".format(output))

if __name__ == '__main__':
    # save_tflite(model_name='yolov3_tiny',
    #             weight_path='D:\coursera\YoLoSerirs\checkpoint\yolo3_tiny_original_anchors.npy',
    #             quantize_mode='full_int8', output='D:\coursera\YoLoSerirs\checkpoint\\yolov3_tiny.tflite',
    #             input_size=cfg.TRAIN.INPUT_SIZE)

    transfer_tflite(model_name='yolov3_tiny',
                weight_path='D:\coursera\YoLoSerirs\checkpoint\yolo3_tiny_original_anchors.npy',
                    output='D:\coursera\YoLoSerirs\checkpoint\\yolov3_tiny_352.tflite', input_size=352)