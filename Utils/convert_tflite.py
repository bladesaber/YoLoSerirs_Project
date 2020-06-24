import tensorflow as tf
import numpy as np
import cv2
from model.yolov4 import YOLOv4
from model.yolov3 import YOLOv3_tiny, YOLOv3
from model.ops import decode
from Utils import utils
import os
from Config.config import cfg
from model import ops
from absl import logging
from tensorflow.python.framework import graph_util

def representative_data_gen(input_size):
    fimage = open('D:\\coursera\\YoLoSerirs\\data\\val2017.txt').read().split()
    for input_value in range(100):
        if os.path.exists(fimage[input_value]):
            original_image = cv2.imread(fimage[input_value])
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
            img_in = image_data[np.newaxis, ...].astype(np.float32)
            yield [img_in]
        else:
            continue

def save_pb(model_name, weight_path, output, input_size):
    assert model_name in ['yolov3_tiny', 'yolov3', 'yolov4']

    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
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
        raise ValueError

    if weight_path.split(".")[-1] == "weights":
        if model_name == 'yolov3_tiny':
            utils.load_weights_tiny(model, weight_path)
        elif model_name == ' yolov3':
            utils.load_weights_v3(model, weight_path)
        elif model_name == 'yolov4':
            utils.load_weights(model, weight_path)
        else:
            raise ValueError
    else:
        model.load_weights(weight_path).expect_partial()
    print('Restoring weights from: %s ... ' % weight_path)

    tf.saved_model.save(model, output)

def transfer_tflite(model_name, weight_path, output, input_size):
    assert model_name in ['yolov3_tiny', 'yolov3', 'yolov4']

    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
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
        elif model_name == ' yolov3':
            utils.load_weights_v3(model, weight_path)
        elif model_name == 'yolov4':
            utils.load_weights(model, weight_path)
        else:
            raise ValueError
    else:
        model.load_weights(weight_path).expect_partial()
    print('Restoring weights from: %s ... ' % weight_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()
    open(output, 'wb').write(tflite_model)

def save_tflite(model_name, weight_path, quantize_mode, output, input_size):
    assert model_name in ['yolov3_tiny', 'yolov3', 'yolov4']
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
        elif model_name == ' yolov3':
            utils.load_weights_v3(model, weight_path)
        elif model_name == 'yolov4':
            utils.load_weights(model, weight_path)
        else:
            raise ValueError
    else:
        model.load_weights(weight_path).expect_partial()
    print('Restoring weights from: %s ... ' % weight_path)

    # model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if tf.__version__ >= '2.2.0':
        converter.experimental_new_converter = False

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


def demo(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    logging.info('tflite model loaded')

    input_details = interpreter.get_input_details()
    print(input_details)
    output_details = interpreter.get_output_details()
    print(output_details)

    input_shape = input_details[0]['shape']

    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    print(output_data)

def save_tiny_npz(model_name, weight_path, output, input_size):
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
        raise ValueError

    if weight_path.split(".")[-1] == "weights":
        if model_name == 'yolov3_tiny':
            extract_weight = utils.extract_weights_tiny(model, weight_path)
        else:
            raise ValueError
    else:
        raise ValueError
    print('Restoring weights from: %s ... ' % weight_path)

    np.save(output, extract_weight)

if __name__ == '__main__':
    # save_pb('yolov4', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov4.weights',
    #             output='D:\\coursera\\YoLoSerirs\\checkpoint', input_size=416)

    # demo("D:\coursera\YoLoSerirs\checkpoint\yolov4.tflite")

    # save_tflite('full_int8', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov4.weights',
    #             quantize_mode='int8', output='D:\\coursera\\YoLoSerirs\\checkpoint\\yolov4.tflite',
    #             input_size=416)

    # transfer_tflite('D:\\coursera\\YoLoSerirs\\checkpoint',
    #                 output='D:\\coursera\\YoLoSerirs\\checkpoint\\yolov4.tflite')

    # --------------------------------------------------
    # save_pb('yolov3_tiny', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov3-tiny.weights',
    #             output='D:\\coursera\\YoLoSerirs\\checkpoint', input_size=416)

    # transfer_tflite('D:\\coursera\\YoLoSerirs\\checkpoint', output='D:\\coursera\\YoLoSerirs\\checkpoint\\yolov3_tiny.tflite')

    # save_tiny_npz('yolov3_tiny', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov3-tiny.weights',
    #             output='D:\\coursera\\YoLoSerirs\\pretrain\\yolov3_tiny.npy', input_size=416)

    # save_tflite('full_int8', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov4.weights',
    #             quantize_mode='int8', output='D:\\coursera\\YoLoSerirs\\checkpoint\\yolov4.tflite',
    #             input_size=416)

    transfer_tflite('yolov3_tiny', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov3-tiny.weights',
                output='D:\\coursera\\YoLoSerirs\\checkpoint\\yolov3_tiny_multi.tflite', input_size=416)
