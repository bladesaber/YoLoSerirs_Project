import cv2
import tensorflow as tf
import numpy as np
from Config.config import cfg
from Utils import utils
from model.yolov4 import YOLOv4
from model.yolov3 import YOLOv3, YOLOv3_tiny
from model import ops
from Utils import visualize
import matplotlib.pylab as plt
import time

def load_model(model_name, weight_path, input_size, framework):
    assert model_name in ['yolov3_tiny', 'yolov3', 'yolov4']

    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))

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
                print('load yolo tiny 3')

            elif model_name == 'yolov3':
                utils.load_weights_v3(model, weight_path)
                print('load yolo 3')

            elif model_name == 'yolov4':
                utils.load_weights(model, weight_path)
                print('load yolo 4')
            else:
                raise ValueError
        else:
            model.load_weights(weight_path).expect_partial()
        print('Restoring weights from: %s ' % weight_path)

        return model

def video_fps(model_name, weight_path, input_size, framework):
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

    if model_name == 'yolov4':
        XYSCALE = cfg.YOLO.XYSCALE
    else:
        XYSCALE = [1.0, 1.0, 1.0]

    model = load_model(model_name, weight_path, input_size, framework)

    video_path = 'D:\\coursera\\YoLoSerirs\\dataset\\test.mp4'

    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    start = time.time()
    count = 0
    success = True
    while success:
        success, image = vcapture.read()

        if success:

            original_image = image
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image_size = original_image.shape[:2]

            image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            pred_bbox = model.predict(image_data)

            pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.5)
            bboxes = utils.nms(bboxes, 0.3, method='nms')
            image = visualize.draw_bbox(original_image, bboxes)

            image = image[:, :, [2, 1, 0]]
            cv2.imshow('cap video', image)
            # plt.imshow(image)
            # plt.show()

            if cv2.waitKey(40) & 0xFF == ord('q'):
                break

            count += 1
            print("FPS of the video is {:5.2f}".format((time.time()-start)/count))

if __name__ == '__main__':
    video_fps('yolov4', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov4.weights',
           input_size=416, framework='tf')

    # video_fps('yolov3', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov3.weights',
    #           input_size=416, framework='tf')

    # video_fps('yolov3_tiny', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov3-tiny.weights',
    #           input_size=416, framework='tf')

