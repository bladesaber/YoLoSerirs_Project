import cv2
import tensorflow as tf
import numpy as np
from Config.tinyConfig import cfg
from Utils import utils
from model.yolov3 import YOLOv3_tiny
from model import ops
from Utils import visualize
import matplotlib.pylab as plt
import time


def load_model(model_name, weight_path, input_size):
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

    if weight_path:
        weight = np.load(weight_path, allow_pickle=True)
        model.set_weights(weight)
    else:
        raise ValueError
    print('Restoring weights from: %s ' % weight_path)

    return model

def load_model_lite(weight_path):
    interpreter = tf.lite.Interpreter(model_path=weight_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # print(input_details)
    # print(output_details)

    return interpreter, input_details, output_details

def video_fps(model_name, weight_path, input_size, framework):
    assert model_name in ['yolov3_tiny']

    STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, True)

    XYSCALE = [1.0, 1.0, 1.0]

    classes = utils.read_class_names(cfg.YOLO.CLASSES)

    if framework=='tf':
        model = load_model(model_name, weight_path, input_size)
    else:
        model, input_details, output_details = load_model_lite(weight_path)

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

            if framework=='tf':
                pred_bbox = model.predict(image_data)
            else:
                model.set_tensor(input_details[0]['index'], image_data)
                model.invoke()
                pred_bbox = [model.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

            pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.6)
            bboxes = utils.nms(bboxes, 0.2, method='nms')
            image = visualize.draw_bbox(original_image, bboxes, classes=classes)

            image = image[:, :, [2, 1, 0]]
            cv2.imshow('cap video', image)
            # plt.imshow(image)
            # plt.show()

            if cv2.waitKey(40) & 0xFF == ord('q'):
                break

            count += 1
            print("FPS of the video is {:5.2f}".format((time.time() - start) / count))


if __name__ == '__main__':
    # video_fps('yolov3_tiny', weight_path='D:\\coursera\\YoLoSerirs\\checkpoint\\yolov3_tiny.tflite',
    #           input_size=416, framework='tflite')

    video_fps('yolov3_tiny', weight_path='D:\\coursera\\YoLoSerirs\\checkpoint\\yolov3_tiny_352.tflite',
              input_size=352, framework='tflite')
