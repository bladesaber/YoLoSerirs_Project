import tensorflow as tf
from model.yolov4 import YOLOv4
from model.yolov3 import YOLOv3, YOLOv3_tiny
from data.Tiny_Dataset import TinyDataset
from Config.tinyConfig import cfg
import numpy as np
from Utils import utils
from model import ops
import os
import shutil
from tqdm import tqdm
import re

def train(model_name, weight_path, stage, save_path, use_self_npy, logdir=None):
    assert model_name in ['yolov3_tiny']

    num_stage = 3

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    XYSCALE = cfg.YOLO.XYSCALE
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, tiny=True)

    trainset = TinyDataset('train')
    testset = TinyDataset('test')

    steps_per_epoch = len(trainset)

    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch
    epoches = cfg.TRAIN.EPOCHS

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    if model_name=='yolov3_tiny':
        feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = ops.decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)
    else:
        raise ValueError

    if weight_path:
        if weight_path.split(".")[-1] == "npy":
            if stage=='last':
                if use_self_npy:
                    weight = np.load(weight_path, allow_pickle=True)
                    model.set_weights(weight)
                else:
                    utils.load_weights_tiny_npy(model, weight_path, True)
            else:
                if use_self_npy:
                    weight = np.load(weight_path, allow_pickle=True)
                    model.set_weights(weight)
                else:
                    utils.load_weights_tiny_npy(model, weight_path, False)
        else:
            model.load_weights(weight_path, by_name=True)
        print('Restoring weights from: %s ... ' % weight_path)

    middle_layers, final_layers = utils.weights_tiny_name()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # if logdir:
    #     if os.path.exists(logdir):
    #         shutil.rmtree(logdir)
    #     writer = tf.summary.create_file_writer(logdir)
    # else:
    #     writer = None

    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(2):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = ops.compute_loss(pred, conv, target[i][0], target[i][1],
                                              STRIDES=STRIDES, NUM_CLASS=NUM_CLASS,
                                              IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))

            # update learning rate
            global_steps.assign_add(1)
            if global_steps%int(total_steps/num_stage)==0:
                lr = optimizer.lr.numpy()/10.0
                optimizer.lr.assign(lr)

    def test_step(image_data, target):
        pred_result = model(image_data, training=True)
        giou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(2):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = ops.compute_loss(pred, conv, target[i][0], target[i][1],
                                          STRIDES=STRIDES, NUM_CLASS=NUM_CLASS,
                                          IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
                                                               prob_loss, total_loss))

    if stage=='last':
        for layer in model.layers:
            if layer.name not in final_layers:
                layer.trainable = False

    # for layer in model.layers:
    #     print(layer.name, layer.trainable)

    for epoch in range(epoches):
        for image_data, target in trainset:
            train_step(image_data, target)

        # for image_data, target in testset:
        #     test_step(image_data, target)

        if save_path:
            np.save(save_path,model.get_weights())

if __name__ == '__main__':
    # train(model_name='yolov3_tiny', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov3_tiny.npy',
    #       save_path='D:\\coursera\\YoLoSerirs\\checkpoint\\yolo3_tiny_original_anchors.npy', stage='last', use_self_npy=False)

    train(model_name='yolov3_tiny', weight_path='D:\coursera\YoLoSerirs\checkpoint\yolo3_tiny_original_anchors.npy',
          save_path='D:\\coursera\\YoLoSerirs\\checkpoint\\yolo3_tiny_2.npy', stage='full', use_self_npy=True)
