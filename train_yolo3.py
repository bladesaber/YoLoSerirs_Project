import tensorflow as tf
from model.yolov4 import YOLOv4
from model.yolov3 import YOLOv3, YOLOv3_tiny
from data.General_DataSet import General_Dataset
from Config.fddbConfig import cfg
import numpy as np
from Utils import utils
from model import ops
import os
from tqdm import tqdm

def train(model_name, weight_path, save_path, stage):
    assert model_name in ['yolov3', 'yolov4']

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    STRIDES = np.array(cfg.YOLO.STRIDES)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    XYSCALE = cfg.YOLO.XYSCALE
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)

    trainset = General_Dataset('train', cfg=cfg)

    isfreeze = False
    steps_per_epoch = len(trainset)
    epochs = cfg.TRAIN.EPOCHS

    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    total_steps = epochs * steps_per_epoch
    print('steps_per_epoch:', steps_per_epoch)

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    feature_maps = YOLOv3(input_layer, NUM_CLASS)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = ops.decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i)
        bbox_tensors.append(fm)
        bbox_tensors.append(bbox_tensor)
    model = tf.keras.Model(input_layer, bbox_tensors)


    if weight_path:
        final_layers = utils.load_weights_v3_npy(model, weight_path, exclude=True)
        print('Restoring weights from: %s ... ' % weight_path)
    else:
        final_layers = []

    optimizer = tf.keras.optimizers.Adam()

    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(3):
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

            # tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
            #          "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
            #                                                    giou_loss, conf_loss,
            #                                                    prob_loss, total_loss))

            global_steps.assign_add(1)

    if stage=='last':
        for layer in model.layers:
            if layer.name not in final_layers:
                layer.trainable = False
            else:
                print(layer.name)

    for epoch in range(epochs):
        for image_data, target in tqdm(trainset):
            train_step(image_data, target)

        if save_path:
            np.save(save_path, model.get_weights())


if __name__ == '__main__':
    train(model_name='yolov3', weight_path='D:/coursera/YoLoSerirs/pretrain/yolov3.npy',
          save_path='D:\coursera\YoLoSerirs\checkpoint\yolo3_fddb.npy', stage='last')
