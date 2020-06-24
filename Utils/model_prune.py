import tensorflow_model_optimization as tfmot
import os
import cv2
import tensorflow as tf
import numpy as np
from model.yolov4 import YOLOv4
from model.yolov3 import YOLOv3, YOLOv3_tiny
from Config.config import cfg
from model import ops
from Utils import utils
from data.Dataset import Dataset
import shutil
from tqdm import tqdm

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

def prune_train(model_name, weight_path, logdir, save_path, epoches):
    assert model_name in ['yolov3_tiny', 'yolov3', 'yolov4']

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    STRIDES = np.array(cfg.YOLO.STRIDES)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    XYSCALE = cfg.YOLO.XYSCALE
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)

    trainset = Dataset('train')

    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS

    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    if model_name=='yolov3_tiny':
        feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = ops.decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)
    elif model_name=='yolov3':
        feature_maps = YOLOv3(input_layer, NUM_CLASS)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = ops.decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)
    elif model_name=='yolov4':
        feature_maps = YOLOv4(input_layer, NUM_CLASS)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = ops.decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)
    else:
        raise ValueError

    if weight_path:
        if weight_path.split(".")[-1] == "weights":
            if model_name == 'yolov3_tiny':
                utils.load_weights_tiny(model, weight_path)
            elif model_name=='yolov3':
                utils.load_weights_v3(model, weight_path)
            elif model_name=='yolov4':
                utils.load_weights(model, weight_path)
            else:
                raise ValueError
        else:
            model.load_weights(weight_path)
        print('Restoring weights from: %s ... ' % weight_path)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    if os.path.exists(logdir):
        shutil.rmtree(logdir)

    # for layer in model.layers:
    #     print(layer.name, isinstance(layer, tf.keras.layers.Conv2D))

    def apply_pruning_to_dense(layer):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return tfmot.sparsity.keras.prune_low_magnitude(layer)
        return layer

    # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense`
    # to the layers of the model.
    model_for_pruning = tf.keras.models.clone_model(model, clone_function=apply_pruning_to_dense,)
    # model_for_pruning.summary()

    unused_arg = -1
    model_for_pruning.optimizer = optimizer

    step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    step_callback.set_model(model_for_pruning)

    log_callback = tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)  # Log sparsity and other metrics in Tensorboard.
    log_callback.set_model(model_for_pruning)

    step_callback.on_train_begin()  # run pruning callback
    for epoch in range(epoches):
        log_callback.on_epoch_begin(epoch=unused_arg)  # run pruning callback

        for image_data, target in trainset:
            step_callback.on_train_batch_begin(batch=unused_arg)  # run pruning callback
            with tf.GradientTape() as tape:
                pred_result = model_for_pruning(image_data, training=True)
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
                gradients = tape.gradient(total_loss, model_for_pruning.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model_for_pruning.trainable_variables))

                tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                         "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                                   giou_loss, conf_loss,
                                                                   prob_loss, total_loss))

        step_callback.on_epoch_end(batch=unused_arg)  # run pruning callback

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    return model_for_export
    # tf.keras.models.save_model(model_for_export, save_path, include_optimizer=False)

if __name__ == '__main__':
    prune_train('yolov4', weight_path='D:\\coursera\\YoLoSerirs\\pretrain\\yolov4.weights',
          logdir='D:\\coursera\\YoLoSerirs\\checkpoint', save_path='D:\coursera\YoLoSerirs\checkpoint\pruned.h5',
                epoches=3)
