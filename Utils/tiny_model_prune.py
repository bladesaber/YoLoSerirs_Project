import tensorflow_model_optimization as tfmot
import os
import cv2
import tensorflow as tf
import numpy as np
from model.yolov3 import YOLOv3_tiny
from Config.tinyConfig import cfg
from model import ops
from Utils import utils
from data.Tiny_Dataset import TinyDataset
from tqdm import tqdm

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

def show_prun(model):
    for i, w in enumerate(model.get_weights()):
        print("{} -- Total:{}, Zeros: {:.2f}%".format(model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100))

def prune_train(model_name, weight_path, logdir, save_path):
    assert model_name in ['yolov3_tiny']

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, True)

    trainset = TinyDataset('train')

    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    total_steps = cfg.TRAIN.PRUN_EPOCHS * steps_per_epoch

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    if model_name == 'yolov3_tiny':
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
        if model_name == 'yolov3_tiny':
            weight = np.load(weight_path, allow_pickle=True)
            model.set_weights(weight)
        else:
            raise ValueError
    print('Restoring weights from: %s ... ' % weight_path)

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                                 final_sparsity=0.50,
                                                                 begin_step=0,
                                                                 end_step=total_steps)
    }

    def apply_pruning_to_dense_conv(layer):
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            print('find it')
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

    # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense to the layers of the model.
    model_for_pruning = tf.keras.models.clone_model(model, clone_function=apply_pruning_to_dense_conv, )
    # model_for_pruning.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    unused_arg = -1
    model_for_pruning.optimizer = optimizer

    step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    step_callback.set_model(model_for_pruning)
    log_callback = tfmot.sparsity.keras.PruningSummaries(
        log_dir=logdir)  # Log sparsity and other metrics in Tensorboard.
    log_callback.set_model(model_for_pruning)

    step_callback.on_train_begin()  # run pruning callback
    for epoch in range(cfg.TRAIN.PRUN_EPOCHS):
        log_callback.on_epoch_begin(epoch=unused_arg)  # run pruning callback

        for image_data, target in trainset:
            step_callback.on_train_batch_begin(batch=unused_arg)  # run pruning callback

            with tf.GradientTape() as tape:
                pred_result = model_for_pruning(image_data, training=True)
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
                gradients = tape.gradient(total_loss, model_for_pruning.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model_for_pruning.trainable_variables))

                tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                         "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                                   giou_loss, conf_loss,
                                                                   prob_loss, total_loss))
                global_steps.assign_add(1)

        step_callback.on_epoch_end(batch=unused_arg)  # run pruning callback

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    np.save('D:\coursera\YoLoSerirs\checkpoint\\yolo3_tiny_prun.npy', model_for_export.get_weights())

    show_prun(model_for_export)

    # tf.keras.models.save_model(model_for_export, save_path, include_optimizer=False)

if __name__ == '__main__':
    prune_train('yolov3_tiny', weight_path='D:\coursera\YoLoSerirs\checkpoint\yolo3_tiny_original_anchors.npy',
                logdir='D:\\coursera\\YoLoSerirs\\checkpoint\\log',
                save_path='D:\coursera\YoLoSerirs\checkpoint\pruned.h5')
