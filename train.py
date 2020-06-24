import tensorflow as tf
from model.yolov4 import YOLOv4
from model.yolov3 import YOLOv3, YOLOv3_tiny
from data.Dataset import Dataset
from Config.config import cfg
import numpy as np
from Utils import utils
from model import ops
import os
import shutil

def train(model_name, weight_path, save_path, logdir=None):
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
    testset = Dataset('test')

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

    # for name in ['conv2d_93', 'conv2d_101', 'conv2d_109']:
    #     layer = model.get_layer(name)
    #     print(layer.name, layer.output_shape)

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

    optimizer = tf.keras.optimizers.Adam()

    if logdir:
        if os.path.exists(logdir):
            shutil.rmtree(logdir)
        writer = tf.summary.create_file_writer(logdir)
    else:
        writer = None

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

            tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))

            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            else:
                lr = cfg.TRAIN.LR_END + \
                     0.5*(cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * \
                     ((1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
            optimizer.lr.assign(lr.numpy())

            # if writer:
            #     # writing summary data
            #     with writer.as_default():
            #         tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            #         tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            #         tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            #         tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            #         tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            #     writer.flush()

    def test_step(image_data, target):
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

        tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
                                                               prob_loss, total_loss))

    for epoch in range(first_stage_epochs + second_stage_epochs):
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in ['conv2d_93', 'conv2d_101', 'conv2d_109']:
                    freeze = model.get_layer(name)
                    ops.freeze_all(freeze)

        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze = False
                for name in ['conv2d_93', 'conv2d_101', 'conv2d_109']:
                    freeze = model.get_layer(name)
                    ops.unfreeze_all(freeze)

        for image_data, target in trainset:
            train_step(image_data, target)

        for image_data, target in testset:
            test_step(image_data, target)

        if save_path:
            model.save_weights(save_path)


if __name__ == '__main__':
    train(model_name='yolov4', weight_path=None,
          save_path='D:\coursera\YoLoSerirs\checkpoint\yolo3-tiny')
