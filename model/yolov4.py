import numpy as np
import tensorflow as tf
import model.ops as ops
import model.backbone as backbone

def YOLOv4(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.cspdarknet53(input_layer)

    route = conv
    conv = ops.convolutional(conv, (1, 1, 512, 256))
    conv = ops.upsample(conv)
    route_2 = ops.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = ops.convolutional(conv, (1, 1, 512, 256))
    conv = ops.convolutional(conv, (3, 3, 256, 512))
    conv = ops.convolutional(conv, (1, 1, 512, 256))
    conv = ops.convolutional(conv, (3, 3, 256, 512))
    conv = ops.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = ops.convolutional(conv, (1, 1, 256, 128))
    conv = ops.upsample(conv)
    route_1 = ops.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = ops.convolutional(conv, (1, 1, 256, 128))
    conv = ops.convolutional(conv, (3, 3, 128, 256))
    conv = ops.convolutional(conv, (1, 1, 256, 128))
    conv = ops.convolutional(conv, (3, 3, 128, 256))
    conv = ops.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = ops.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = ops.convolutional(conv, (1, 1, 256, 3*(NUM_CLASS+5)), activate=False, bn=False)

    # -----------------------------------------
    conv = ops.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = ops.convolutional(conv, (1, 1, 512, 256))
    conv = ops.convolutional(conv, (3, 3, 256, 512))
    conv = ops.convolutional(conv, (1, 1, 512, 256))
    conv = ops.convolutional(conv, (3, 3, 256, 512))
    conv = ops.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = ops.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = ops.convolutional(conv, (1, 1, 512, 3*(NUM_CLASS+5)), activate=False, bn=False)

    # -----------------------------------
    conv = ops.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = ops.convolutional(conv, (1, 1, 1024, 512))
    conv = ops.convolutional(conv, (3, 3, 512, 1024))
    conv = ops.convolutional(conv, (1, 1, 1024, 512))
    conv = ops.convolutional(conv, (3, 3, 512, 1024))
    conv = ops.convolutional(conv, (1, 1, 1024, 512))

    conv = ops.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = ops.convolutional(conv, (1, 1, 1024, 3*(NUM_CLASS+5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]




