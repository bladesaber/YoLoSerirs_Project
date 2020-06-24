import numpy as np
import tensorflow as tf
import model.ops as ops
import model.backbone as backbone

def YOLOv3(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.darknet53(input_layer)

    conv = ops.convolutional(conv, (1, 1, 1024, 512))
    conv = ops.convolutional(conv, (3, 3, 512, 1024))
    conv = ops.convolutional(conv, (1, 1, 1024, 512))
    conv = ops.convolutional(conv, (3, 3, 512, 1024))
    conv = ops.convolutional(conv, (1, 1, 1024, 512))

    conv_lobj_branch = ops.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = ops.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(NUM_CLASS+5)), activate=False, bn=False)

    # -----------------------------------------------
    conv = ops.convolutional(conv, (1, 1, 512, 256))
    conv = ops.upsample(conv)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = ops.convolutional(conv, (1, 1, 768, 256))
    conv = ops.convolutional(conv, (3, 3, 256, 512))
    conv = ops.convolutional(conv, (1, 1, 512, 256))
    conv = ops.convolutional(conv, (3, 3, 256, 512))
    conv = ops.convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = ops.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = ops.convolutional(conv_mobj_branch, (1, 1, 512, 3*(NUM_CLASS+5)), activate=False, bn=False)

    # ----------------------
    conv = ops.convolutional(conv, (1, 1, 256, 128))
    conv = ops.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv = ops.convolutional(conv, (1, 1, 384, 128))
    conv = ops.convolutional(conv, (3, 3, 128, 256))
    conv = ops.convolutional(conv, (1, 1, 256, 128))
    conv = ops.convolutional(conv, (3, 3, 128, 256))
    conv = ops.convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = ops.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = ops.convolutional(conv_sobj_branch, (1, 1, 256, 3*(NUM_CLASS+5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv3_tiny(input_layer, NUM_CLASS):
    route_1, conv = backbone.darknet53_tiny(input_layer)

    conv = ops.convolutional(conv, (1, 1, 1024, 256))

    conv_lobj_branch = ops.convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = ops.convolutional(conv_lobj_branch, (1, 1, 512, 3*(NUM_CLASS+5)), activate=False, bn=False)

    conv = ops.convolutional(conv, (1, 1, 256, 128))
    conv = ops.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = ops.convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = ops.convolutional(conv_mobj_branch, (1, 1, 256, 3*(NUM_CLASS+5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

if __name__ == '__main__':
    input_layer = tf.keras.layers.Input([416, 416, 3])
    feature_maps = YOLOv3_tiny(input_layer, 2)
    model = tf.keras.Model(input_layer, feature_maps)

    for layer in model.layers:
        print(layer.name, str(type(layer)).split('.')[-1])
