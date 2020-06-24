import tensorflow as tf
import model.ops as ops

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = ops.convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = ops.convolutional(conv, filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)
    residual_output = short_cut + conv
    return residual_output

def darknet53(input_data):
    input_data = ops.convolutional(input_data, (3, 3, 3, 32))
    input_data = ops.convolutional(input_data, (3, 3, 32, 64), downsample=True)

    for i in range(1):
        input_data = residual_block(input_data, 64, 32, 64)

    input_data = ops.convolutional(input_data, (3, 3, 64, 128), downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 128, 64, 128)

    input_data = ops.convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = ops.convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = ops.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data

def darknet53_tiny(input_data):
    input_data = ops.convolutional(input_data, (3, 3, 3, 16))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = ops.convolutional(input_data, (3, 3, 16, 32))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = ops.convolutional(input_data, (3, 3, 32, 64))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = ops.convolutional(input_data, (3, 3, 64, 128))
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = ops.convolutional(input_data, (3, 3, 128, 256))

    route_1 = input_data

    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
    input_data = ops.convolutional(input_data, (3, 3, 256, 512))

    input_data = tf.keras.layers.MaxPool2D(2, 1, 'same')(input_data)
    input_data = ops.convolutional(input_data, (3, 3, 512, 1024))

    return route_1, input_data

def cspdarknet53(input_data):

    input_data = ops.convolutional(input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = ops.convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")

    route = input_data
    route = ops.convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = ops.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64, activate_type="mish")
    input_data = ops.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = ops.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = ops.convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
    route = input_data
    route = ops.convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = ops.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = residual_block(input_data, 64, 64, 64, activate_type="mish")
    input_data = ops.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = ops.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = ops.convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    route = input_data
    route = ops.convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = ops.convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(8):
        input_data = residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = ops.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = ops.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data
    input_data = ops.convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    route = input_data
    route = ops.convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = ops.convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(8):
        input_data = residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = ops.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = ops.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data
    input_data = ops.convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = ops.convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = ops.convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(4):
        input_data = residual_block(input_data, 512, 512, 512, activate_type="mish")
    input_data = ops.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = ops.convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = ops.convolutional(input_data, (1, 1, 1024, 512))
    input_data = ops.convolutional(input_data, (3, 3, 512, 1024))
    input_data = ops.convolutional(input_data, (1, 1, 1024, 512))

    input_data = tf.concat([
        tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1),
        tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1), 
        tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), 
        input_data
    ], axis=-1)

    input_data = ops.convolutional(input_data, (1, 1, 2048, 512))
    input_data = ops.convolutional(input_data, (3, 3, 512, 1024))
    input_data = ops.convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data
    