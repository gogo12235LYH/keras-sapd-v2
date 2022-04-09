import tensorflow as tf
import tensorflow.keras as keras

_k_init = tf.initializers.RandomNormal(0.0, 0.01)

_lateral_dict = {
    'filters': 256,
    'kernel_size': 1,
    'strides': 1,
    'padding': 'same',
}

_same_dict = {
    'filters': 256,
    'kernel_size': 3,
    'strides': 1,
    'padding': 'same',
}

_down_dict = {
    'filters': 256,
    'kernel_size': 3,
    'strides': 2,
    'padding': 'same',
}


class FeaturePyramidNetwork(keras.layers.Layer):
    """
        From : https://keras.io/examples/vision/retinanet/
    """

    def __init__(self, interpolation='nearest', name='FPN', **kwargs):
        super(FeaturePyramidNetwork, self).__init__(name=name, **kwargs)

        self.interpolation = interpolation

        # lateral branch
        self.l_conv2d_c3 = keras.layers.Conv2D(**_lateral_dict)
        self.l_conv2d_c4 = keras.layers.Conv2D(**_lateral_dict)
        self.l_conv2d_c5 = keras.layers.Conv2D(**_lateral_dict)

        # same branch
        self.conv2d_p3 = keras.layers.Conv2D(**_same_dict)
        self.conv2d_p4 = keras.layers.Conv2D(**_same_dict)
        self.conv2d_p5 = keras.layers.Conv2D(**_same_dict)

        # down branch
        self.down_conv2d_p6 = keras.layers.Conv2D(**_down_dict)
        self.down_conv2d_p7 = keras.layers.Conv2D(**_down_dict)

        self.sampling = keras.layers.UpSampling2D(2, interpolation=interpolation)

    def call(self, inputs, training=None, mask=None):
        # P3, P4, P5, P6, P7
        # 80, 40, 20, 10, 5
        c3, c4, c5 = inputs[0], inputs[1], inputs[2]

        p5 = self.l_conv2d_c5(c5)
        p5 = self.conv2d_p5(p5)

        # Up-sample and Merge
        p4 = self.l_conv2d_c4(c4)
        p4 = keras.layers.Add()([self.sampling(p5), p4])
        p4 = self.conv2d_p4(p4)

        # Up-sample and Merge
        p3 = self.l_conv2d_c3(c3)
        p3 = keras.layers.Add()([self.sampling(p4), p3])
        p3 = self.conv2d_p3(p3)

        # down-sampling
        p6 = self.down_conv2d_p6(c5)

        # down-sampling
        p7 = self.down_conv2d_p7(tf.nn.relu(p6))

        return p3, p4, p5, p6, p7

    def get_config(self):
        c = super(FeaturePyramidNetwork, self).get_config()
        return c
