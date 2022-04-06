import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow_addons.layers import GroupNormalization


class Resizing(keras.layers.Layer):
    def __init__(self, size, name='resize', **kwargs):
        super(Resizing, self).__init__(name=name, **kwargs)

        self.size = size

    @tf.function
    def call(self, inputs, *args, **kwargs):
        return tf.image.resize(
            inputs, size=self.size
        )

    def get_config(self):
        c = super(Resizing, self).get_config()
        c.update({
            'size': self.size,
        })
        return c


class StandardHead(keras.layers.Layer):
    def __init__(self, width, depth, num_cls, gn=1, name='Std_head', **kwargs):
        super(StandardHead, self).__init__(name=name, **kwargs)

        _conv2d_setting = {
            'filters': width,
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': tf.initializers.RandomNormal(0.0, 0.01),
        }

        self.cls_blocks = keras.Sequential()
        self.loc_blocks = keras.Sequential()

        for _ in range(depth):
            self.cls_blocks.add(keras.layers.Conv2D(**_conv2d_setting))
            self.loc_blocks.add(keras.layers.Conv2D(**_conv2d_setting))

            if gn:
                self.cls_blocks.add(GroupNormalization(groups=32))

            self.cls_blocks.add(keras.layers.ReLU())
            self.loc_blocks.add(keras.layers.ReLU())

        self.mp_filter = MaxPoolFilter()

        self.cls_conv2d = keras.layers.Conv2D(
            filters=num_cls, kernel_size=3, strides=1, padding='same',
            kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
            bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        )
        self.loc_conv2d = keras.layers.Conv2D(
            filters=4, kernel_size=3, strides=1, padding='same',
            kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
            bias_initializer=tf.constant_initializer(0.1),
        )

        self.cls_act = keras.layers.Activation('sigmoid', dtype=tf.float32)
        self.loc_act = keras.layers.Activation('relu', dtype=tf.float32)
        self.cls_reshape = keras.layers.Reshape((-1, num_cls))
        self.loc_reshape = keras.layers.Reshape((-1, 4))

    @tf.function
    def call(self, inputs, training=None, mask=None):
        cls = self.cls_blocks(inputs)
        loc = self.loc_blocks(inputs)

        loc_pool = self.mp_filter(loc)
        loc_pool = self.cls_act(loc_pool)

        cls = self.cls_conv2d(cls)
        loc = self.loc_conv2d(loc)

        cls = self.cls_act(cls) * loc_pool
        loc = self.loc_act(loc)

        cls = self.cls_reshape(cls)
        loc = self.loc_reshape(loc)
        return cls, loc

    def get_config(self):
        cfg = super(StandardHead, self).get_config()
        return cfg


class MaxPoolFilter(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaxPoolFilter, self).__init__(name='maxpool_filter', **kwargs)

        _conv2d_setting = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': tf.initializers.RandomNormal(0.0, 0.01),
        }

        self.conv2d_f = keras.layers.Conv2D(filters=256, activation='relu', **_conv2d_setting)
        self.conv2d_s = keras.layers.Conv2D(filters=1, **_conv2d_setting)

        self.pool = keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=1,
            padding='same',
        )

        self.normal = GroupNormalization(groups=32)
        self.act = keras.layers.ReLU()

    @tf.function
    def call(self, inputs, *args, **kwargs):
        # out = self.conv2d_f(inputs)
        # out += inputs
        # out = self.normal(out)
        # out = self.act(out)
        # out = self.conv2d_s(out)
        # return out

        loc_features = [self.conv2d_f(feature) for feature in inputs]
        loc_outputs = []

        for i, loc_feature in enumerate(loc_features):
            max_loc_3d = self.pool(loc_feature)
            max_loc_3d += inputs[i]
            max_loc_3d = self.act(self.normal(max_loc_3d))
            max_loc_3d = self.conv2d_s(max_loc_3d)
            loc_outputs.append(max_loc_3d)

        return loc_outputs

    def get_config(self):
        cfg = super(MaxPoolFilter, self).get_config()
        return cfg


class MaxPool3DFilter(keras.layers.Layer):
    def __init__(self, name='maxpool3d_filter', **kwargs):
        super(MaxPool3DFilter, self).__init__(name=name, **kwargs)

        self.m = 1

        self.conv2d_f = keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=tf.initializers.RandomNormal(.0, .01),
            activation='relu',
        )
        self.conv2d_s = keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=tf.initializers.RandomNormal(.0, .01),
        )

        self.pool = keras.layers.MaxPooling3D(
            pool_size=(2 + self.m, 3, 3),
            strides=1,
            padding='same',
        )

        self.normal = GroupNormalization(groups=32)
        self.act = keras.layers.ReLU()

    @tf.function
    def call(self, inputs, *args, **kwargs):
        loc_features = [self.conv2d_f(feature) for feature in inputs]
        loc_outputs = []

        for i, loc_feature in enumerate(loc_features):
            stt_ind = max(0, i - self.m)
            stp_ind = min(len(inputs), i + self.m + 1)

            resize = Resizing(size=tf.shape(loc_feature)[1:3])

            loc_3ds = [resize(loc_features[c]) for c in range(stt_ind, stp_ind)]

            # (B, 2 + m, fh, fw, 256)
            loc_3ds = tf.stack(loc_3ds, axis=1)
            max_loc_3d = self.pool(loc_3ds)[:, min(i, self.m), :, :, :]
            max_loc_3d += inputs[i]
            max_loc_3d = self.act(self.normal(max_loc_3d))
            max_loc_3d = self.conv2d_s(max_loc_3d)

            loc_outputs.append(max_loc_3d)

        return loc_outputs

    def get_config(self):
        cfg = super(MaxPool3DFilter, self).get_config()
        return cfg


def MPSubnetworks(input_features, width=256, depth=4, num_cls=20):
    cls, reg = [], []

    subnetworks = StandardHead(width=width, depth=depth, num_cls=num_cls)

    for feature in input_features:
        outputs = subnetworks(feature)
        cls.append(outputs[0])
        reg.append(outputs[1])

    cls_out = keras.layers.Concatenate(axis=1, name='cls_head')(cls)
    reg_out = keras.layers.Concatenate(axis=1, name='reg_head')(reg)
    return cls_out, reg_out


class Subnetworks(keras.Model):
    def __init__(self, width=256, depth=4, num_cls=20, name='MPHead', th=1, **kwargs):
        super(Subnetworks, self).__init__(name=name if th else 'MP3dHead', **kwargs)

        self.cls_subnet = keras.Sequential()
        self.loc_subnet = keras.Sequential()

        _common_init = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': tf.initializers.RandomNormal(.0, .01),
        }

        for _ in range(depth):
            self.cls_subnet.add(keras.layers.Conv2D(filters=width, **_common_init))
            self.loc_subnet.add(keras.layers.Conv2D(filters=width, **_common_init))

            self.cls_subnet.add(GroupNormalization(groups=32))

            self.cls_subnet.add(keras.layers.ReLU())
            self.loc_subnet.add(keras.layers.ReLU())

        self.cls_conv2d = keras.layers.Conv2D(
            filters=num_cls,
            bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
            **_common_init,
        )
        self.loc_conv2d = keras.layers.Conv2D(
            filters=4,
            bias_initializer=tf.constant_initializer(0.1),
            **_common_init,
        )

        self.cls_act = keras.layers.Activation('sigmoid', dtype=tf.float32)
        self.loc_act = keras.layers.Activation('relu', dtype=tf.float32)

        self.cls_reshape = keras.layers.Reshape((-1, num_cls))
        self.loc_reshape = keras.layers.Reshape((-1, 4))

        self.maxpool = MaxPoolFilter() if th else MaxPool3DFilter()
        self.mpf_reshape = keras.layers.Reshape((-1, 1))

        self.cls_merge = keras.layers.Multiply()

    @tf.function
    def call(self, inputs, training=None, mask=None):
        cls, loc, loc_f = [], [], []

        for input_f in inputs:
            cls_out = self.cls_subnet(input_f)
            loc_ = self.loc_subnet(input_f)

            loc_f.append(loc_)

            cls_out = self.cls_conv2d(cls_out)
            cls.append(self.cls_reshape(self.cls_act(cls_out)))

            loc_out = self.loc_conv2d(loc_)
            loc.append(self.loc_reshape(self.loc_act(loc_out)))

        loc_fs = [self.mpf_reshape(self.cls_act(x)) for x in self.maxpool(loc_f)]
        loc_fs = keras.layers.Concatenate(axis=1, name='loc_filters')(loc_fs)

        cls_out = keras.layers.Concatenate(axis=1, name='cls_out')(cls)
        cls_out = self.cls_merge([cls_out, loc_fs])

        loc_out = keras.layers.Concatenate(axis=1, name='loc_out')(loc)

        return cls_out, loc_out

    def get_config(self):
        cfg = super(Subnetworks, self).get_config()
        return cfg
