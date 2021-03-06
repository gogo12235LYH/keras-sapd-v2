import tensorflow as tf
import tensorflow.keras.backend as k
import numpy as np
import tensorflow.keras as keras
from utils import util_graph


@tf.function(jit_compile=True)
def standardize_weight(kernel, eps):
    mean = tf.math.reduce_mean(kernel, axis=(0, 1, 2), keepdims=True)
    # std = tf.math.reduce_std(kernel, axis=(0, 1, 2), keepdims=True)
    std = tf.sqrt(tf.math.reduce_variance(kernel, axis=(0, 1, 2), keepdims=True) + 1e-12)
    return (kernel - mean) / (std + eps)


class WSConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super(WSConv2D, self).__init__(*args, **kwargs)

    def call(self, inputs, eps=1e-5):
        self.kernel.assign(standardize_weight(self.kernel, eps))
        return super().call(inputs)


@tf.function(jit_compile=True)
def _layer_centerness(reg_input, factor=0.5, bias=0.):
    # r, t, l, b
    # 0, 1, 2, 3
    align_ = tf.minimum(reg_input[..., 0], reg_input[..., 2]) * tf.minimum(reg_input[..., 1], reg_input[..., 3]) / \
             tf.maximum(reg_input[..., 0], reg_input[..., 2]) / tf.maximum(reg_input[..., 1], reg_input[..., 3])
    align_ = (tf.expand_dims(align_, axis=-1) ** factor) + bias
    return align_


class AlignLayer(keras.layers.Layer):
    def __init__(self, width=256, factor=1., bias=0., *args, **kwargs):
        super(AlignLayer, self).__init__(*args, **kwargs)
        self.width = width
        self.factor = factor
        self.bias = bias

        self.align_conv2d = keras.layers.Conv2D(
            filters=width,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01)
        )

        self.multiply = keras.layers.Multiply()
        self.merge = keras.layers.Add()

    def call(self, inputs, **kwargs):
        reg_out = inputs[0]

        # compute centerness, (Batch, fmap_h, fmap_w) --> (Batch, fmap_h, fmap_w, 256)
        reg_out = _layer_centerness(reg_out, self.factor, self.bias)  # (Batch, fmap_h, fmap_w, 1)
        # reg_out = tf.tile(reg_out, (1, 1, 1, self.width))  # TODO: Not sure for this.

        # multiply with FPN's feature map
        reg_out = self.multiply([reg_out, inputs[1]])

        # using 1 * 1 conv2d
        reg_out = self.align_conv2d(reg_out)
        reg_out = self.merge([reg_out, inputs[1]])
        return reg_out


class BatchNormalization(keras.layers.BatchNormalization):
    """
        Rebuild keras.layers.BatchNormalization.

        To avoid error on keyword "freeze".
    """

    def __init__(self, freeze, *args, **kwargs):
        self.freeze = freeze
        super(BatchNormalization, self).__init__(*args, **kwargs)

        self.trainable = not self.freeze

    def call(self, inputs, training=None, **kwargs):
        if not training:
            return super(BatchNormalization, self).call(inputs, training=False)
        else:
            return super(BatchNormalization, self).call(inputs, training=(not self.freeze))

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        config.update({
            'freeze': self.freeze
        })
        return config


class UpsampleLike(keras.layers.Layer):
    """
        FPN's up-sample-like layers.

        Src : RetinaNet, https://github.com/fizyr/keras-retinanet
    """

    def call(self, inputs, **kwargs):
        src, target = inputs
        target_shape = k.shape(target)
        return util_graph.resize_images(src, (target_shape[1], target_shape[2]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1])
        # return input_shape[0][0], input_shape[1][1], input_shape[1][2], input_shape[0][-1]


class Locations2(keras.layers.Layer):
    def __init__(self, strides=(8, 16, 32, 64, 128), *args, **kwargs):
        self.strides = strides
        super(Locations2, self).__init__(dtype='float32', *args, **kwargs)

    def call(self, inputs, **kwargs):
        # Each inputs' shape : (B, F_H, F_W, Filter) from FPN's [P3, P4, P5, P6, P7]
        feature_shapes = [k.shape(feature)[1:3] for feature in inputs]

        locations_per_feature = []
        strides_per_feature = []
        # Ex: size = [80, 40, 20, 10, 5]
        for feature_shape, stride in zip(feature_shapes, self.strides):
            # Ex : feature_shape = (80, 80, 256), stride = 8
            height, width = feature_shape[0], feature_shape[1]

            # (80,)
            shift_x = tf.range(0, width * stride, delta=stride, dtype=np.float32)
            shift_y = tf.range(0, height * stride, delta=stride, dtype=np.float32)

            # (80, 80)
            shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

            # (6400,)
            shift_x = tf.reshape(shift_x, (-1,))
            shift_y = tf.reshape(shift_y, (-1,))

            # From FCOS's paper : location was defined as (stride * x + stride // 2, stride * y + stride // 2)
            locations = tf.stack((shift_x, shift_y), axis=1) + stride // 2
            locations_per_feature.append(locations)

            """
            """
            strides = tf.ones((height, width)) * stride
            strides = tf.reshape(strides, (-1,))
            strides_per_feature.append(strides)

        # total = 6400 + 1600 + 400 + 100 + 25 = 8525
        # (8525, 2)
        locations = tf.concat(locations_per_feature, axis=0)

        # (B, 8525, 2)
        locations = tf.tile(tf.expand_dims(locations, axis=0), (tf.shape(inputs[0])[0], 1, 1))

        strides = tf.concat(strides_per_feature, axis=0)
        strides = tf.tile(tf.expand_dims(strides, axis=0), (tf.shape(inputs[0])[0], 1))
        return [locations, strides]

    def get_config(self):
        config = super(Locations2, self).get_config()
        config.update(
            {
                'strides': self.strides
            }
        )
        return config


class RegressionBoxes2(keras.layers.Layer):
    """
        Input locations and regression which are from Feature maps and model.outputs[0].
        locations : Feature maps' location.
        Regression : It is predicted by model's regression sub-model.

        Src : Retinanet, https://github.com/fizyr/keras-retinanet
    """

    def __init__(self, **kwargs):
        super(RegressionBoxes2, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        locations, strides, regression = inputs

        x1 = locations[:, :, 0] - regression[:, :, 0] * 4.0 * strides[:, :]
        y1 = locations[:, :, 1] - regression[:, :, 1] * 4.0 * strides[:, :]
        x2 = locations[:, :, 0] + regression[:, :, 2] * 4.0 * strides[:, :]
        y2 = locations[:, :, 1] + regression[:, :, 3] * 4.0 * strides[:, :]

        return tf.stack([x1, y1, x2, y2], axis=-1)

    def get_config(self):
        config = super(RegressionBoxes2, self).get_config()
        return config


class ClipBoxes2(keras.layers.Layer):
    """
        ??????????????????????????????????????????????????????????????????????????????
    """

    def __init__(self, **kwargs):
        super(ClipBoxes2, self).__init__(dtype='float32', **kwargs)

    def call(self, inputs, *args, **kwargs):
        # inputs.shape = [(B, height, width, channel)(B, FS, 4)]
        inputs_image, boxes = inputs
        shape = tf.cast(tf.shape(inputs_image), tf.float32)
        width = shape[2]
        height = shape[1]

        # (B, FS, 1)
        x1 = tf.clip_by_value(boxes[:, :, 0], 0, width - 1)
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, height - 1)
        x2 = tf.clip_by_value(boxes[:, :, 2], 0, width - 1)
        y2 = tf.clip_by_value(boxes[:, :, 3], 0, height - 1)

        # (B, FS, 4)
        return tf.stack([x1, y1, x2, y2], axis=-1)

    def compute_output_shape(self, input_shape):
        # (B, FS, 4)
        return input_shape[1]


def filter_detections2(
        boxes,
        classification,
        class_specific_filter=True,
        nms=1,
        s_nms=0,
        score_threshold=0.01,
        max_detections=300,
        nms_threshold=0.5,
):
    def _filter_detections(scores_, labels_):
        # threshold based on score
        # (num_score_keeps, 1)
        indices_ = tf.where(keras.backend.greater(scores_, score_threshold))

        if nms:
            # (num_score_keeps, 4)
            filtered_boxes = tf.gather_nd(boxes, indices_)
            filtered_scores = keras.backend.gather(scores_, indices_)[:, 0]

            # perform NMS
            nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections,
                                                       iou_threshold=nms_threshold)

            # filter indices based on NMS
            # (num_score_nms_keeps, 1)
            indices_ = keras.backend.gather(indices_, nms_indices)

        elif s_nms:
            # (num_score_keeps, 4)
            filtered_boxes = tf.gather_nd(boxes, indices_)
            filtered_scores = keras.backend.gather(scores_, indices_)[:, 0]

            # perform Soft_NMS
            nms_indices = tf.image.non_max_suppression_with_scores(
                filtered_boxes, filtered_scores,
                max_output_size=max_detections,
                iou_threshold=nms_threshold,
                soft_nms_sigma=0.5,
            )[0]

            # filter indices based on NMS
            # (num_score_nms_keeps, 1)
            indices_ = keras.backend.gather(indices_, nms_indices)

        # add indices to list of all indices
        # (num_score_nms_keeps, )
        labels_ = tf.gather_nd(labels_, indices_)
        # (num_score_nms_keeps, 2)
        indices_ = keras.backend.stack([indices_[:, 0], labels_], axis=1)

        return indices_

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        # (concatenated_num_score_nms_keeps, 2)
        indices = keras.backend.concatenate(all_indices, axis=0)
    else:
        scores = keras.backend.max(classification, axis=1)
        labels = keras.backend.argmax(classification, axis=1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores = tf.gather_nd(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices = keras.backend.gather(indices[:, 0], top_indices)
    boxes = keras.backend.gather(boxes, indices)
    labels = keras.backend.gather(labels, top_indices)

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = keras.backend.cast(labels, 'int32')

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])

    return [boxes, scores, labels]


class FilterDetections2(keras.layers.Layer):
    """
    Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
            self,
            nms=0,
            s_nms=0,
            class_specific_filter=True,
            nms_threshold=0.5,
            score_threshold=0.01,
            max_detections=300,
            parallel_iterations=32,
            **kwargs
    ):
        super(FilterDetections2, self).__init__(dtype='float32', **kwargs)
        self.nms = nms
        self.s_nms = s_nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations

    def call(self, inputs, **kwargs):
        """
        Constructs the NMS graph.
        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes = inputs[0]
        classification = inputs[1]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes_ = args[0]
            classification_ = args[1]

            return filter_detections2(
                boxes_,
                classification_,
                nms=self.nms,
                s_nms=self.s_nms,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold,
            )

        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes, classification],
            fn_output_signature=['float32', 'float32', 'int32'],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def get_config(self):
        """
        Gets the configuration of this layer.
        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections2, self).get_config()
        config.update({
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            'score_threshold': self.score_threshold,
            'max_detections': self.max_detections,
            'parallel_iterations': self.parallel_iterations,
        })
        return config
