"""

    Ref: https://keras.io/examples/vision/gradient_centralization/

"""

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.optimizers import SGDW, AdamW


class SGD_GC(SGD):
    def get_gradients(self, loss, params):
        grads = []
        gradients = super().get_gradients()

        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads


class SGDW_GC(SGDW):
    def get_gradients(self, loss, params):
        grads = []
        gradients = super().get_gradients()

        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads


class Adam_GC(Adam):
    def get_gradients(self, loss, params):
        grads = []
        gradients = super().get_gradients()

        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads


class AdamW_GC(AdamW):
    def get_gradients(self, loss, params):
        grads = []
        gradients = super().get_gradients()

        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads

