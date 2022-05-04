import tensorflow as tf
import tensorflow.keras as keras


class CENLoss(keras.layers.Layer):
    def __init__(self, factor=1.0, name='cen_loss', **kwargs):
        super(CENLoss, self).__init__(dtype='float32', name=name, **kwargs)
        self.factor = factor

    def call(self, inputs, **kwargs):
        y_true, pos_mask = inputs[0][..., None], inputs[1][..., None]
        y_pred = inputs[2]

        y_true = tf.boolean_mask(y_true, pos_mask)[..., None]
        y_pred = tf.boolean_mask(y_pred, pos_mask)[..., None]

        loss_bce = keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
        loss = tf.reduce_mean(loss_bce)

        self.add_loss(loss)
        self.add_metric(loss, name=self.name)

        return loss
        # return loss_bce
        # return loss_bce, loss_1, loss_2

    def get_config(self):
        cfg = super(CENLoss, self).get_config()
        cfg.update(
            {'factor': self.factor}
        )
        return cfg


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    batch_size = 1

    _cen_pred = tf.random.uniform((batch_size, 8525, 1))

    _test_cen_layer = CENLoss(factor=1.0)

    """ """

    import numpy as np
    from generators.pipeline import create_pipeline_v2

    train_t, test_t = create_pipeline_v2(
        phi=1,
        batch_size=batch_size,
        debug=True,
        db="DPCB"
    )

    _cls_tar, _loc_tar, _ind_tar, _int_tar = None, None, None, None
    iterations = 1

    for step, inputs_batch in enumerate(train_t):

        _cls_tar = inputs_batch['cls_target'].numpy()
        _loc_tar = inputs_batch['loc_target'].numpy()
        _ind_tar = inputs_batch['ind_target'].numpy()
        _int_tar = inputs_batch['bboxes_cnt'].numpy()

        _cen_loss_bce = _test_cen_layer(
            [_cls_tar[..., -2], _cls_tar[..., -1], _cen_pred]
        )

        _cen_loss_bce = _cen_loss_bce.numpy()
        # _cen_loss_1 = _cen_loss_1.numpy()
        # _cen_loss_2 = _cen_loss_2.numpy()

        if iterations > 1:
            break
