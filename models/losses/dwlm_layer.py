import config
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from models.losses import compute_focal_v2, compute_iou_v2

_IMG_SIZE = [512, 640]

_FMAP_AREA = np.power(
    [
        _IMG_SIZE[config.PHI] // 8,
        _IMG_SIZE[config.PHI] // 16,
        _IMG_SIZE[config.PHI] // 32,
        _IMG_SIZE[config.PHI] // 64,
        _IMG_SIZE[config.PHI] // 128
    ],
    2
)

_FMAP_AREA_SUM = np.sum(_FMAP_AREA)


@tf.function
def _create_detections(fpn_level=5):
    detections = []
    for i in range(fpn_level):
        padding_fa_fnt = np.sum(_FMAP_AREA[:i])
        padding_fa_bak = _FMAP_AREA_SUM - _FMAP_AREA[i] - padding_fa_fnt
        detections.append(
            tf.pad(
                tf.ones((_FMAP_AREA[i],)),
                tf.constant([[padding_fa_fnt, padding_fa_bak]])
            )
        )

    # (fpn_level, anchor-points)
    detections = tf.stack(detections)
    return detections


@tf.function
def _map_fn_dwlm(total_loss, mask, ind, bbox_cnt):
    ind = tf.cast(ind, tf.int32)
    bbox_cnt = tf.cast(bbox_cnt, tf.int32)

    detections = _create_detections()  # level_onehot

    def _compute_target(args):
        # _batch_total_loss: (anchor-points, )
        # _batch_mask: (anchor-points, )
        # _batch_ind: (anchor-points, )
        # _batch_bbox_cnt: (1, )

        _batch_loss = args[0]
        _batch_mask = args[1]
        _batch_ind = args[2]
        _batch_bbox_cnt = args[3]

        # (objects, ) -> (objects, 1)
        object_ids = tf.range(0, _batch_bbox_cnt, dtype=tf.int32)
        object_ids = tf.reshape(object_ids, [-1, 1])

        def _build_target(arg):
            object_id = arg

            # (anchor-points, )
            object_boolean_mask = tf.where(tf.equal(_batch_ind, object_id[0]), 1., 0.)

            # (anchor-points) * (5, anchor-points) ->　(5, anchor-points)
            detections_mask = object_boolean_mask * detections

            # (5, anchor-points)
            object_loss = _batch_loss * detections_mask

            # (5, )
            object_loss_mean = tf.reduce_sum(object_loss, axis=1) / tf.maximum(1.,
                                                                               tf.reduce_sum(detections_mask, axis=1))
            object_loss_min = tf.reduce_min(object_loss_mean)
            object_loss_max = tf.reduce_max(object_loss_mean)

            object_target = 1 - ((object_loss_mean - object_loss_min) / (object_loss_max - object_loss_min))

            min_weight = tf.nn.top_k(object_target, k=4)[0][..., -1]

            object_target = tf.where(
                tf.greater(object_target, min_weight),
                object_target,
                0.
            )

            # (5, 1) tile(1, 8525) -> (5, 8525) reduce(axis=0) -> (8525, )
            object_target = tf.reduce_sum(
                (tf.tile(object_target[..., None], (1, 8525)) * detections_mask),
                axis=0
            )
            return object_target

        object_targets = tf.map_fn(
            _build_target,
            elems=[object_ids],
            fn_output_signature=tf.float32
        )

        # (objects, anchor-points) -> (anchor-points, )
        object_targets = tf.reduce_sum(object_targets, axis=0)

        return object_targets

    # (batch, anchor-points)
    return tf.map_fn(
        _compute_target,
        elems=[total_loss, mask, ind, bbox_cnt],
        fn_output_signature=tf.float32
    )


@tf.function
def _test_bench(total_loss, mask, ind, bbox_cnt, top_k=1):
    # take first batch
    _batch_ind = tf.cast(ind, tf.int32)[0]
    _batch_bbox_cnt = tf.cast(bbox_cnt, tf.int32)[0]
    _batch_mask = mask[0]
    _batch_loss = total_loss[0]

    # (objects, ) -> (objects, 1)
    object_ids = tf.range(0, _batch_bbox_cnt, dtype=tf.int32)
    object_ids = tf.reshape(object_ids, [-1, 1])

    detections = _create_detections()

    def _build_target(arg):
        object_id = arg

        object_boolean_mask = tf.equal(_batch_ind, object_id[0])

        # (anchor-points, )
        object_mask = tf.where(object_boolean_mask, 1., 0.)

        detections_mask = object_mask * detections

        # (5, anchor-points)
        object_loss = _batch_loss * detections_mask

        # (5, )
        object_ll_mean = tf.reduce_sum(object_loss, axis=1) / tf.maximum(1., tf.reduce_sum(detections_mask, axis=1))
        object_l_max = tf.reduce_max(object_ll_mean)
        object_l_min = tf.reduce_min(object_ll_mean)

        object_target = 1 - (object_ll_mean - object_l_min) / (object_l_max - object_l_min)
        # object_target = tf.nn.softmax(object_target * 10)

        min_weight = tf.nn.top_k(object_target, k=(top_k + 1))[0][..., -1]

        object_target = tf.where(
            tf.greater(object_target, min_weight),
            object_target,
            0.
        )

        # object_target = tf.reduce_sum(
        #     (tf.tile(object_target[..., None], (1, 8525)) * detections_mask), axis=0
        # )
        return object_target

    object_targets = tf.map_fn(
        _build_target,
        elems=[object_ids],
        fn_output_signature=tf.float32
    )

    return object_targets


class DWLMLayer(keras.layers.Layer):
    def __init__(self, name='dwlm_layer', **kwargs):
        super(DWLMLayer, self).__init__(dtype='float32', name=name, **kwargs)

        self.cls_loss_fn = compute_focal_v2(alpha=.25, gamma=2.)
        self.loc_loss_fn = compute_iou_v2(mode='giou')

    def call(self, inputs, *args, **kwargs):
        # (batch, anchor-points, classes); (batch, anchor-points, 4)
        cls_pred, loc_pred = inputs[0], inputs[1]

        # (batch, anchor-points, classes + 2); (batch, anchor-points, 4 + 2)
        cls_tar, loc_tar = inputs[2], inputs[3]

        # (batch, anchor-points, ); (batch, )
        ind_tar, bboxes_cnt = inputs[4][..., 0], inputs[5][..., 0]

        # (None, 8525, )
        cls_loss = self.cls_loss_fn(cls_tar[..., :-2], cls_pred)
        loc_loss = self.loc_loss_fn(loc_tar, loc_pred)

        dwlm_out = _map_fn_dwlm((cls_loss + loc_loss), cls_tar[..., -1], ind_tar, bboxes_cnt)
        test_tar = _test_bench((cls_loss + loc_loss), cls_tar[..., -1], ind_tar, bboxes_cnt)

        dwlm_out_masked = tf.where(tf.greater(cls_tar[..., -1], 0.), dwlm_out, 1.)
        # cls_loss = tf.where(tf.greater(cls_tar[..., -1], 0.), cls_loss, 1.)
        # loc_loss = tf.where(tf.greater(cls_tar[..., -1], 0.), loc_loss, 1.)

        return tf.expand_dims(dwlm_out_masked, axis=-1)
        # return tf.expand_dims(out_masked, axis=-1), tf.expand_dims(test_tar, axis=-1), cls_loss, loc_loss

    def get_config(self):
        cfg = super(DWLMLayer, self).get_config()
        return cfg


if __name__ == '__main__':
    """ Test """

    batch_size = 1

    _cls_pred = tf.random.uniform((batch_size, 8525, config.NUM_CLS))
    _loc_pred = tf.random.uniform((batch_size, 8525, 4))

    """ """

    import numpy as np
    from generators.pipeline import create_pipeline_v2

    train_t, test_t = create_pipeline_v2(
        phi=1,
        batch_size=batch_size,
        debug=True,
        db="VOC"
    )

    _cls_tar, _loc_tar, _ind_tar, _int_tar = None, None, None, None
    iterations = 1

    for step, inputs_batch in enumerate(train_t):
        if (step + 1) > iterations:
            break

        _cls_tar = inputs_batch['cls_target'].numpy()
        _loc_tar = inputs_batch['loc_target'].numpy()
        _ind_tar = inputs_batch['ind_target'].numpy()
        _int_tar = inputs_batch['bboxes_cnt'].numpy()

    """ """

    _dm_out, _dm_out_test, _cls_loss, _loc_loss = DWLMLayer()(
        [_cls_pred, _loc_pred, _cls_tar, _loc_tar, _ind_tar, _int_tar]
    )

    _dm_out = _dm_out.numpy()
    _dm_out_test = _dm_out_test.numpy()

    _cls_loss = _cls_loss.numpy()
    _loc_loss = _loc_loss.numpy()

    p7_sap = np.reshape(_cls_tar[0, 8500:, -2], (5, 5))
    p6_sap = np.reshape(_cls_tar[0, 8400:8500, -2], (10, 10))
    p5_sap = np.reshape(_cls_tar[0, 8000:8400, -2], (20, 20))

    # p7_tar = np.reshape(_fmn_tar_[0, 8500:], (5, 5))
    # p6_tar = np.reshape(_fmn_tar_[0, 8400:8500], (10, 10))
    # p5_tar = np.reshape(_fmn_tar_[0, 8000:8400], (20, 20))

    p7_ind = np.reshape(_ind_tar[0, 8500:], (5, 5))
    p6_ind = np.reshape(_ind_tar[0, 8400:8500], (10, 10))
    p5_ind = np.reshape(_ind_tar[0, 8000:8400], (20, 20))
