import config
import tensorflow as tf
import tensorflow.keras as keras
from utils.util_graph import trim_zero_padding_boxes, shrink_and_normalize_boxes, \
    create_reg_positive_sample
from models.neck import FeaturePyramidNetwork
from models.layers import Locations2, RegressionBoxes2, ClipBoxes2, FilterDetections2
from models.losses import FocalLoss, IoULoss, DWLMLayer, CENLoss, FMAP_AREA_SUM

STRIDES = (8, 16, 32, 64, 128)


def _build_map_function_target_dwlm(
        gt_boxes,
        feature_maps_shape,
):
    num_cls = config.NUM_CLS

    gt_labels = tf.cast(gt_boxes[:, 4], tf.int32)
    gt_boxes = gt_boxes[:, :4]
    gt_boxes, non_zeros = trim_zero_padding_boxes(gt_boxes)
    gt_labels = tf.boolean_mask(gt_labels, non_zeros)

    cls_target_ = tf.zeros((0, num_cls + 1 + 1), dtype=tf.float32)
    reg_target_ = tf.zeros((0, 4 + 1 + 1), dtype=tf.float32)
    ind_target_ = tf.zeros((0, 1), dtype=tf.int32)

    for level_id in range(len(STRIDES)):
        stride = STRIDES[level_id]

        fh = feature_maps_shape[level_id][0]
        fw = feature_maps_shape[level_id][1]

        pos_x1, pos_y1, pos_x2, pos_y2 = shrink_and_normalize_boxes(gt_boxes, fw, fh, stride, config.SHRINK_RATIO)

        def build_map_function_target(args):
            pos_x1_ = args[0]
            pos_y1_ = args[1]
            pos_x2_ = args[2]
            pos_y2_ = args[3]
            gt_box = args[4]
            gt_label = args[5]

            """ Create Negative sample """
            neg_top_bot = tf.stack((pos_y1_, fh - pos_y2_), axis=0)
            neg_lef_rit = tf.stack((pos_x1_, fw - pos_x2_), axis=0)
            neg_pad = tf.stack([neg_top_bot, neg_lef_rit], axis=0)

            """ Regression Target: create positive sample """
            # pos_shift_xx = (tf.cast(tf.range(pos_x1_, pos_x2_), dtype=tf.float32) + 0.5) * stride
            # pos_shift_yy = (tf.cast(tf.range(pos_y1_, pos_y2_), dtype=tf.float32) + 0.5) * stride
            # pos_shift_xx, pos_shift_yy = tf.meshgrid(pos_shift_xx, pos_shift_yy)
            # pos_shifts = tf.stack((pos_shift_xx, pos_shift_yy), axis=-1)
            # dl = tf.maximum(pos_shifts[:, :, 0] - gt_box[0], 0)
            # dt = tf.maximum(pos_shifts[:, :, 1] - gt_box[1], 0)
            # dr = tf.maximum(gt_box[2] - pos_shifts[:, :, 0], 0)
            # db = tf.maximum(gt_box[3] - pos_shifts[:, :, 1], 0)
            # deltas = tf.stack((dl, dt, dr, db), axis=-1)
            # level_box_regr_pos_target = deltas / 4.0 / stride
            # level_pos_box_ap_weight= tf.minimum(dl, dr) * tf.minimum(dt, db) / tf.maximum(dl, dr) / tf.maximum(dt,
            #                                                                                                    db)
            level_box_regr_pos_target, level_pos_box_ap_weight, level_box_pos_area = create_reg_positive_sample(
                gt_box, pos_x1_, pos_y1_, pos_x2_, pos_y2_, stride
            )
            level_pos_box_soft_weight = level_pos_box_ap_weight
            # level_pos_box_soft_weight = (1 - level_pos_box_ap_weight) * level_box_meta_select_weight  # ?

            """ Classification Target: create positive sample """
            level_pos_box_cls_target = tf.zeros((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, num_cls), dtype=tf.float32)
            level_pos_box_gt_label_col = tf.ones((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, 1), dtype=tf.float32)
            level_pos_box_cls_target = tf.concat((level_pos_box_cls_target[..., :gt_label],
                                                  level_pos_box_gt_label_col,
                                                  level_pos_box_cls_target[..., gt_label + 1:]), axis=-1)

            """ Padding Classification Target's negative sample """
            level_box_cls_target = tf.pad(level_pos_box_cls_target,
                                          tf.concat((neg_pad, tf.constant([[0, 0]])), axis=0))

            """ Padding Soft Anchor's negative sample """
            level_box_soft_weight = tf.pad(level_pos_box_soft_weight, neg_pad, constant_values=1)

            """ Creating Positive Sample locations and padding it's negative sample """
            level_pos_box_regr_mask = tf.ones((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_))
            level_box_regr_mask = tf.pad(level_pos_box_regr_mask, neg_pad)

            """ Padding Regression Target's negative sample """
            level_box_regr_target = tf.pad(level_box_regr_pos_target,
                                           tf.concat((neg_pad, tf.constant([[0, 0]])), axis=0))

            """ Output Target """
            # shape = (fh, fw, cls_num + 2)
            level_box_cls_target = tf.concat([level_box_cls_target, level_box_soft_weight[..., None],
                                              level_box_regr_mask[..., None]], axis=-1)
            # shape = (fh, fw, 4 + 2)
            level_box_regr_target = tf.concat([level_box_regr_target, level_box_soft_weight[..., None],
                                               level_box_regr_mask[..., None]], axis=-1)
            # level_box_pos_area = (dl + dr) * (dt + db)
            # (fh, fw)
            level_box_area = tf.pad(level_box_pos_area, neg_pad, constant_values=1e7)
            return level_box_cls_target, level_box_regr_target, level_box_area

        # cls_target : shape = (True_Label_count, fh, fw, cls_num + 2)
        # reg_target : shape = (True_Label_count, fh, fw, 4 + 2)
        # area : shape = (True_Label_count, fh, fw)
        level_cls_target, level_regr_target, level_area = tf.map_fn(
            build_map_function_target,
            elems=[pos_x1, pos_y1, pos_x2, pos_y2, gt_boxes, gt_labels],
            fn_output_signature=(tf.float32, tf.float32, tf.float32),
        )

        # min area : shape = (objects, fh, fw) --> (fh, fw)
        level_min_area_box_indices = tf.argmin(level_area, axis=0, output_type=tf.int32)
        # (fh, fw) --> (fh * fw)
        level_min_area_box_indices = tf.reshape(level_min_area_box_indices, (-1,))

        # (fw, ), (fh, )
        locs_x, locs_y = tf.range(0, fw), tf.range(0, fh)

        # (fh, fw) --> (fh * fw)
        locs_xx, locs_yy = tf.meshgrid(locs_x, locs_y)
        locs_xx = tf.reshape(locs_xx, (-1,))
        locs_yy = tf.reshape(locs_yy, (-1,))

        # (fh * fw, 3)
        level_indices = tf.stack((level_min_area_box_indices, locs_yy, locs_xx), axis=-1)

        """ Select """
        level_cls_target = tf.gather_nd(level_cls_target, level_indices)
        level_regr_target = tf.gather_nd(level_regr_target, level_indices)

        cls_target_ = tf.concat([cls_target_, level_cls_target], axis=0)
        reg_target_ = tf.concat([reg_target_, level_regr_target], axis=0)
        ind_target_ = tf.concat([ind_target_, tf.expand_dims(level_min_area_box_indices, -1)], axis=0)

    ind_target_ = tf.where(
        tf.equal(cls_target_[..., -1], 1.), ind_target_[..., 0], -1
    )[..., None]

    return [cls_target_, reg_target_, ind_target_, tf.shape(gt_boxes)[0][..., None]]


class Target_v1(keras.layers.Layer):
    def __init__(
            self,
            num_cls=config.NUM_CLS,
            strides=STRIDES,
            **kwargs
    ):
        super(Target_v1, self).__init__(dtype='float32', **kwargs)
        self.num_cls = num_cls,
        self.strides = strides,

    def call(self, inputs, **kwargs):
        # (Batch, 5, 2) -> (5, 2)
        feature_map_shapes = inputs[0][0]

        # (Batch, Max_Bboxes_count, 5)
        batch_gt_boxes = inputs[1]

        def _build_map_function_batch(args):
            """ For Batch axis. """
            gt_boxes = args[0]
            return _build_map_function_target_dwlm(gt_boxes=gt_boxes, feature_maps_shape=feature_map_shapes)

        # [cls_target_, reg_target_, ind_target_, tf.shape(gt_boxes)[0][..., None]]
        outputs = tf.map_fn(
            _build_map_function_batch,
            elems=[batch_gt_boxes],
            fn_output_signature=[tf.float32, tf.float32, tf.int32, tf.int32],
        )
        return outputs

    def get_config(self):
        c = super(Target_v1, self).get_config()
        c.update({
            'num_cls': self.num_cls,
            'strides': self.strides,
        })
        return c


def freeze_model_bn(model_):
    for layer in model_.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def _build_backbone_v2(name='resnet', depth=50):
    backbone = None
    outputs = None

    if name == 'resnet':
        output_layers = ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]

        if depth == 50:
            from tensorflow.keras.applications import ResNet50
            backbone = ResNet50(include_top=False, input_shape=[None, None, 3])

            if config.FROZEN_BACKBONE_STAGE:
                print("[INFO] Frozen Stage 0 and Stage 1 from Backbone ...")
                th = False
                for layer in backbone.layers:
                    if layer.name == "conv3_block1_1_conv":
                        th = True
                    if th:
                        layer.trainable = True
                    else:
                        layer.trainable = False
            if config.FROZEN_BACKBONE_BN:
                print("[INFO] Frozen BatchNormalization Layer ...")
                freeze_model_bn(backbone)

            outputs = [backbone.get_layer(layer_name).output for layer_name in output_layers]

    else:
        raise ValueError('Wrong Backbone Name !')

    return keras.Model(
        inputs=[backbone.inputs], outputs=outputs, name=f'{name}-{depth}'
    )


def _build_head_subnets(input_features, width, depth, num_cls):
    cls_pred, reg_pred, cen_pred = [], [], []

    _setting = {
        'input_features': input_features,
        'width': width,
        'depth': depth,
        'num_cls': num_cls,
    }

    if config.HEAD == 'Mix':
        from models.head import MixSubnetworks
        cls_pred, reg_pred = MixSubnetworks(**_setting)

    elif config.HEAD == 'Std':
        # from models.head.std_head import Subnetworks
        # head = Subnetworks(width, depth, num_cls)
        # cls_pred, reg_pred = head(input_features)

        from models.head.std_head import Subnetworks
        head = Subnetworks(width, depth, num_cls)
        cls_pred, reg_pred, cen_pred = head(input_features)
        return cls_pred, reg_pred, cen_pred

    elif config.HEAD == 'MP':
        from models.head.mp_filter import Subnetworks
        head = Subnetworks(width, depth, num_cls)
        cls_pred, reg_pred = head(input_features)

        return cls_pred, reg_pred


class centerness_rebuild(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(centerness_rebuild, self).__init__(dtype=tf.float32, **kwargs)

    def call(self, inputs, *args, **kwargs):
        mask = inputs[0]
        pred_cen = inputs[1]

        out = tf.where(
            tf.greater(mask, 0.),
            pred_cen,
            1.
        )

        return out[..., None]

    def get_config(self):
        cfg = super(centerness_rebuild, self).get_config()
        return cfg


def detector(
        num_cls=20,
        width=256,
        depth=4,
):
    if config.DB_MODE == "keras":
        images = keras.layers.Input((None, None, 3), name='image')
        gt_boxes_input = keras.layers.Input((100, 5), name='bboxes')
        feature_maps_shape_input = keras.layers.Input((5, 2), dtype=tf.int32, name='fmaps_shape')
        cls_tar, loc_tar, ind_tar, bboxes_cnt = Target_v1()([feature_maps_shape_input, gt_boxes_input])
        inputs = [images, gt_boxes_input, feature_maps_shape_input]

    else:
        images = keras.layers.Input((None, None, 3), name='image')
        cls_tar = keras.layers.Input((None, num_cls + 2), name='cls_target')
        loc_tar = keras.layers.Input((None, 4 + 2), name='loc_target')
        ind_tar = keras.layers.Input((None, 1), name='ind_target')
        bboxes_cnt = keras.layers.Input((1,), name='bboxes_cnt')
        mask_tar = keras.layers.Input((None, FMAP_AREA_SUM, 1), name="mask_target")
        inputs = [images, cls_tar, loc_tar, ind_tar, bboxes_cnt, mask_tar]

    """ Backbone """
    backbone = _build_backbone_v2(name='resnet', depth=50)
    c3, c4, c5 = backbone(images)

    """ Feature Pyramid Network """
    fpn = FeaturePyramidNetwork()
    features = fpn([c3, c4, c5])

    """ Subnetworks """
    # (None, 8525, num_cls), (None, 8525, 4), (None, 8525, 1)
    cls_out, loc_out, cen_out = _build_head_subnets(
        input_features=features,
        width=width,
        depth=depth,
        num_cls=num_cls
    )

    dwlm_out, dc_mask = DWLMLayer()(
        [
            cls_out, loc_out,
            cls_tar, loc_tar,
            ind_tar, bboxes_cnt,
            mask_tar
        ]
    )

    """ Focal & IoU Loss """
    focal_loss = FocalLoss(test=True, name='cls_loss')(
        [cls_tar, cls_out, dwlm_out, cen_out]
    )
    iou_loss = IoULoss(test=True, mode=config.IOU_LOSS, name='loc_loss')(
        [loc_tar, loc_out, dwlm_out, cen_out]
    )
    cen_loss = CENLoss(name='cen_loss')(
        [cls_tar[..., -2], dc_mask, cen_out]
    )

    """ Training model """
    train_model = keras.models.Model(
        # inputs=[
        #     images, cls_tar, loc_tar,
        #     ind_tar, bboxes_cnt
        # ],
        inputs=inputs,
        outputs=[cls_out, loc_out, dwlm_out, focal_loss, iou_loss, cen_loss],
        name='training_model'
    )

    """ Inference """
    locs, strides = Locations2()(features)
    boxes = RegressionBoxes2(name='boxes')([locs, strides, loc_out])
    boxes = ClipBoxes2(name='clip_boxes')([images, boxes])
    detections = FilterDetections2(
        nms=1 if config.NMS == 1 else 0,
        s_nms=1 if config.NMS == 2 else 0,
        nms_threshold=config.NMS_TH,
        name='filtered_detections',
        score_threshold=config.SCORE_TH,
        max_detections=config.DETECTIONS,
    )([boxes, cls_out])

    prediction_model = keras.models.Model(
        inputs=[images],
        outputs=detections,
        name='inference_model'
    )

    """ Training and Inference """
    return train_model, prediction_model


if __name__ == '__main__':
    _test, _ = detector()
    _test.summary()
