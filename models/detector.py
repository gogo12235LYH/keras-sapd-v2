import config
import tensorflow.keras as keras
from models.neck import FeaturePyramidNetwork
from models.layers import Locations2, RegressionBoxes2, ClipBoxes2, FilterDetections2
from models.losses import FocalLoss, IoULoss, DWLMLayer


def _build_backbone_v2(name='resnet', depth=50):
    backbone = None
    outputs = None

    if name == 'resnet':
        output_layers = ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]

        if depth == 50:
            from tensorflow.keras.applications import ResNet50

            backbone = ResNet50(include_top=False, input_shape=[None, None, 3])
            outputs = [backbone.get_layer(layer_name).output for layer_name in output_layers]

    else:
        raise ValueError('Wrong Backbone Name !')

    return keras.Model(
        inputs=[backbone.inputs], outputs=outputs, name=f'{name}-{depth}'
    )


def _build_head_subnets(input_features, width, depth, num_cls):
    cls_pred, reg_pred = [], []

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
        from models.head.std_head import Subnetworks
        head = Subnetworks(width, depth, num_cls)
        cls_pred, reg_pred = head(input_features)

    return cls_pred, reg_pred


def detector(
        num_cls=20,
        width=256,
        depth=4,
):
    images = keras.layers.Input((None, None, 3), name='image')
    cls_tar = keras.layers.Input((None, num_cls + 2), name='cls_target')
    loc_tar = keras.layers.Input((None, 4 + 2), name='loc_target')
    ind_tar = keras.layers.Input((None, 1), name='ind_target')
    bboxes_cnt = keras.layers.Input((1, ), name='bboxes_cnt')

    """ Backbone """
    backbone = _build_backbone_v2(name='resnet', depth=50)
    c3, c4, c5 = backbone(images)

    """ Feature Pyramid Network """
    fpn = FeaturePyramidNetwork()
    features = fpn([c3, c4, c5])

    """ Subnetworks """
    # (None, 8525, num_cls), (None, 8525, 4)
    cls_out, loc_out = _build_head_subnets(
        input_features=features,
        width=width,
        depth=depth,
        num_cls=num_cls
    )

    dwlm_out = DWLMLayer()(
            [
                cls_out, loc_out,
                cls_tar, loc_tar,
                ind_tar, bboxes_cnt,
            ]
        )

    """ Focal & IoU Loss """
    focal_loss = FocalLoss(test=True, name='cls_loss')(
        [cls_tar, cls_out, dwlm_out]
    )
    iou_loss = IoULoss(test=True, name='loc_loss')(
        [loc_tar, loc_out, dwlm_out]
    )

    """ Training model """
    train_model = keras.models.Model(
        inputs=[
            images, cls_tar, loc_tar,
            ind_tar, bboxes_cnt
        ],
        outputs=[cls_out, loc_out, dwlm_out, focal_loss, iou_loss],
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
