import config
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
from utils.util_graph import shrink_and_normalize_boxes, create_reg_positive_sample

_image_size = [512, 640, 768, 896, 1024, 1280, 1408]
_STRIDES = [8, 16, 32, 64, 128]
_ALPHA = 0.0


@tf.function
def _normalization_image(image, mode):
    if mode == 'ResNetV1':
        # Caffe
        image = image[..., ::-1]  # RGB -> BGR
        image -= [103.939, 116.779, 123.68]

    elif mode == 'ResNetV2':
        image /= 127.5
        image -= 1.

    elif mode == 'EffNet':
        image = image

    elif mode in ['DenseNet', 'SEResNet']:
        # Torch
        image /= 255.
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]

    return image


def _fmap_shapes(phi: int = 0, level: int = 5):
    _img_size = int(phi * 128) + 512
    _strides = [int(2 ** (x + 3)) for x in range(level)]

    shapes = []

    for i in range(level):
        fmap_shape = _img_size // _strides[i]
        shapes.append([fmap_shape, fmap_shape])

    return shapes


@tf.function
def random_flip_horizontal(image, image_shape, bboxes, prob=0.5):
    """Flips image and boxes horizontally

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      image_shape:
      bboxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.
      prob: Chance.

    Returns:
      Randomly flipped image and boxes
    """

    if tf.random.uniform(()) > (1 - prob):
        image = tf.image.flip_left_right(image)
        bboxes = tf.stack(
            [
                image_shape[1] - bboxes[..., 2] - 1,
                bboxes[..., 1],
                image_shape[1] - bboxes[..., 0] - 1,
                bboxes[..., 3]
            ],
            axis=-1
        )
    return image, bboxes


@tf.function
def random_rotate(image, image_shape, bboxes, prob=0.5):
    offset = image_shape / 2.
    rotate_k = tf.random.uniform((), minval=1, maxval=4, dtype=tf.int32)

    def _r_method(x, y, angle):
        tf_cos = tf.math.cos(angle)
        tf_sin = tf.math.sin(angle)

        tf_abs_cos = tf.abs(tf_cos)
        tf_abs_sin = tf.abs(tf_sin)

        offset_h, offset_w = offset[0], offset[1]

        new_offset_w = offset_w * (tf_abs_cos - tf_cos) + offset_h * (tf_abs_sin - tf_sin)
        new_offset_h = offset_w * (tf_abs_sin + tf_sin) + offset_h * (tf_abs_cos - tf_cos)

        x_r = x * tf_cos + y * tf_sin + new_offset_w
        y_r = x * tf_sin * -1 + y * tf_cos + new_offset_h

        x_r = tf.round(x_r)
        y_r = tf.round(y_r)
        return x_r, y_r

    def _rotate_bbox(bbox):
        # degree: pi/2, pi, 3*pi/2
        angle = tf.cast(rotate_k, dtype=tf.float32) * (np.pi / 2.)

        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        x1_n, y1_n = _r_method(x1, y1, angle)
        x2_n, y2_n = _r_method(x2, y2, angle)

        bbox = tf.stack([
            tf.minimum(x1_n, x2_n),
            tf.minimum(y1_n, y2_n),
            tf.maximum(x1_n, x2_n),
            tf.maximum(y1_n, y2_n)
        ])
        return bbox

    if tf.random.uniform(()) > (1 - prob):
        image = tf.image.rot90(image, k=rotate_k)

        bboxes = tf.map_fn(
            _rotate_bbox,
            elems=bboxes,
            fn_output_signature=tf.float32
        )
        image_shape = tf.cast(tf.shape(image)[:2], tf.float32)
        bboxes = tf.stack(
            [
                tf.clip_by_value(bboxes[:, 0], 0., image_shape[1] - 2),  # x1
                tf.clip_by_value(bboxes[:, 1], 0., image_shape[0] - 2),  # y1
                tf.clip_by_value(bboxes[:, 2], 1., image_shape[1] - 1),  # x2
                tf.clip_by_value(bboxes[:, 3], 1., image_shape[0] - 1),  # y2
                bboxes[:, -1]
            ],
            axis=-1
        )
    return image, image_shape, bboxes


@tf.function
def multi_scale(image, image_shape, bboxes, prob=0.5):
    new_image_shape = image_shape

    if tf.random.uniform(()) > (1 - prob):
        # start, end, step = 0.25, 1.3, 0.05
        # scale = np.random.choice(np.arange(start, end, step))
        scale = tf.random.uniform((), minval=0.8, maxval=1.3)

        new_image_shape = tf.round(image_shape * scale)
        image = tf.image.resize(
            image,
            tf.cast(new_image_shape, tf.int32),
            method=tf.image.ResizeMethod.BILINEAR
        )
        bboxes = tf.stack(
            [
                tf.clip_by_value(bboxes[..., 0] * scale, 0, new_image_shape[1] - 2),
                tf.clip_by_value(bboxes[..., 1] * scale, 0, new_image_shape[0] - 2),
                tf.clip_by_value(bboxes[..., 2] * scale, 1, new_image_shape[1] - 1),
                tf.clip_by_value(bboxes[..., 3] * scale, 1, new_image_shape[0] - 1),
            ],
            axis=-1
        )
        bboxes = tf.round(bboxes)
    return image, new_image_shape, bboxes


@tf.function
def random_crop(image, image_shape, bboxes, prob=0.5):
    if tf.random.uniform(()) > (1 - prob):
        min_x1y1 = tf.cast(tf.math.reduce_min(bboxes, axis=0)[:2], tf.int32)
        max_x2y2 = tf.cast(tf.math.reduce_max(bboxes, axis=0)[2:], tf.int32)
        new_image_shape = tf.cast(image_shape, tf.int32)

        random_x1 = tf.random.uniform((), minval=0, maxval=tf.maximum(min_x1y1[0] // 2, 1), dtype=tf.int32)
        random_y1 = tf.random.uniform((), minval=0, maxval=tf.maximum(min_x1y1[1] // 2, 1), dtype=tf.int32)

        random_x2 = tf.random.uniform(
            (),
            minval=max_x2y2[0] + 1,
            maxval=tf.math.maximum(
                tf.math.minimum(new_image_shape[1], max_x2y2[0] + (new_image_shape[1] - max_x2y2[0]) // 2),
                max_x2y2[0] + 2
            ),
            dtype=tf.int32
        )
        random_y2 = tf.random.uniform(
            (),
            minval=max_x2y2[1] + 1,
            maxval=tf.math.maximum(
                tf.math.minimum(new_image_shape[0], max_x2y2[1] + (new_image_shape[0] - max_x2y2[1]) // 2),
                max_x2y2[1] + 2
            ),
            dtype=tf.int32
        )

        image = tf.image.crop_to_bounding_box(
            image,
            offset_height=random_y1,
            offset_width=random_x1,
            target_height=(random_y2 - random_y1),
            target_width=(random_x2 - random_x1)
        )

        bboxes = tf.stack(
            [
                bboxes[:, 0] - tf.cast(random_x1, tf.float32),
                bboxes[:, 1] - tf.cast(random_y1, tf.float32),
                bboxes[:, 2] - tf.cast(random_x1, tf.float32),
                bboxes[:, 3] - tf.cast(random_y1, tf.float32),
            ],
            axis=-1
        )
        image_shape = tf.cast(tf.shape(image)[:2], tf.float32)

    return image, image_shape, bboxes


def random_image_saturation(image, prob=.5):
    if tf.random.uniform(()) > (1 - prob):
        image = tf.image.random_saturation(image, 1, 5)

    return image


def random_image_brightness(image, prob=.5):
    if tf.random.uniform(()) > (1 - prob):
        image = tf.image.random_brightness(image, 0.8, 1.)

    return image


def random_image_contrast(image, prob=.5):
    if tf.random.uniform(()) > (1 - prob):
        image = tf.image.random_contrast(image, 0.2, 1.)

    return image


@tf.function
def image_color_augmentation(image):
    ids = int(tf.random.uniform((), minval=0, maxval=3))

    if ids == 0:
        image = random_image_saturation(image)

    elif ids == 1:
        image = random_image_brightness(image)

    elif ids == 2:
        image = random_image_contrast(image)

    return image


@tf.function
def _image_transform(image, target_size=512, padding_value=.0):
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]

    if image_height > image_width:
        scale = tf.cast((target_size / image_height), dtype=tf.float32)
        resized_height = target_size
        resized_width = tf.cast((tf.cast(image_width, dtype=tf.float32) * scale), dtype=tf.int32)
    else:
        scale = tf.cast((target_size / image_width), dtype=tf.float32)
        resized_height = tf.cast((tf.cast(image_height, dtype=tf.float32) * scale), dtype=tf.int32)
        resized_width = target_size

    image = tf.image.resize(
        image,
        (resized_height, resized_width),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    offset_h = (target_size - resized_height) // 2
    offset_w = (target_size - resized_width) // 2

    # (h, w, c)
    pad = tf.stack(
        [
            tf.stack([offset_h, target_size - resized_height - offset_h], axis=0),
            tf.stack([offset_w, target_size - resized_width - offset_w], axis=0),
            tf.constant([0, 0]),
        ],
        axis=0
    )

    image = tf.pad(image, pad, constant_values=padding_value)

    return image, scale, [offset_h, offset_w]


@tf.function
def _bboxes_transform(bboxes, classes, scale, offset_hw, max_bboxes=20, padding=False):
    bboxes *= scale
    bboxes = tf.stack(
        [
            (bboxes[:, 0] + tf.cast(offset_hw[1], dtype=tf.float32)),
            (bboxes[:, 1] + tf.cast(offset_hw[0], dtype=tf.float32)),
            (bboxes[:, 2] + tf.cast(offset_hw[1], dtype=tf.float32)),
            (bboxes[:, 3] + tf.cast(offset_hw[0], dtype=tf.float32)),
            classes
        ],
        axis=-1,
    )

    if padding:
        # true_label_count
        bboxes_count = tf.shape(bboxes)[0]
        max_bbox_pad = tf.stack(
            [
                tf.stack([tf.constant(0), max_bboxes - bboxes_count], axis=0),
                tf.constant([0, 0]),
            ],
            axis=0
        )
        bboxes = tf.pad(bboxes, max_bbox_pad, constant_values=0.)

    else:
        bboxes_count = tf.shape(bboxes)[0]

    return bboxes, bboxes_count


@tf.function
def _clip_transformed_bboxes(image, bboxes, debug=False):
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)

    if debug:
        bboxes = tf.stack(
            [
                tf.clip_by_value(bboxes[:, 0] / image_shape[1], 0., 1.),  # x1
                tf.clip_by_value(bboxes[:, 1] / image_shape[0], 0., 1.),  # y1
                tf.clip_by_value(bboxes[:, 2] / image_shape[1], 0., 1.),  # x2
                tf.clip_by_value(bboxes[:, 3] / image_shape[0], 0., 1.),  # y2
                bboxes[:, -1]
            ],
            axis=-1
        )

    else:
        bboxes = tf.stack(
            [
                tf.clip_by_value(bboxes[:, 0], 0., image_shape[1] - 2),  # x1
                tf.clip_by_value(bboxes[:, 1], 0., image_shape[0] - 2),  # y1
                tf.clip_by_value(bboxes[:, 2], 1., image_shape[1] - 1),  # x2
                tf.clip_by_value(bboxes[:, 3], 1., image_shape[0] - 1),  # y2
                bboxes[:, -1]
            ],
            axis=-1
        )
    return bboxes


@tf.function
def compute_inputs(sample):
    image = tf.cast(sample["image"], dtype=tf.float32)
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    bboxes = tf.cast(sample["objects"]["bbox"], dtype=tf.float32)
    classes = tf.cast(sample["objects"]["label"], dtype=tf.float32)

    bboxes = tf.stack(
        [
            bboxes[:, 0] * image_shape[1],
            bboxes[:, 1] * image_shape[0],
            bboxes[:, 2] * image_shape[1],
            bboxes[:, 3] * image_shape[0],
        ],
        axis=-1
    )
    return image, image_shape, bboxes, classes


def preprocess_data_v1(
        phi: int = 0,
        mode: str = "ResNetV1",
        fmap_shapes: any = None,
        max_bboxes: int = 100,
        padding_value: float = 128.,
        debug: bool = False,
):
    """Applies preprocessing step to a single sample

    ref: https://keras.io/examples/vision/retinanet/#preprocessing-data

    """

    def _preprocess_data(sample):
        #
        image, image_shape, bboxes, classes = compute_inputs(sample)

        # Image Shape aug.
        if config.MISC_AUG:
            # image, image_shape, bboxes = multi_scale(image, image_shape, bboxes, prob=0.5)
            # image, image_shape, bboxes = random_rotate(image, image_shape, bboxes, prob=.01)
            image, bboxes = random_flip_horizontal(image, image_shape, bboxes, prob=0.5)
            # image, image_shape, bboxes = random_crop(image, image_shape, bboxes, prob=0.5)

        # Image Color aug.
        if config.VISUAL_AUG:
            image = image_color_augmentation(image)

        # Transforming image and bboxes into fixed-size.
        image, scale, offset_hw = _image_transform(image, _image_size[phi], padding_value)
        image = _normalization_image(image, mode) if not debug else image

        # Clipping bboxes
        bboxes, bboxes_count = _bboxes_transform(bboxes, classes, scale, offset_hw, max_bboxes, padding=False)
        bboxes = _clip_transformed_bboxes(image, bboxes, debug=debug)

        fmaps_shape = tf.constant(fmap_shapes, dtype=tf.int32)
        # return image, bboxes, bboxes_count[None], fmaps_shape
        return image, bboxes, scale, image_shape

    return _preprocess_data


def preprocess_data_v2(
        phi: int = 0,
        mode: str = "ResNetV1",
        fmap_shapes: any = None,
        padding_value: float = 128.,
        debug: bool = False,
):
    """Applies preprocessing step to a single sample

    ref: https://keras.io/examples/vision/retinanet/#preprocessing-data

    """

    def _preprocess_data(sample):
        #
        image, image_shape, bboxes, classes = compute_inputs(sample)

        # Image Shape aug.
        if config.MISC_AUG:
            image, image_shape, bboxes = multi_scale(image, image_shape, bboxes, prob=0.5)
            image, image_shape, bboxes = random_rotate(image, image_shape, bboxes, prob=.5)
            image, bboxes = random_flip_horizontal(image, image_shape, bboxes, prob=0.5)
            image, image_shape, bboxes = random_crop(image, image_shape, bboxes, prob=0.5)

        # Image Color aug.
        if config.VISUAL_AUG:
            image = image_color_augmentation(image)

        # Transforming image and bboxes into fixed-size.
        image, scale, offset_hw = _image_transform(image, _image_size[phi], padding_value)
        image = _normalization_image(image, mode) if not debug else image

        # Clipping bboxes
        bboxes, bboxes_count = _bboxes_transform(bboxes, classes, scale, offset_hw, padding=False)
        bboxes = _clip_transformed_bboxes(image, bboxes, debug=debug)

        fmaps_shape = tf.constant(fmap_shapes, dtype=tf.int32)
        return image, bboxes[:, :4], bboxes[:, -1], fmaps_shape

    return _preprocess_data


@tf.function
def _compute_targets_v1(image, bboxes, classes, fmap_shapes):
    num_cls = config.NUM_CLS

    cls_target_ = tf.zeros((0, num_cls + 2), dtype=tf.float32)
    reg_target_ = tf.zeros((0, 4 + 2), dtype=tf.float32)
    ind_target_ = tf.zeros((0, 1), dtype=tf.int32)

    classes = tf.cast(classes, tf.int32)

    for level in range(len(_STRIDES)):
        stride = _STRIDES[level]

        fh = fmap_shapes[level][0]
        fw = fmap_shapes[level][1]

        pos_x1, pos_y1, pos_x2, pos_y2 = shrink_and_normalize_boxes(bboxes, fh, fw, stride, config.SHRINK_RATIO)

        def build_map_function_target(args):
            pos_x1_ = args[0]
            pos_y1_ = args[1]
            pos_x2_ = args[2]
            pos_y2_ = args[3]
            box = args[4]
            cls = args[5]

            """ Create Negative sample """
            neg_top_bot = tf.stack((pos_y1_, fh - pos_y2_), axis=0)
            neg_lef_rit = tf.stack((pos_x1_, fw - pos_x2_), axis=0)
            neg_pad = tf.stack([neg_top_bot, neg_lef_rit], axis=0)

            """ Regression Target: create positive sample """
            _loc_target, _ap_weight, _area = create_reg_positive_sample(
                box, pos_x1_, pos_y1_, pos_x2_, pos_y2_, stride
            )

            """ Classification Target: create positive sample """
            _cls_target = tf.zeros((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, num_cls), dtype=tf.float32) + (
                    _ALPHA / config.NUM_CLS)
            _cls_onehot = tf.ones((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, 1), dtype=tf.float32) * (1 - _ALPHA)
            _cls_target = tf.concat((_cls_target[..., :cls], _cls_onehot, _cls_target[..., cls + 1:]), axis=-1)

            """ Padding Classification Target's negative sample """
            _cls_target = tf.pad(
                _cls_target,
                tf.concat((neg_pad, tf.constant([[0, 0]])), axis=0),
            )

            """ Padding Soft Anchor's negative sample """
            _ap_weight = tf.pad(_ap_weight, neg_pad, constant_values=1)

            """ Creating Positive Sample locations and padding it's negative sample """
            _pos_mask = tf.ones((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_))
            _pos_mask = tf.pad(_pos_mask, neg_pad)

            """ Padding Regression Target's negative sample """
            _loc_target = tf.pad(_loc_target, tf.concat((neg_pad, tf.constant([[0, 0]])), axis=0))

            """ Output Target """
            # shape = (fh, fw, cls_num + 2)
            _cls_target = tf.concat([_cls_target, _ap_weight[..., None], _pos_mask[..., None]], axis=-1)
            # shape = (fh, fw, 4 + 2)
            _loc_target = tf.concat([_loc_target, _ap_weight[..., None], _pos_mask[..., None]], axis=-1)
            # (fh, fw)
            _area = tf.pad(_area, neg_pad, constant_values=1e7)

            return _cls_target, _loc_target, _area

        # cls_target : shape = (objects, fh, fw, cls_num + 2)
        # reg_target : shape = (objects, fh, fw, 4 + 2)
        # area : shape = (objects, fh, fw)
        level_cls_target, level_reg_target, level_area = tf.map_fn(
            build_map_function_target,
            elems=[pos_x1, pos_y1, pos_x2, pos_y2, bboxes, classes],
            fn_output_signature=(tf.float32, tf.float32, tf.float32),
        )
        # min area : shape = (objects, fh, fw) --> (fh, fw)
        level_min_area_indices = tf.argmin(level_area, axis=0, output_type=tf.int32)
        # (fh, fw) --> (fh * fw)
        level_min_area_indices = tf.reshape(level_min_area_indices, (-1,))

        # (fw, ), (fh, )
        locs_x, locs_y = tf.range(0, fw), tf.range(0, fh)

        # (fh, fw) --> (fh * fw)
        locs_xx, locs_yy = tf.meshgrid(locs_x, locs_y)
        locs_xx = tf.reshape(locs_xx, (-1,))
        locs_yy = tf.reshape(locs_yy, (-1,))

        # (fh * fw, 3)
        level_indices = tf.stack((level_min_area_indices, locs_yy, locs_xx), axis=-1)

        """ Select """
        level_cls_target = tf.gather_nd(level_cls_target, level_indices)
        level_reg_target = tf.gather_nd(level_reg_target, level_indices)
        level_min_area_indices = tf.expand_dims(
            tf.where(tf.equal(level_cls_target[..., -1], 1.), level_min_area_indices, -1),
            axis=-1)

        cls_target_ = tf.concat([cls_target_, level_cls_target], axis=0)
        reg_target_ = tf.concat([reg_target_, level_reg_target], axis=0)
        ind_target_ = tf.concat([ind_target_, level_min_area_indices], axis=0)
        # ind_target_ = tf.concat([ind_target_, tf.expand_dims(level_min_area_indices, -1)], axis=0)

    # ind_target_ = tf.where(tf.equal(cls_target_[..., -1], 1.), ind_target_[..., 0], -1)[..., None]
    # Shape: (anchor-points, cls_num + 2), (anchor-points, 4 + 2)
    return image, cls_target_, reg_target_, ind_target_, tf.shape(bboxes)[0][..., None]


@tf.function
def _compute_targets_v2(image, bboxes, classes, fmap_shapes):
    num_cls = config.NUM_CLS

    cls_target_ = tf.zeros((0, num_cls + 2), dtype=tf.float32)
    reg_target_ = tf.zeros((0, 4 + 2), dtype=tf.float32)
    ind_target_ = tf.zeros((0, 1), dtype=tf.int32)
    mk_target_ = tf.zeros((tf.shape(bboxes)[0], 0, 1), dtype=tf.float32)

    classes = tf.cast(classes, tf.int32)

    for level in range(len(_STRIDES)):
        stride = _STRIDES[level]

        fh = fmap_shapes[level][0]
        fw = fmap_shapes[level][1]

        pos_x1, pos_y1, pos_x2, pos_y2 = shrink_and_normalize_boxes(bboxes, fh, fw, stride, config.SHRINK_RATIO)

        def build_map_function_target(args):
            pos_x1_ = args[0]
            pos_y1_ = args[1]
            pos_x2_ = args[2]
            pos_y2_ = args[3]
            box = args[4]
            cls = args[5]

            """ Create Negative sample """
            neg_top_bot = tf.stack((pos_y1_, fh - pos_y2_), axis=0)
            neg_lef_rit = tf.stack((pos_x1_, fw - pos_x2_), axis=0)
            neg_pad = tf.stack([neg_top_bot, neg_lef_rit], axis=0)

            """ Regression Target: create positive sample """
            _loc_target, _ap_weight, _area = create_reg_positive_sample(
                box, pos_x1_, pos_y1_, pos_x2_, pos_y2_, stride
            )

            """ Classification Target: create positive sample """
            _cls_target = tf.zeros((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, num_cls), dtype=tf.float32) + (
                    _ALPHA / config.NUM_CLS)
            _cls_onehot = tf.ones((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_, 1), dtype=tf.float32) * (1 - _ALPHA)
            _cls_target = tf.concat((_cls_target[..., :cls], _cls_onehot, _cls_target[..., cls + 1:]), axis=-1)

            """ Padding Classification Target's negative sample """
            _cls_target = tf.pad(
                _cls_target,
                tf.concat((neg_pad, tf.constant([[0, 0]])), axis=0),
            )

            """ Padding Soft Anchor's negative sample """
            _ap_weight = tf.pad(_ap_weight, neg_pad, constant_values=1)

            """ Creating Positive Sample locations and padding it's negative sample """
            _pos_mask = tf.ones((pos_y2_ - pos_y1_, pos_x2_ - pos_x1_))
            _pos_mask = tf.pad(_pos_mask, neg_pad)

            """ Padding Regression Target's negative sample """
            _loc_target = tf.pad(_loc_target, tf.concat((neg_pad, tf.constant([[0, 0]])), axis=0))

            """ Output Target """
            # shape = (fh, fw, cls_num + 2)
            _cls_target = tf.concat([_cls_target, _ap_weight[..., None], _pos_mask[..., None]], axis=-1)
            # shape = (fh, fw, 4 + 2)
            _loc_target = tf.concat([_loc_target, _ap_weight[..., None], _pos_mask[..., None]], axis=-1)
            # (fh, fw)
            _area = tf.pad(_area, neg_pad, constant_values=1e7)

            return _cls_target, _loc_target, _area

        # cls_target : shape = (objects, fh, fw, cls_num + 2)
        # reg_target : shape = (objects, fh, fw, 4 + 2)
        # area : shape = (objects, fh, fw)
        level_cls_target, level_reg_target, level_area = tf.map_fn(
            build_map_function_target,
            elems=[pos_x1, pos_y1, pos_x2, pos_y2, bboxes, classes],
            fn_output_signature=(tf.float32, tf.float32, tf.float32),
        )
        objects_mask = tf.reshape(level_cls_target[..., -1], (tf.shape(level_cls_target)[0], -1, 1))

        # min area : shape = (objects, fh, fw) --> (fh, fw)
        level_min_area_indices = tf.argmin(level_area, axis=0, output_type=tf.int32)
        # (fh, fw) --> (fh * fw)
        level_min_area_indices = tf.reshape(level_min_area_indices, (-1,))

        # (fw, ), (fh, )
        locs_x, locs_y = tf.range(0, fw), tf.range(0, fh)

        # (fh, fw) --> (fh * fw)
        locs_xx, locs_yy = tf.meshgrid(locs_x, locs_y)
        locs_xx = tf.reshape(locs_xx, (-1,))
        locs_yy = tf.reshape(locs_yy, (-1,))

        # (fh * fw, 3)
        level_indices = tf.stack((level_min_area_indices, locs_yy, locs_xx), axis=-1)

        """ Select """
        level_cls_target = tf.gather_nd(level_cls_target, level_indices)
        level_reg_target = tf.gather_nd(level_reg_target, level_indices)
        level_min_area_indices = tf.expand_dims(
            tf.where(tf.equal(level_cls_target[..., -1], 1.), level_min_area_indices, -1),
            axis=-1)

        cls_target_ = tf.concat([cls_target_, level_cls_target], axis=0)
        reg_target_ = tf.concat([reg_target_, level_reg_target], axis=0)
        ind_target_ = tf.concat([ind_target_, level_min_area_indices], axis=0)
        mk_target_ = tf.concat([mk_target_, objects_mask], axis=1)

    # Shape: (anchor-points, cls_num + 2), (anchor-points, 4 + 2)
    return image, cls_target_, reg_target_, ind_target_, tf.shape(bboxes)[0][..., None], mk_target_


def inputs_targets_v1(image, bboxes, bboxes_count, fmaps_shape):
    inputs = {
        "image": image,
        "bboxes": bboxes,
        "bboxes_count": bboxes_count,
        "fmaps_shape": fmaps_shape,
    }
    return inputs


def inputs_targets_v2(image, cls_target, reg_target, ind_target, bboxes_cnt):
    inputs = {
        "image": image,
        "cls_target": cls_target,
        "loc_target": reg_target,
        "ind_target": ind_target,
        "bboxes_cnt": bboxes_cnt
    }
    return inputs


def inputs_targets_v3(image, cls_target, reg_target, ind_target, bboxes_cnt, mask_target, ):
    inputs = {
        "image": image,
        "cls_target": cls_target,
        "loc_target": reg_target,
        "ind_target": ind_target,
        "bboxes_cnt": bboxes_cnt,
        "mask_target": mask_target,
    }
    return inputs


def _load_data_from_tfrecord(ds_name, path="D:/datasets/"):
    if ds_name == "DPCB":
        (train, test), ds_info = tfds.load(name="dpcb_db",
                                           split=["train", "test"],
                                           data_dir=path,
                                           with_info=True)
    elif ds_name == "VOC":
        (train, test), ds_info = tfds.load(name="pascal_voc",
                                           split=["train", "test"],
                                           data_dir=path,
                                           with_info=True,
                                           shuffle_files=True)
    elif ds_name == "VOC_mini":
        (train, test), ds_info = tfds.load(name="pascal_voc_mini",
                                           split=["train", "test"],
                                           data_dir=path,
                                           with_info=True,
                                           shuffle_files=True)
    else:
        train, test, ds_info = None, None, None

    return train, test, ds_info.splits["train"].num_examples, ds_info.splits["test"].num_examples


def create_pipeline_v1(phi=0, mode="ResNetV1", db="DPCB", batch_size=1):
    autotune = tf.data.AUTOTUNE

    train, test, train_num, test_num = _load_data_from_tfrecord(db)

    # if db == "DPCB":
    #     (train, test) = tfds.load(name="dpcb_db", split=["train", "test"], data_dir="C:/works/datasets/")
    # else:
    #     train = None
    #     test = None

    train = train.map(preprocess_data_v1(phi=phi, mode=mode, fmap_shapes=_fmap_shapes(phi)),
                      num_parallel_calls=autotune)
    train = train.shuffle(train_num)
    train = train.padded_batch(batch_size=batch_size, padding_values=(0.0, 0.0, 0, 0), drop_remainder=True)
    train = train.map(inputs_targets_v1, num_parallel_calls=autotune)
    train = train.repeat().prefetch(autotune)
    return train, test


def create_pipeline_v2(phi=0, mode="ResNetV1", db="DPCB", batch_size=1, debug=False):
    autotune = tf.data.AUTOTUNE
    _buffer = 1000

    if db == "DPCB":
        (train, test), ds_info = tfds.load(name="dpcb_db", split=["train", "test"], data_dir="D:/datasets/",
                                           with_info=True)
    elif db == "VOC":
        (train, test), ds_info = tfds.load(name="pascal_voc", split=["train", "test"], data_dir="D:/datasets/",
                                           with_info=True,
                                           shuffle_files=True)
    elif db == "VOC_mini":
        (train, test), ds_info = tfds.load(name="pascal_voc_mini", split=["train", "test"], data_dir="D:/datasets/",
                                           with_info=True,
                                           shuffle_files=True)
    else:
        train, test, ds_info = None, None, None

    train_examples = ds_info.splits["train"].num_examples
    test_examples = ds_info.splits["test"].num_examples
    print(f"[INFO] {db}: train( {train_examples} ), test( {test_examples} )")

    train = train.map(preprocess_data_v2(
        phi=phi,
        mode=mode,
        fmap_shapes=_fmap_shapes(phi),
        debug=debug
    ), num_parallel_calls=autotune)

    train = (train.shuffle(_buffer, reshuffle_each_iteration=True).repeat())
    train = train.map(_compute_targets_v2, num_parallel_calls=autotune)  # padded tensor.
    # train = train.batch(batch_size=batch_size, drop_remainder=True)   # with _compute_targets_v1
    train = train.padded_batch(
        batch_size=batch_size,
        padding_values=(0., 0., 0., 0, 0, 0.),
        drop_remainder=True)  # with _compute_targets_v2
    train = train.map(inputs_targets_v3, num_parallel_calls=autotune)
    train = train.prefetch(autotune)

    return train, test


def create_pipeline_test(phi=0, mode="ResNetV1", db="DPCB", batch_size=1, debug=False):
    autotune = tf.data.AUTOTUNE

    if db == "DPCB":
        (train, test) = tfds.load(name="dpcb_db", split=["train", "test"], data_dir="D:/datasets/")

    elif db == "VOC":
        (train, test) = tfds.load(name="pascal_voc", split=["train", "test"], data_dir="D:/datasets/",
                                  shuffle_files=True)

    elif db == "VOC_mini":
        (train, test) = tfds.load(name="pascal_voc_mini", split=["train", "test"], data_dir="D:/datasets/",
                                  shuffle_files=True)

    else:
        train = None
        test = None

    feature_maps_shapes = _fmap_shapes(phi)

    train = train.map(preprocess_data_v1(phi=phi, mode=mode, fmap_shapes=feature_maps_shapes, max_bboxes=100,
                                         debug=debug), num_parallel_calls=autotune)

    train = train.shuffle(1000)
    train = train.padded_batch(batch_size=batch_size, padding_values=(0.0, 0.0, 0., 0.), drop_remainder=True)
    train = train.map(inputs_targets_v1, num_parallel_calls=autotune)
    train = train.prefetch(autotune)
    return train, test


if __name__ == '__main__':
    eps = 10
    bs = 1

    train_t, test_t = create_pipeline_v2(
        phi=config.PHI,
        batch_size=bs,
        # debug=True,
        db="VOC"
    )

    """ """
    # for ep in range(eps):
    #     for step, inputs_batch in enumerate(train_t):
    #         # _cls = inputs_batch['cls_target'].numpy()
    #         # _loc = inputs_batch['loc_target'].numpy()
    #         # _ind = inputs_batch['ind_target'].numpy()
    #         _int = inputs_batch['bboxes_cnt'].numpy()
    #
    #         print(f"Ep: {ep + 1}/{eps} - {step + 1}, Batch: {_int.shape[0]}, {_int[:, 0]}")
    #
    #         if np.min(_int) == 0:
    #             break
    #
    #         # if step > (16551 // bs) - 3:
    #         #     min_cnt = np.min(_int)
    #         #     print(f"Ep: {ep + 1}/{eps} - {step + 1}, Batch: {_int.shape[0]}, {min_cnt}")

    """ """

    # iterations = 1
    # for step, inputs_batch in enumerate(train_t):
    #     # if (step + 1) > iterations:
    #     #     break
    #
    #     print(f"[INFO] {step + 1} / {iterations}")
    #
    #     _cls = inputs_batch['cls_target'].numpy()
    #     _loc = inputs_batch['loc_target'].numpy()
    #     _ind = inputs_batch['ind_target'].numpy()
    #     _int = inputs_batch['bboxes_cnt'].numpy()
    #     _mks = inputs_batch['mask_target'].numpy()
    #
    #     if _int > 15:
    #         break
    #
    # obj_cnt = _int[0, 0]
    # p7_mk = np.reshape(_cls[0, 8500:, -1], (5, 5))
    # p6_mk = np.reshape(_cls[0, 8400:8500, -1], (10, 10))
    # p5_mk = np.reshape(_cls[0, 8000:8400, -1], (20, 20))
    #
    # p7_mk_obj = np.reshape(_mks[0, :, 8500:, 0], (obj_cnt, 5, 5))
    # p6_mk_obj = np.reshape(_mks[0, :, 8400:8500, 0], (obj_cnt, 10, 10))
    # p5_mk_obj = np.reshape(_mks[0, :, 8000:8400, 0], (obj_cnt, 20, 20))
    #
    # p7_ap = np.reshape(_cls[0, 8500:, -2], (5, 5))
    # p6_ap = np.reshape(_cls[0, 8400:8500, -2], (10, 10))
    # p5_ap = np.reshape(_cls[0, 8000:8400, -2], (20, 20))
    #
    # p7_ind = np.reshape(_ind[0, 8500:], (5, 5))
    # p6_ind = np.reshape(_ind[0, 8400:8500], (10, 10))
    # p5_ind = np.reshape(_ind[0, 8000:8400], (20, 20))

    """ """

    # import matplotlib.pyplot as plt
    #
    # iterations = 10
    # print('test')
    # plt.figure(figsize=(10, 8))
    # for step, inputs_batch in enumerate(train_t):
    #     if (step + 1) > iterations:
    #         break
    #
    #     print(f"[INFO] {step + 1} / {iterations}")
    #
    #     _images = inputs_batch['image'].numpy()
    #     _bboxes = inputs_batch['bboxes'].numpy()
    #     _scales = inputs_batch['bboxes_count'].numpy()
    #     _images_shape = inputs_batch['fmaps_shape'].numpy()
    #
    #     _bboxes = tf.stack(
    #         [
    #             _bboxes[..., 1],
    #             _bboxes[..., 0],
    #             _bboxes[..., 3],
    #             _bboxes[..., 2],
    #         ],
    #         axis=-1
    #     )
    #
    #     colors = np.array([[255.0, 0.0, 0.0]])
    #     _images = tf.image.draw_bounding_boxes(
    #         _images,
    #         _bboxes,
    #         colors=colors
    #     )
    #
    #     for i in range(bs):
    #         plt.subplot(2, 2, i + 1)
    #         plt.imshow(_images[i].numpy().astype("uint8"))
    #         # print(bboxes[i])
    #     plt.tight_layout()
    #     plt.pause(1)
    #     # plt.close()

    """ """

    # tfds.benchmark(train_t, batch_size=bs)
    # tfds.benchmark(train_t, batch_size=bs)

    # image : (Batch, None, None, 3)
    # bboxes : (Batch, None, 5)
    # bboxes_count : (Batch, 1)
    # fmaps_shape : (Batch, 5, 2)
