import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    OBS: Code implemented by Tensorflow

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def _get_shapes_and_one_hot(y_true, y_pred):
    shape = y_pred.get_shape()
    n_classes = shape[-1]
    # Squeeze dim -1 if it is == 1, otherwise leave it
    dims = tf.cond(tf.equal(y_true.shape[-1] or -1, 1), lambda: tf.shape(y_true)[:-1], lambda: tf.shape(y_true))
    y_true = tf.reshape(y_true, dims)
    y_true = tf.one_hot(tf.cast(y_true, tf.uint8), depth=n_classes)
    return y_true, shape, n_classes


def sparse_jaccard_distance_loss(y_true, y_pred, smooth=1):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Approximates the class-wise jaccard distance computed per-batch element
    across spatial image dimensions. Returns the 1 - mean(per_class_distance)
    for each batch element.

    :param y_true:
    :param y_pred:
    :param smooth:
    :return:

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    intersection = tf.reduce_sum(y_true * y_pred, axis=reduction_dims)
    sum_ = tf.reduce_sum(y_true + y_pred, axis=reduction_dims)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return 1.0 - tf.reduce_mean(jac, axis=-1, keepdims=True)


class SparseJaccardDistanceLoss(LossFunctionWrapper):
    """ tf reduction wrapper for sparse_jaccard_distance_loss """
    def __init__(self,
                 reduction,
                 smooth=1,
                 name='sparse_jaccard_distance_loss',
                 **kwargs):
        super(SparseJaccardDistanceLoss, self).__init__(
            sparse_jaccard_distance_loss,
            name=name,
            reduction=reduction,
            smooth=smooth
        )


def sparse_dice_loss(y_true, y_pred, smooth=1):
    """
    Approximates the class-wise dice coefficient computed per-batch element
    across spatial image dimensions. Returns the 1 - mean(per_class_dice) for
    each batch element.

    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    intersection = tf.reduce_sum(y_true * y_pred, axis=reduction_dims)
    union = tf.reduce_sum(y_true + y_pred, axis=reduction_dims)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice, axis=-1, keepdims=True)


class SparseDiceLoss(LossFunctionWrapper):
    """ tf reduction wrapper for sparse_dice_loss """
    def __init__(self,
                 reduction,
                 smooth=1,
                 name='sparse_dice_loss',
                 **kwargs):
        super(SparseDiceLoss, self).__init__(
            sparse_dice_loss,
            name=name,
            reduction=reduction,
            smooth=smooth
        )


def sparse_exponential_logarithmic_loss(y_true, y_pred, gamma_dice,
                                        gamma_cross, weight_dice,
                                        weight_cross):
    """
    TODO

    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    # Clip for numerical stability
    _epsilon = _to_tensor(10e-8, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)

    # Compute exp log dice
    intersect = 2 * tf.reduce_sum(y_true * y_pred, axis=reduction_dims) + 1
    union = tf.reduce_sum(y_true + y_pred, axis=reduction_dims) + 1
    exp_log_dice = tf.math.pow(-tf.math.log(intersect/union), gamma_dice)
    mean_exp_log_dice = tf.reduce_mean(exp_log_dice, axis=-1, keepdims=True)

    # Compute exp cross entropy
    entropy = tf.reduce_sum(y_true * -tf.math.log(y_pred), axis=-1, keepdims=True)
    exp_entropy = tf.reduce_mean(tf.math.pow(entropy, gamma_cross), axis=reduction_dims)

    # Compute output
    res = weight_dice*mean_exp_log_dice + weight_cross*exp_entropy
    return res


class SparseExponentialLogarithmicLoss(LossFunctionWrapper):
    """
    https://link.springer.com/content/pdf/10.1007%2F978-3-030-00931-1_70.pdf
    """
    def __init__(self, reduction, gamma_dice=0.3, gamma_cross=0.3,
                 weight_dice=1, weight_cross=1,
                 name="sparse_exponential_logarithmic_loss"):
        super(SparseExponentialLogarithmicLoss, self).__init__(
            sparse_exponential_logarithmic_loss,
            name=name,
            reduction=reduction,
            gamma_dice=gamma_dice,
            gamma_cross=gamma_cross,
            weight_dice=weight_dice,
            weight_cross=weight_cross
        )


def sparse_focal_loss(y_true, y_pred, gamma, class_weights):
    """
    TODO

    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    # Clip for numerical stability
    _epsilon = _to_tensor(10e-8, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)

    if class_weights is None:
        class_weights = [1] * n_classes

    # Compute the focal loss
    entropy = tf.math.log(y_pred)
    modulator = tf.math.pow((1 - y_pred), gamma)
    loss = -tf.reduce_sum(class_weights * y_true * modulator * entropy, axis=-1, keepdims=True)
    return tf.reduce_mean(loss, axis=reduction_dims)


class SparseFocalLoss(LossFunctionWrapper):
    """
    https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, reduction, gamma=2,
                 class_weights=None, name="sparse_focal_loss"):
        super(SparseFocalLoss, self).__init__(
            sparse_focal_loss,
            name=name,
            reduction=reduction,
            gamma=gamma,
            class_weights=class_weights
        )


def sparse_generalized_dice_loss(y_true, y_pred, type_weight):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    ref_vol = tf.reduce_sum(y_true, axis=reduction_dims)
    intersect = tf.reduce_sum(y_true * y_pred, axis=reduction_dims)
    seg_vol = tf.reduce_sum(y_pred, axis=reduction_dims)

    if type_weight.lower() == 'square':
        weights = tf.math.reciprocal(tf.math.square(ref_vol))
    elif type_weight.lower() == 'simple':
        weights = tf.math.reciprocal(ref_vol)
    elif type_weight.lower() == 'uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))

    # Make array of new weight in which infinite values are replaced by
    # ones.
    new_weights = tf.where(tf.math.is_inf(weights),
                           tf.zeros_like(weights),
                           weights)

    # Set final weights as either original weights or highest observed
    # non-infinite weight
    weights = tf.where(tf.math.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)

    # calculate generalized dice score
    eps = 1e-6
    numerator = 2 * tf.multiply(weights, intersect)
    denom = tf.multiply(weights, seg_vol + ref_vol) + eps
    generalised_dice_score = numerator / denom
    return 1 - tf.reduce_mean(generalised_dice_score, axis=-1, keepdims=True)


class SparseGeneralizedDiceLoss(LossFunctionWrapper):
    """
    Based on implementation in NiftyNet at:

    http://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/
    loss_segmentation.html#generalised_dice_loss

    Class based to allow passing of parameters to the function at construction
    time in keras.
    """
    def __init__(self, reduction, type_weight="Square",
                 name='sparse_generalized_dice_loss'):
        super(SparseGeneralizedDiceLoss, self).__init__(
            sparse_generalized_dice_loss,
            name=name,
            reduction=reduction,
            type_weight=type_weight
        )


# Aliases
SparseExpLogDice = SparseExponentialLogarithmicLoss
