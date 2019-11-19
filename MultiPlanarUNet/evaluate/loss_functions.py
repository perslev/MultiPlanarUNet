import tensorflow as tf
from MultiPlanarUNet.logging import ScreenLogger
from MultiPlanarUNet.utils import print_options_context
from tensorflow.python.ops import array_ops


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


def sparse_jaccard_distance_loss(y_true, y_pred, smooth=1):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    # Output shape
    n_classes = y_pred.get_shape()[-1].value

    # One hot encode set shape
    y_true = tf.one_hot(tf.cast(y_true, tf.uint8), depth=n_classes)
    y_true = array_ops.reshape(y_true, [-1, n_classes])
    y_pred = array_ops.reshape(y_pred, [-1, n_classes])

    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    sum_ = tf.reduce_sum(y_true + y_pred, axis=0)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return 1.0 - tf.reduce_mean(jac)


def sparse_dice_loss(y_true, y_pred, smooth=1):
    # Output shape
    n_classes = y_pred.get_shape()[-1]

    # One hot encode set shape
    y_true = tf.one_hot(tf.cast(y_true, tf.uint8), depth=n_classes)
    y_true = array_ops.reshape(y_true, [-1, n_classes])
    y_pred = array_ops.reshape(y_pred, [-1, n_classes])

    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice)


def sparse_dice_ce_loss(y_true, y_pred, smooth=1, dice_weight=1, ce_weight=1):
    from tensorflow._api.v1.keras.losses import sparse_categorical_crossentropy
    dice_loss = sparse_dice_loss(y_true, y_pred, smooth)
    ce_loss = tf.reduce_mean(sparse_categorical_crossentropy(y_true, y_pred))
    return dice_weight * dice_loss + ce_weight * ce_loss


class ExponentialLogarithmicLoss(object):
    """
    https://link.springer.com/content/pdf/10.1007%2F978-3-030-00931-1_70.pdf
    """
    def __init__(self, n_classes, gamma_dice=0.3, gamma_cross=0.3,
                 weight_dice=1, weight_cross=1, logger=None,
                 int_targets=True, **kwargs):
        self.gamma_dice = gamma_dice
        self.gamma_cross = gamma_cross
        self.weight_dice = weight_dice
        self.weight_cross = weight_cross

        self.n_classes = n_classes
        self.int_targets = int_targets
        self.logger = logger or ScreenLogger()
        self.__name__ = "ExponentialLogarithmicLoss"
        self.log()

    def log(self):
        self.logger("Exponential Logarithmic Loss")
        self.logger("N classes:     ", self.n_classes)
        self.logger("Gamma dice:    ", self.gamma_dice)
        self.logger("Gamme cross:   ", self.gamma_cross)

    def __call__(self, y_true, y_pred):
        # Output shape
        output_shape = y_pred.get_shape()
        out_shape_tensor = array_ops.shape(y_pred)

        if self.int_targets:
            # One hot encode set shape
            y_true = tf.one_hot(tf.cast(y_true, tf.uint8), depth=self.n_classes)
            y_true = array_ops.reshape(y_true, out_shape_tensor)

        # Cast if needed
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip for numerical stability
        _epsilon = _to_tensor(10e-8, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)

        # Compute exp log dice
        intersect = 2 * tf.reduce_sum(y_true * y_pred, axis=0, keepdims=True) + 1
        union = tf.reduce_sum(y_true + y_pred, axis=0, keepdims=True) + 1
        exp_log_dice = tf.pow(-tf.log(intersect/union), self.gamma_dice)
        mean_exp_log_dice = tf.reduce_mean(exp_log_dice, axis=-1, keepdims=True)

        # Compute exp cross entropy
        entropy = tf.reduce_sum(y_true * -tf.log(y_pred), axis=-1, keepdims=True)
        exp_entropy = tf.pow(entropy, self.gamma_cross)

        # Compute output
        res = self.weight_dice*mean_exp_log_dice + self.weight_cross*exp_entropy

        if len(output_shape) >= 3:
            # If our output includes timesteps or spatial dimensions we need to reshape
            return tf.reshape(res, out_shape_tensor[:-1])
        else:
            return res


class FocalLoss(object):
    """

    """
    def __init__(self, n_classes, class_weights=None, gamma=2,
                 int_targets=True, logger=None, **kwargs):

        self.logger = logger or ScreenLogger()
        self.__name__ = "FocalLoss"
        self.n_classes = n_classes

        if not class_weights:
            class_weights = [1]*self.n_classes

        self.weights = tf.constant(class_weights, tf.float32, [self.n_classes])
        if n_classes != len(class_weights):
            raise ValueError("The number of classes (%i) does not match"
                             " the length of the class weights array "
                             "(len=%i)" % (
                             self.n_classes, len(class_weights)))

        self.gamma = gamma
        self.int_targets = int_targets
        self.log()

    def __call__(self, y_true, y_pred):
        # Output shape
        output_shape = y_pred.get_shape()
        out_shape_tensor = array_ops.shape(y_pred)

        if self.int_targets:
            # One hot encode set shape
            y_true = tf.one_hot(tf.cast(y_true, tf.uint8), depth=self.n_classes)
            y_true = array_ops.reshape(y_true, out_shape_tensor)

        # Cast if needed
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip for numerical stability
        _epsilon = _to_tensor(10e-8, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)

        # Compute the focal loss
        entropy = tf.log(y_pred)
        modulator = tf.pow((1-y_pred), self.gamma)
        loss = -tf.reduce_sum(self.weights * y_true * modulator * entropy,
                              axis=-1, keepdims=True)

        if len(output_shape) >= 3:
            # If our output includes timesteps or spatial dimensions we need to reshape
            return tf.reshape(loss, out_shape_tensor[:-1])
        else:
            return loss

    def log(self):
        self.logger("Focal Loss")
        self.logger("N classes:   ", self.n_classes)
        self.logger("Gamma:       ", self.gamma)
        self.logger("Weights:     ", self.weights)


class GeneralizedDiceLoss(object):
    """
    Based on implementation in NiftyNet at:

    http://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/
    loss_segmentation.html#generalised_dice_loss

    Class based to allow passing of parameters to the function at construction
    time in keras.
    """
    def __init__(self, n_classes, type_weight="Square", logger=None, **kwargs):
        self.type_weight = type_weight
        self.n_classes = n_classes
        self.logger = logger or ScreenLogger()
        self.__name__ = "GeneralizedDiceLoss"
        self.log()

    def log(self):
        self.logger("Generalized Dice Loss")
        self.logger("N classes:   ", self.n_classes)
        self.logger("Weight type: ", self.type_weight)

    def __call__(self, y_true, y_pred):
        """
        Function to calculate the Generalised Dice Loss defined in
            Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
            loss function for highly unbalanced segmentations. DLMIA 2017

        :param y_true: the segmentation ground truth, one-hot encoded
        :param y_pred: softmax distribution over classes, same shape a y_true
        :return: the loss
        """
        one_hot = tf.one_hot(tf.cast(y_true, tf.uint8), depth=self.n_classes)

        # Reshape to [None, n_classes] tensors
        prediction = tf.reshape(y_pred, [-1, self.n_classes])
        one_hot = tf.reshape(one_hot, [-1, self.n_classes])

        ref_vol = tf.reduce_sum(one_hot, axis=0, keepdims=True)
        intersect = tf.reduce_sum(one_hot * prediction, axis=0, keepdims=True)
        seg_vol = tf.reduce_sum(prediction, axis=0, keepdims=True)

        if self.type_weight.lower() == 'square':
            weights = tf.reciprocal(tf.square(ref_vol))
        elif self.type_weight.lower() == 'simple':
            weights = tf.reciprocal(ref_vol)
        elif self.type_weight.lower() == 'uniform':
            weights = tf.ones_like(ref_vol)
        else:
            raise ValueError("The variable type_weight \"{}\""
                             "is not defined.".format(self.type_weight))

        # Make array of new weight in which infinite values are replaced by
        # ones.
        new_weights = tf.where(tf.is_inf(weights),
                               tf.zeros_like(weights),
                               weights)

        # Set final weights as either original weights or highest observed
        # non-infinite weight
        weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                           tf.reduce_max(new_weights), weights)

        # calculate generalized dice score
        eps = 1e-6
        numerator = 2 * tf.reduce_sum(tf.multiply(weights, intersect))
        denom = tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + eps
        generalised_dice_score = numerator / denom
        return 1 - generalised_dice_score


class WeightedSemanticCCE(object):
    def __init__(self, class_weights, logger=None, *args, **kwargs):
        """
        weights: A Nx1 matrix of class weights for N classes.
        """

        if class_weights is None or class_weights is False:
            raise ValueError("No class weights passed.")
        self.logger = logger if logger is not None else ScreenLogger()
        self.__name__ = "WeightedSemanticCCE"

        self.weights = _to_tensor(class_weights, dtype=tf.float32)
        self.n_classes = self.weights.get_shape()[0]

        with print_options_context(precision=3, suppress=True):
            logger("Class weights:\n%s" % class_weights)

        # Log to console/file
        self._log()

    def __call__(self, target, output, from_logits=False, *args, **kwargs):
        """
        Weighted categorical crossentropy between an output tensor and a
        target tensor.

        # Arguments
            target: A tensor rank(output)-1 with integer targets.
            output: A tensor resulting from a softmax
                (unless `from_logits` is True, in which
                case `output` is expected to be the logits).
            from_logits: Boolean, whether `output` is the
                result of a softmax, or is a tensor of logits.

        # Returns
            Output tensor.
        """
        # Note: tf.nn.softmax_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.

        """
        Note: the following code is adapted from tensorflows cross_entropy
        implementation
        """
        if not from_logits:
            # scale preds so that the class probas of each sample sum to 1
            output /= tf.reduce_sum(output,
                                    axis=len(output.get_shape()) - 1,
                                    keepdims=True)
            # manual computation of crossentropy
            _epsilon = _to_tensor(10e-8, output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)

            # Calculate weighted loss
            ce = target * tf.log(output)
            wce = tf.multiply(ce, self.weights)
            loss = -tf.reduce_mean(tf.reduce_sum(wce, axis=len(output.get_shape()) - 1))
            return loss
        else:
            raise NotImplementedError

    def _log(self):
        self.logger("OBS: Using weighted semantic cross entropy")
        self.logger("N classes: %s" % self.n_classes)
        self.logger("Weights  : %s" % self.weights)


class BatchWeightedCrossEntropyWithLogits(object):
    def __init__(self, n_classes, logger=None, **kwargs):
        self.logger = logger if logger is not None else ScreenLogger()
        self.__name__ = "BatchWeightedCrossEntropyWithLogits"

        # if not class_weights
        self.n_classes = n_classes

        self._log()

    def __call__(self, target, output, *args, **kwargs):
        # Flatten
        target = tf.cast(tf.reshape(target, [-1]), tf.int32)
        output = tf.reshape(output, [-1, self.n_classes])

        # Calculate in-batch class counts and total counts
        target_one_hot = tf.one_hot(target, self.n_classes)
        counts = tf.cast(tf.reduce_sum(target_one_hot, axis=0), tf.float32)
        total_counts = tf.reduce_sum(counts)

        # Compute balanced sample weights
        weights = (tf.ones_like(counts) * total_counts) / (counts * self.n_classes)

        # Compute sample weights from class weights
        weights = tf.gather(weights, target)

        return tf.losses.sparse_softmax_cross_entropy(target, output, weights)

    def _log(self):
        self.logger("OBS: Using weighted cross entropy (logit targets)")
        self.logger("N classes: %s" % self.n_classes)


class OneHotLossWrapper(object):
    def __init__(self, loss_func, n_classes):
        self.loss_func = loss_func
        self.n_classes = n_classes

    def __str__(self):
        return "OneHotLossWrapper(" + self.loss_func.__name__ + ")"

    def __call__(self, target, output, *args, **kwargs):
        target.set_shape(output.shape[:-1].concatenate([1]))
        target = tf.cast(target, tf.int32)
        target = tf.reshape(tf.one_hot(target, self.n_classes, axis=-1,
                                       dtype=tf.float32), tf.shape(output))

        return self.loss_func(target, output, *args, **kwargs)


# Aliases
ExpLogDice = ExponentialLogarithmicLoss
