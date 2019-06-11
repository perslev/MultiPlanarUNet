import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.initializers import constant
from MultiPlanarUNet.logging import ScreenLogger
from MultiPlanarUNet.evaluate.loss_functions import GeneralizedDiceLoss


def reg(W):
    r = tf.reduce_sum(tf.square(W)) / tf.cast(tf.size(W), tf.float32)
    return 1e-6 * r


class FusionLayer(Layer):
    def __init__(self, logger=None, activation_func=None, **kwargs):
        self.logger = logger or ScreenLogger()
        self.W, self.b = None, None
        self.activation = activation_func or tf.nn.softmax
        super(FusionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(name='W',
                                 shape=(input_shape[1], input_shape[2]),
                                 initializer=constant(1.0),
                                 trainable=True,
                                 regularizer=reg)
        self.b = self.add_weight(name="b",
                                 shape=(1, input_shape[2]),
                                 initializer=constant(0.0),
                                 trainable=True,
                                 regularizer=reg)

        # Build the layer from the base class
        # This also sets self.built = True
        super(FusionLayer, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        return self.activation(tf.reduce_sum(self.W * x, axis=1) + self.b)

    def compute_output_shape(self, input_shape):
        return None, input_shape[2]


class FusionModel(Model):
    def __init__(self, n_inputs, n_classes, weight="Simple", logger=None,
                 verbose=True):
        self.n_inputs = n_inputs
        self.n_classes = n_classes

        # Set Logger object
        self.logger = logger or ScreenLogger()

        # Set loss
        self.loss = GeneralizedDiceLoss(n_classes, type_weight=weight,
                                        logger=logger, sparse=True)
        # self.loss = tf.keras.losses.categorical_crossentropy

        # Init model
        super().__init__(*self.init_model(n_inputs, n_classes))

        if verbose:
            self._log()

    def init_model(self, n_inputs, n_classes):
        if self.loss.__name__ == "BatchWeightedCrossEntropyWithLogitsAndSparseTargets":
            af = tf.identity
        else:
            af = tf.nn.softmax

        inputs = Input(shape=(n_inputs, n_classes))
        fusion = FusionLayer(activation_func=af)(inputs)

        return [inputs], [fusion]

    def _log(self):
        self.logger("Optimizer:  %s" % self.optimizer)
        self.logger("Loss:       %s" % self.loss)
        self.logger("Input:      %s" % self.input)
        self.logger("Output:     %s" % self.output)
        self.logger("N weights:  %s" % self.count_params())
