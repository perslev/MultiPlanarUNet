from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, \
                                    MaxPooling2D
import tensorflow as tf

from mpunet.models import UNet


def check_all_same_length(attributes, kwargs, target_length):
    for attr in attributes:
        v = kwargs[attr]
        if not isinstance(v, (tuple, list)):
            v = [v]
            kwargs[attr] = v
        if len(v) != target_length:
            raise ValueError("Must pass a '%s' of length %i (one for "
                             "each task) - got %s" % (attr, target_length, v))


class MultiTaskUNet2D(UNet):
    def __init__(self, task_names, **kwargs):
        """
        """
        if not isinstance(task_names, (tuple, list)):
            raise ValueError("'task_names' must be a list or tuple object.")
        self.n_tasks = len(task_names)
        self.task_IDs = tuple(task_names)
        self._n_classes = None

        # Check that each task received the right number of parameters
        check = ("n_classes", "dim", "out_activation", "n_channels")
        check_all_same_length(check, kwargs, target_length=self.n_tasks)

        # Store encoder layers for shared use
        self.encoder_layers = None
        self.out_layers = None

        # Init base UNet class
        super().__init__(**kwargs)

    def _init_encoder(self, init_filters, kernel_reg=None, name="encoder"):
        self.encoder_layers = {}
        filters = init_filters
        for i in range(self.depth):
            l_name = name + "_L%s" % i
            conv1 = Conv2D(int(filters * self.cf), self.kernel_size,
                           activation=self.activation, padding=self.padding,
                           kernel_regularizer=kernel_reg,
                           name=l_name + "_conv1")
            conv2 = Conv2D(int(filters * self.cf), self.kernel_size,
                           activation=self.activation, padding=self.padding,
                           kernel_regularizer=kernel_reg,
                           name=l_name + "_conv2")
            bn = BatchNormalization(name=l_name + "_BN")
            max_pool = MaxPooling2D(pool_size=(2, 2), name=l_name + "_pool")

            # Add to dict for potential reuse
            layers = {
                "layer%s/conv1" % i: conv1,
                "layer%s/conv2" % i: conv2,
                "layer%s/batch_norm" % i: bn,
                "layer%s/max_pool" % i: max_pool
            }
            self.encoder_layers.update(layers)

            # Update filter count and add bn layer to list for residual conn.
            filters *= 2
        return filters

    def _apply_encoder(self, task_input):
        residual_connections = []
        in_ = task_input
        for i in range(self.depth):
            conv1 = self.encoder_layers["layer%s/conv1" % i](in_)
            conv2 = self.encoder_layers["layer%s/conv2" % i](conv1)
            bn = self.encoder_layers["layer%s/batch_norm" % i](conv2)
            in_ = self.encoder_layers["layer%s/max_pool" % i](bn)
            residual_connections.append(bn)

        return in_, residual_connections

    def init_model(self):
        """
        Build the UNet model with the specified input image shape.
        """
        self.img_shape = tuple([t for t in zip(*self.img_shape)])
        inputs = [Input(shape=s,
                        name="Input_%s" % t) for s, t in zip(self.img_shape,
                                                             self.task_IDs)]

        # Apply regularization if not None or 0
        kr = regularizers.l2(self.l2_reg) if self.l2_reg else None

        """
        Encoding path
        """
        # Init the encoder layers
        filters = self._init_encoder(init_filters=64, kernel_reg=kr)

        out_layers = []
        outputs = []
        zipped = zip(self.task_IDs, inputs, self.n_classes, self.out_activation)
        for task, in_, n_classes, activation in zipped:
            with tf.name_scope("Task_%s" % task):
                with tf.name_scope("encoder"):
                    # Apply the encoder to all inputs
                    in_, res = self._apply_encoder(in_)

                # Bottom (no max-pool)
                with tf.name_scope("bottom"):
                    bn = self._create_bottom(in_, filters, kr, name=task)

                # Up-sampling
                with tf.name_scope("decoder"):
                    bn = self._create_upsample(bn, res, filters, kr, name=task)

                """
                Output modeling layer
                """
                with tf.name_scope("classifier"):
                    out_layer = Conv2D(n_classes, 1,
                                       activation=activation,
                                       name="%s" % task)
                    out = out_layer(bn)
                    out_layers.append(out_layer)
                    outputs.append(out)

        self.out_layers = tuple(out_layers)
        return inputs, outputs

    def log(self):
        self.logger("Multi-Task UNet Model Summary\n"
                    "-----------------------------")
        self.logger("N classes:         %s" % list(self.n_classes))
        self.logger("CF factor:         %.3f" % self.cf**2)
        self.logger("Depth:             %i" % self.depth)
        self.logger("l2 reg:            %s" % self.l2_reg)
        self.logger("Padding:           %s" % self.padding)
        self.logger("Conv activation:   %s" % self.activation)
        self.logger("Out activation:    %s" % list(self.out_activation))
        self.logger("Receptive field:   %s" % self.receptive_field)
        self.logger("N params:          %i" % self.count_params())
        self.logger("N tasks:           %i" % self.n_tasks)
        if self.n_tasks > 1:
            inputs = self.input
            outputs = self.output
        else:
            inputs = [self.input]
            outputs = [self.output]
        for i, (id_, in_, out) in enumerate(zip(self.task_IDs, inputs, outputs)):
            self.logger("\n--- Task %s ---" % id_)
            self.logger("In shape:  %s" % in_.shape)
            self.logger("Out shape: %s\n" % out.shape)
