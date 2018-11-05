"""
Mathias Perslev
MSc Bioinformatics

University of Copenhagen
November 2017
"""

from MultiViewUNet.logging import ScreenLogger

# Import keras modules
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
                                    Concatenate, Conv2D, MaxPooling2D, \
                                    UpSampling2D
from autofocus import Autofocus2D
import numpy as np


class AutofocusUNet2D(Model):
    """
    2D UNet implementation with batch normalization and complexity factor adj.

    See original paper at http://arxiv.org/abs/1505.04597
    """
    def __init__(self, n_classes, img_rows=None, img_cols=None,
                 dim=None, n_channels=1, out_activation="softmax",
                 complexity_factor=0.75, dilations=(1, 2, 4),
                 l1_reg=None, l2_reg=None, base_model=None, logger=None,
                 **kwargs):
        """
        img_rows, img_cols (int, int):
            Image dimensions. Note that depending on image dims cropping may
            be necessary. To avoid this, use image dimensions DxD for which
            D * (1/2)^n is an integer, where n is the number of (2x2)
            max-pooling layers; in this implementation 4.
            For n=4, D \in {..., 192, 208, 224, 240, 256, ...} etc.
        n_classes (int):
            The number of classes to model, gives the number of filters in the
            final 1x1 conv layer.
        n_channels (int):
            Number of channels in the input image.
        out_activation (string):
            Activation function of output 1x1 conv layer. Usually one of
            'softmax', 'sigmoid' or 'linear'.
        complexity_factor (int/float):
            Use int(N * sqrt(complexity_factor)) number of filters in each
            2D convolution layer instead of default N.
        l1_reg (float in [0, 1])
            L1 regularization on Conv2D activities
        l2_reg (float in [0, 1])
            21 regularization on Conv2D weights
        logger (MultiViewUNet.logging.Logger | ScreenLogger):
            MutliViewUNet.Logger object, logging to files or screen.
        """
        if not ((img_rows and img_cols) or dim):
            raise ValueError("Must specify either img_rows and img_col or dim")
        if dim:
            img_rows, img_cols = dim, dim

        # Set logger or standard print wrapper
        self.logger = logger or ScreenLogger()

        self.img_shape = (img_rows, img_cols, n_channels)
        self.n_classes = n_classes
        self.cf = np.sqrt(complexity_factor)
        self.dilations = dilations

        # Shows the number of pixels cropped of the input image to the output
        self.label_crop = np.array([[0, 0], [0, 0]])

        # Build model and init base keras Model class
        if not base_model:
            # New training session
            super().__init__(*self.init_model(out_activation, l1_reg,
                                              l2_reg))
        else:
            # Resumed training
            super().__init__(base_model.input, base_model.output)

        # Log the model definition
        self._log()

    def init_model(self, out_activation, l1, l2, **kwargs):
        """
        Build the UNet model with the specified input image shape.

        OBS: In some cases, the output is smaller than the input.
        self.label_crop stores the number of pixels that must be cropped from
        the target labels matrix to compare correctly.
        """
        inputs = Input(shape=self.img_shape)

        # Apply regularization if not None or 0
        # Note, currently:
        # l1 regularization applied for layer activity
        # l2 regularization applied for convolution weights
        kr = regularizers.l2(l2) if l2 else None
        ar = regularizers.l1(l1) if l1 else None

        self.logger("l1 regular ization (activity):  %s" % l1)
        self.logger("l2 regularization (weights):   %s" % l2)

        """
        Contracting path
        Note: Listed tensor shapes assume img_row = 256, img_col = 256, cf=1 
        """
        # [256, 256, 1] -> [256, 256, 64] -> [256, 256, 64] -> [128, 128, 64]
        conv1 = Autofocus2D(self.dilations, filters=int(64*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar,
                            kernel_regularizer=kr)(inputs)
        conv1 = Conv2D(int(64*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv1)
        bn1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

        # [128, 128, 64] -> [128, 128, 128] -> [128, 128, 128] -> [64, 64, 128]
        conv2 = Autofocus2D(self.dilations, filters=int(128*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(pool1)
        conv2 = Conv2D(int(128*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv2)
        bn2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

        # [64, 64, 128] -> [64, 64, 256] -> [64, 64, 256] -> [32, 32, 256]
        conv3 = Autofocus2D(self.dilations, filters=int(256*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(pool2)
        conv3 = Conv2D(int(256*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv3)
        bn3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

        # [32, 32, 256] -> [32, 32, 512] -> [32, 32, 512] -> [16, 16, 512]
        conv4 = Autofocus2D(self.dilations, filters=int(512*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(pool3)
        conv4 = Conv2D(int(512*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv4)
        bn4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

        # [16, 16, 512] -> [16, 16, 1024] -> [16, 16, 1024]
        conv5 = Autofocus2D(self.dilations, filters=int(1024*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(pool4)
        conv5 = Conv2D(int(1024*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv5)
        bn5 = BatchNormalization()(conv5)

        """
        Up-sampling
        """
        # [16, 16, 1024] -> [32, 32, 1024] -> [32, 32, 512]
        up1 = UpSampling2D(size=(2, 2))(bn5)
        conv6 = Conv2D(int(512*self.cf), 2, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(up1)
        bn6 = BatchNormalization()(conv6)

        # Merge conv4 [32, 32, 512] with conv6 [32, 32, 512]
        # --> [32, 32, 1024]
        cropped_bn4 = self.crop_nodes_to_match(bn4, bn6)
        merge6 = Concatenate(axis=-1)([cropped_bn4, bn6])

        # [32, 32, 1024] -> [32, 32, 512] -> [32, 32, 512]
        conv6 = Autofocus2D(self.dilations, filters=int(512*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(merge6)
        conv6 = Conv2D(int(512*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv6)
        bn7 = BatchNormalization()(conv6)

        # [32, 32, 512] -> [64, 64, 512] -> [64, 64, 256]
        up2 = UpSampling2D(size=(2, 2))(bn7)
        conv7 = Conv2D(int(256*self.cf), 2, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(up2)
        bn8 = BatchNormalization()(conv7)

        # Merge conv3 [64, 64, 256] with conv7 [64, 64, 256]
        # --> [32, 32, 512]
        cropped_bn3 = self.crop_nodes_to_match(bn3, bn8)
        merge7 = Concatenate(axis=-1)([cropped_bn3, bn8])

        # [64, 64, 512] -> [64, 64, 256] -> [64, 64, 256]
        conv7 = Autofocus2D(self.dilations, filters=int(256*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(merge7)
        conv7 = Conv2D(int(256*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv7)
        bn9 = BatchNormalization()(conv7)

        # [64, 64, 256] -> [128, 128, 256] -> [128, 128, 128]
        up3 = UpSampling2D(size=(2, 2))(bn9)
        conv8 = Conv2D(int(128*self.cf), 2, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(up3)
        bn10 = BatchNormalization()(conv8)

        # Merge conv2 [128, 128, 128] with conv8 [128, 128, 128]
        # --> [128, 128, 256]
        cropped_bn2 = self.crop_nodes_to_match(bn2, bn10)
        merge8 = Concatenate(axis=-1)([cropped_bn2, bn10])

        # [128, 128, 256] -> [128, 128, 128] -> [128, 128, 128]
        conv8 = Autofocus2D(self.dilations, filters=int(128*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(merge8)
        conv8 = Conv2D(int(128*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv8)
        bn11 = BatchNormalization()(conv8)

        # [128, 128, 128] -> [256, 256, 128] -> [256, 256, 64]
        up4 = UpSampling2D(size=(2, 2))(bn11)
        conv9 = Conv2D(int(64*self.cf), 2, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(up4)
        bn12 = BatchNormalization()(conv9)

        # Merge conv1 [256, 256, 64] with conv9 [256, 256, 64]
        # --> [256, 256, 128]
        cropped_bn1 = self.crop_nodes_to_match(bn1, bn12)
        merge9 = Concatenate(axis=-1)([cropped_bn1, bn12])

        # [256, 256, 128] -> [256, 256, 64] -> [256, 256, 64]
        conv9 = Autofocus2D(self.dilations, filters=int(64*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(merge9)
        conv9 = Conv2D(int(64*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv9)
        bn13 = BatchNormalization()(conv9)

        """
        Output modeling layer
        """
        # [256, 256, 64] -> [256, 256, n_classes]
        self.logger("Final layer activation: %s" % out_activation)
        out = Conv2D(self.n_classes, 1, activation=out_activation)(bn13)

        return inputs, out

    def crop_nodes_to_match(self, node1, node2):
        """
        If necessary, applies Cropping2D layer to node1 to match shape of node2
        """
        s1 = np.array(node1.get_shape().as_list())[1:-1]
        s2 = np.array(node2.get_shape().as_list())[1:-1]

        if np.any(s1 != s2):
            print(node1, node2)
            c = (s1 - s2).astype(np.int)
            cr = np.array([c//2, c//2]).T
            cr[:, 1] += c % 2
            cropped_node1 = Cropping2D(cr)(node1)
            self.label_crop += cr
        else:
            cropped_node1 = node1

        return cropped_node1

    def _log(self):
        self.logger("Image rows:        %i" % self.img_shape[0])
        self.logger("Image cols:        %i" % self.img_shape[1])
        self.logger("Image channels:    %i" % self.img_shape[2])
        self.logger("N classes:         %i" % self.n_classes)
        self.logger("CF factor:         %.3f" % self.cf**2)
        self.logger("N params:          %i" % self.count_params())
        self.logger("Output:            %s" % self.output)
        self.logger("Crop:              %s" % (self.label_crop if np.sum(self.label_crop) != 0 else "None"))
