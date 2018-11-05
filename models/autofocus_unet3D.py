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
from tensorflow.keras.layers import Input, BatchNormalization, Cropping3D, \
                                    Concatenate, Conv3D, MaxPooling3D, \
                                    UpSampling3D
from autofocus import Autofocus3D
import numpy as np


class AutofocusUNet3D(Model):
    """
    3D UNet implementation with batch normalization and complexity factor adj.

    See original paper at http://arxiv.org/abs/1505.04597
    """
    def __init__(self, n_classes, dim, n_channels=1,
                 out_activation="softmax", complexity_factor=0.5,
                 dilations=(1, 2, 4), l1_reg=None, l2_reg=None,
                 base_model=None, logger=None,
                 **kwargs):
        """

        """
        # Set logger or standard print wrapper
        self.logger = logger or ScreenLogger()

        self.img_shape = (dim, dim, dim, n_channels)
        self.n_classes = n_classes
        self.cf = np.sqrt(complexity_factor)
        self.dilations = dilations

        # Shows the number of pixels cropped of the input image to the output
        self.label_crop = np.array([[0, 0, 0], [0, 0, 0]])

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
        conv1 = Autofocus3D(self.dilations, filters=int(64*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar,
                            kernel_regularizer=kr)(inputs)
        conv1 = Conv3D(int(64*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv1)
        bn1 = BatchNormalization()(conv1)
        pool1 = MaxPooling3D(pool_size=2)(bn1)

        # [128, 128, 64] -> [128, 128, 128] -> [128, 128, 128] -> [64, 64, 128]
        conv2 = Autofocus3D(self.dilations, filters=int(128*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(pool1)
        conv2 = Conv3D(int(128*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv2)
        bn2 = BatchNormalization()(conv2)
        pool2 = MaxPooling3D(pool_size=2)(bn2)

        # [64, 64, 128] -> [64, 64, 256] -> [64, 64, 256] -> [32, 32, 256]
        conv3 = Autofocus3D(self.dilations, filters=int(256*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(pool2)
        conv3 = Conv3D(int(256*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv3)
        bn3 = BatchNormalization()(conv3)
        pool3 = MaxPooling3D(pool_size=2)(bn3)

        # [32, 32, 256] -> [32, 32, 512] -> [32, 32, 512] -> [16, 16, 512]
        conv4 = Autofocus3D(self.dilations, filters=int(512*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(pool3)
        conv4 = Conv3D(int(512*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv4)
        bn4 = BatchNormalization()(conv4)
        pool4 = MaxPooling3D(pool_size=2)(bn4)

        # [16, 16, 512] -> [16, 16, 1024] -> [16, 16, 1024]
        conv5 = Autofocus3D(self.dilations, filters=int(1024*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(pool4)
        conv5 = Conv3D(int(1024*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv5)
        bn5 = BatchNormalization()(conv5)

        """
        Up-sampling
        """
        # [16, 16, 1024] -> [32, 32, 1024] -> [32, 32, 512]
        up1 = UpSampling3D(size=2)(bn5)
        conv6 = Conv3D(int(512*self.cf), 2, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(up1)
        bn6 = BatchNormalization()(conv6)

        # Merge conv4 [32, 32, 512] with conv6 [32, 32, 512]
        # --> [32, 32, 1024]
        cropped_bn4 = self.crop_nodes_to_match(bn4, bn6)
        merge6 = Concatenate(axis=-1)([cropped_bn4, bn6])

        # [32, 32, 1024] -> [32, 32, 512] -> [32, 32, 512]
        conv6 = Autofocus3D(self.dilations, filters=int(512*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(merge6)
        conv6 = Conv3D(int(512*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv6)
        bn7 = BatchNormalization()(conv6)

        # [32, 32, 512] -> [64, 64, 512] -> [64, 64, 256]
        up2 = UpSampling3D(size=2)(bn7)
        conv7 = Conv3D(int(256*self.cf), 2, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(up2)
        bn8 = BatchNormalization()(conv7)

        # Merge conv3 [64, 64, 256] with conv7 [64, 64, 256]
        # --> [32, 32, 512]
        cropped_bn3 = self.crop_nodes_to_match(bn3, bn8)
        merge7 = Concatenate(axis=-1)([cropped_bn3, bn8])

        # [64, 64, 512] -> [64, 64, 256] -> [64, 64, 256]
        conv7 = Autofocus3D(self.dilations, filters=int(256*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(merge7)
        conv7 = Conv3D(int(256*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv7)
        bn9 = BatchNormalization()(conv7)

        # [64, 64, 256] -> [128, 128, 256] -> [128, 128, 128]
        up3 = UpSampling3D(size=2)(bn9)
        conv8 = Conv3D(int(128*self.cf), 2, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(up3)
        bn10 = BatchNormalization()(conv8)

        # Merge conv2 [128, 128, 128] with conv8 [128, 128, 128]
        # --> [128, 128, 256]
        cropped_bn2 = self.crop_nodes_to_match(bn2, bn10)
        merge8 = Concatenate(axis=-1)([cropped_bn2, bn10])

        # [128, 128, 256] -> [128, 128, 128] -> [128, 128, 128]
        conv8 = Autofocus3D(self.dilations, filters=int(128*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(merge8)
        conv8 = Conv3D(int(128*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv8)
        bn11 = BatchNormalization()(conv8)

        # [128, 128, 128] -> [256, 256, 128] -> [256, 256, 64]
        up4 = UpSampling3D(size=2)(bn11)
        conv9 = Conv3D(int(64*self.cf), 2, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(up4)
        bn12 = BatchNormalization()(conv9)

        # Merge conv1 [256, 256, 64] with conv9 [256, 256, 64]
        # --> [256, 256, 128]
        cropped_bn1 = self.crop_nodes_to_match(bn1, bn12)
        merge9 = Concatenate(axis=-1)([cropped_bn1, bn12])

        # [256, 256, 128] -> [256, 256, 64] -> [256, 256, 64]
        conv9 = Autofocus3D(self.dilations, filters=int(64*self.cf),
                            kernel_size=3, activation='relu',
                            activity_regularizer=ar, kernel_regularizer=kr)(merge9)
        conv9 = Conv3D(int(64*self.cf), 3, activation='relu', padding='SAME',
                       activity_regularizer=ar, kernel_regularizer=kr)(conv9)
        bn13 = BatchNormalization()(conv9)

        """
        Output modeling layer
        """
        # [256, 256, 64] -> [256, 256, n_classes]
        self.logger("Final layer activation: %s" % out_activation)
        out = Conv3D(self.n_classes, 1, activation=out_activation)(bn13)

        return inputs, out

    def crop_nodes_to_match(self, node1, node2):
        """
        If necessary, applies Cropping3D layer to node1 to match shape of node2
        """
        s1 = np.array(node1.get_shape().as_list())[1:-1]
        s2 = np.array(node2.get_shape().as_list())[1:-1]

        if np.any(s1 != s2):
            print(node1, node2)
            c = (s1 - s2).astype(np.int)
            cr = np.array([c//2, c//2]).T
            cr[:, 1] += c % 2
            cropped_node1 = Cropping3D(cr)(node1)
            self.label_crop += cr
        else:
            cropped_node1 = node1

        return cropped_node1

    def _log(self):
        self.logger("Image rows:        %i" % self.img_shape[0])
        self.logger("Image cols:        %i" % self.img_shape[1])
        self.logger("Image depth:       %i" % self.img_shape[2])
        self.logger("Image channels:    %i" % self.img_shape[3])
        self.logger("Autofocus:         %s" % list(self.dilations))
        self.logger("N classes:         %i" % self.n_classes)
        self.logger("CF factor:         %.3f" % self.cf**2)
        self.logger("N params:          %i" % self.count_params())
        self.logger("Output:            %s" % self.output)
        self.logger("Crop:              %s" % (self.label_crop if np.sum(self.label_crop) != 0 else "None"))
