import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from MultiPlanarUNet.logging import ScreenLogger
from MultiPlanarUNet.utils.plotting import imshow_with_label_overlay, imshow

import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime


class DividerLine(Callback):
    """
    Simply prints a line to screen after each epoch
    """
    def __init__(self, logger=None):
        """
        Args:
            logger: An instance of a MultiPlanar Logger that prints to screen
                    and/or file
        """
        super().__init__()
        self.logger = logger or ScreenLogger()

    def on_epoch_end(self, epoch, logs=None):
        self.logger("-"*45 + "\n")


class DelayedCallback(object):
    """
    Callback wrapper that delays the functionality of another callback by N
    number of epochs.
    """
    def __init__(self, callback, start_from=0, logger=None):
        """
        Args:
            callback:   A tf.keras callback
            start_from: Delay the activity of 'callback' until this epoch
                        'start_from'
            logger:     An instance of a MultiPlanar Logger that prints to screen
                        and/or file
        """
        self.logger = logger or ScreenLogger()
        self.callback = callback
        self.start_from = start_from

    def __getattr__(self, item):
        return getattr(self.callback, item)

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_from-1:
            self.callback.on_epoch_end(epoch, logs=logs)
        else:
            self.logger("[%s] Not active at epoch %i - will be at %i" %
                        (self.callback.__class__.__name__,
                         epoch+1, self.start_from))


class TrainTimer(Callback):
    """
    Appends train timing information to the log.
    If called prior to tf.keras.callbacks.CSVLogger this information will
    be written to disk.
    """
    def __init__(self, logger=None, verbose=1):
        super().__init__()
        self.logger = logger or ScreenLogger()
        self.verbose = bool(verbose)

        # Timing attributes
        self.train_begin_time = None
        self.prev_epoch_time = None

    @staticmethod
    def parse_dtime(tdelta, fmt):
        # https://stackoverflow.com/questions/8906926/
        # formatting-python-timedelta-objects/17847006
        d = {"days": tdelta.days}
        d["hours"], rem = divmod(tdelta.seconds, 3600)
        d["minutes"], d["seconds"] = divmod(rem, 60)
        return fmt.format(**d)

    def on_train_begin(self, logs=None):
        self.train_begin_time = datetime.now()

    def on_epoch_begin(self, epoch, logs=None):
        self.prev_epoch_time = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        # Compute epoch execution time
        end_time = datetime.now()
        epoch_time = end_time - self.prev_epoch_time
        train_time = end_time - self.train_begin_time

        # Update attributes
        self.prev_epoch_time = end_time

        # Add to logs
        logs["train_time_epoch"] = self.parse_dtime(epoch_time,
                                                    "{days:02}d:{hours:02}h:"
                                                    "{minutes:02}m:{seconds:02}s")
        logs["train_time_total"] = self.parse_dtime(train_time,
                                                    "{days:02}d:{hours:02}h:"
                                                    "{minutes:02}m:{seconds:02}s")

        if self.verbose:
            self.logger("[TrainTimer] Epoch time: %.1f minutes "
                        "- Total train time: %s"
                        % (epoch_time.total_seconds()/60,
                           logs["train_time_total"]))


class FGBatchBalancer(Callback):
    """
    MultiPlanarUNet callback.

    Sets the forced FG fraction in a batch at each epoch to 1-recall over the
    validation data at the previous epoch
    """
    def __init__(self, train_data, val_data=None, logger=None):
        """
        Args:
            train_data: A MultiPlanarUNet.sequence object representing the
                        training data
            val_data:   A MultiPlanarUNet.sequence object representing the
                        validation data
            logger:     An instance of a MultiPlanar Logger that prints to screen
                        and/or file
        """
        super().__init__()
        self.data = (("train", train_data), ("val", val_data))
        self.logger = logger or ScreenLogger()
        self.active = True

    def on_epoch_end(self, epoch, logs=None):
        if not self.active:
            return None

        recall = logs.get("val_recall")
        if recall is None:
            self.logger("[FGBatchBalancer] No val_recall in logs. "
                        "Disabling callback. "
                        "Did you put this callback before the validation "
                        "callback?")
            self.active = False
        else:
            # Always at least 1 image slice
            fraction = max(0.01, 1 - recall)
            for name, data in self.data:
                if data is not None:
                    data.fg_batch_fraction = fraction
                    self.logger("[FGBatchBalancer] Setting FG fraction for %s "
                                "to: %.4f - Now %s/%s" % (name,
                                                          fraction,
                                                          data.n_fg_slices,
                                                          data.batch_size))


class PrintLayerWeights(Callback):
    """
    Print the weights of a specified layer every some epoch or batch.
    """
    def __init__(self, layer, every=10, first=10, per_epoch=False, logger=None):
        """
        Args:
            layer:      A tf.keras layer
            every:      Print the weights every 'every' batch or epoch if
                        per_epoch=True
            first:      Print the first 'first' elements of each weight matrix
            per_epoch:  Print after 'every' epoch instead of batch
            logger:     An instance of a MultiPlanar Logger that prints to screen
                        and/or file
        """
        super().__init__()
        if isinstance(layer, int):
            self.layer = self.model.layers[layer]
        else:
            self.layer = layer
        self.first = first
        self.every = every
        self.logger = logger or ScreenLogger()

        self.per_epoch = per_epoch
        if per_epoch:
            # Apply on every epoch instead of per batches
            self.on_epoch_begin = self.on_batch_begin
            self.on_batch_begin = lambda *args, **kwargs: None
        self.log()

    def log(self):
        self.logger("PrintLayerWeights Callback")
        self.logger("Layer:      ", self.layer)
        self.logger("Every:      ", self.every)
        self.logger("First:      ", self.first)
        self.logger("Per epoch:  ", self.per_epoch)

    def on_batch_begin(self, batch, logs=None):
        if batch % self.every:
            return
        weights = self.layer.get_weights()
        self.logger("Weights for layer '%s'" % self.layer)
        self.logger("Weights:\n%s" % weights[0].ravel()[:self.first])
        try:
            self.logger("Baises:\n%s" % weights[1].ravel()[:self.first])
        except IndexError:
            pass


class SaveOutputAs2DImage(Callback):
    """
    Save random 2D slices from the output of a given layer during training.
    """
    def __init__(self, layer, sequence, model, out_dir, every=10, logger=None):
        """
        Args:
            layer:    A tf.keras layer
            sequence: A MultiPlanar.sequence object from which batches are
                      sampled and pushed through the graph to output of layer
            model:    A tf.keras model object
            out_dir:  Path to directory (existing or non-existing) in which
                      images will be stored
            every:    Perform this operation every 'every' batches
        """
        super().__init__()
        self.every = every
        self.seq = sequence
        self.layer = layer
        self.epoch = None
        self.model = model
        self.logger = logger or ScreenLogger()

        self.out_dir = out_dir
        if not os.path.exists(out_dir):
            os.makedirs(self.out_dir)
        self.log()

    def log(self):
        self.logger("Save Output as 2D Image Callback")
        self.logger("Layer:      ", self.layer)
        self.logger("Every:      ", self.every)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_end(self, batch, logs=None):
        if batch % self.every:
            return

        # Get output of layer
        self.model.predict_on_batch()
        sess = tf.keras.backend.get_session()
        X, _, _ = self.seq[0]
        outs = sess.run([self.layer.output], feed_dict={self.model.input: X})[0]
        if isinstance(outs, list):
            outs = outs[0]

        for i, (model_in, layer_out) in enumerate(zip(X, outs)):
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            # Plot model input and layer outputs on each ax
            chl1, axis, slice = imshow(ax1, model_in)
            chl2, _, _ = imshow(ax2, layer_out, axis=axis, slice=slice)

            # Set labels and save figure
            ax1.set_title("Model input - Channel %i - Axis %i - Slice %i"
                          % (chl1, axis,slice), size=22)
            ax2.set_title("Layer output - Channel %i - Axis %i - Slice %i"
                          % (chl2, axis, slice), size=22)

            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, "epoch_%i_batch_%i_im_%i" %
                                     (self.epoch, batch, i)))
            plt.close(fig)


class SavePredictionImages(Callback):
    """
    Save images after each epoch of training of the model on a batch of
    training and a batch of validation data sampled from sequence objects.

    Saves the input image with ground truth overlay as well as the predicted
    label masks.
    """
    def __init__(self, train_data, val_data, outdir='images'):
        """
        Args:
            train_data: A MultiPlanarUNet.sequence object from which training
                        data can be sampled via the __getitem__ method.
            val_data:   A MultiPlanarUNet.sequence object from which validation
                        data can be sampled via the __getitem__ method.
            outdir:     Path to directory (existing or non-existing) in which
                        images will be stored.
        """
        super().__init__()

        self.train_data = train_data
        self.val_data = val_data

        self.save_path = os.path.abspath(os.path.join(outdir, "pred_images_at_epoch"))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def pred_and_save(self, data, subdir):
        # Get a random batch
        X, y, _ = data[np.random.randint(len(data))]

        # Predict on the batch
        pred = self.model.predict(X)

        subdir = os.path.join(self.save_path, subdir)
        if not os.path.exists(subdir):
            os.mkdir(subdir)

        # Plot each sample in the batch
        for i, (im, lab, p) in enumerate(zip(X, y, pred)):
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 6))

            # Imshow ground truth on ax2
            # This function will determine which channel, axis and slice to
            # show and return so that we can use them for the other 2 axes
            chnl, axis, slice = imshow_with_label_overlay(ax2, im, lab, lab_alpha=1.0)

            # Imshow pred on ax3
            imshow_with_label_overlay(ax3, im, p, lab_alpha=1.0,
                                      channel=chnl, axis=axis, slice=slice)

            # Imshow raw image on ax1
            # Chose the same slice, channel and axis as above
            im = im[..., chnl]
            im = np.moveaxis(im, axis, 0)
            if slice is not None:
                # Only for 3D imges
                im = im[slice]
            ax1.imshow(im, cmap="gray")

            # Set labels
            ax1.set_title("Image", size=18)
            ax2.set_title("True labels", size=18)
            ax3.set_title("Prediction", size=18)

            fig.tight_layout()
            with np.testing.suppress_warnings() as sup:
                sup.filter(UserWarning)
                fig.savefig(os.path.join(subdir, str(i) + ".png"))
            plt.close(fig.number)

    def on_epoch_end(self, epoch, logs={}):
        self.pred_and_save(self.train_data, "train_%s" % epoch)
        if self.val_data is not None:
            self.pred_and_save(self.val_data, "val_%s" % epoch)
