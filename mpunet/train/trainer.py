import os
import numpy as np
import tensorflow as tf
from mpunet.callbacks import init_callback_objects
from mpunet.logging import ScreenLogger
from mpunet.callbacks import (SavePredictionImages, Validation,
                              FGBatchBalancer, DividerLine,
                              LearningCurve, MemoryConsumption,
                              MeanReduceLogArrays, remove_validation_callbacks)
from mpunet.utils import ensure_list_or_tuple
from mpunet.train.utils import (ensure_sparse,
                                init_losses,
                                init_metrics,
                                init_optimizer)
from tensorflow.python.framework.errors_impl import (ResourceExhaustedError,
                                                     InternalError)


def get_steps(sequence, im_per_epoch=None):
    """ Returns the number of gradient steps to take in an epoch """
    if im_per_epoch:
        steps = int(np.ceil(im_per_epoch / sequence.batch_size))
    else:
        steps = len(sequence)
    return steps


class Trainer(object):
    """
    Handles initialization and logging of model fitting sessions.
    """
    def __init__(self, model, logger=None):
        """
        Init. simply accepts a model and stores it.
        Optionally, an 'org_model' (original model) may be passed and stored
        as well. This is for training multi-GPU models prepared by the
        tf.keras.utils.multi_gpu_model utility, which returns a new, split
        model for training (passed as 'model' parameter here). For properly
        saving the model parameter, however, it is recommended to use the
        original, non-split model (here passed as 'org_model').

        Args:
            model:      (tf.keras Model) Initialized model to train
            org_model:  (tf.keras Model) Optional single-GPU version for the
                                         passed 'model' parameter.
            logger:     (Logger)         Optional Logger instance
        """
        self.model = model
        self.logger = logger if logger is not None else ScreenLogger()

    def compile_model(self, optimizer, loss, metrics, reduction,
                      check_sparse=False, optimizer_kwargs={}, loss_kwargs={},
                      **kwargs):
        """
        Compile the stored tf.keras Model instance stored in self.model
        Sets the loss function, optimizer and metrics

        Args:
            optimizer:        (string) The name of a tf.keras.optimizers Optimizer
            optimizer_kwargs: (dict)   Key-word arguments passed to the Optimizer
            loss:             (string) The name of a tf.keras.losses or
                                       MultiPlanarUnet loss function
            metrics:          (list)   List of tf.keras.metrics or
                                       mpunet metrics.
            reduction:        TODO
            check_sparse:     TODO
            **kwargs:         (dict)   Key-word arguments passed to losses
                                       and/or metrics that accept such.
        """
        # Make sure sparse metrics and loss are specified as sparse
        metrics = ensure_list_or_tuple(metrics)
        losses = ensure_list_or_tuple(loss)
        if check_sparse:
            ensure_sparse(metrics+losses)

        # Initialize optimizer, loss(es) and metric(s) from tf.keras or
        # mpunet
        optimizer = init_optimizer(optimizer, self.logger, **optimizer_kwargs)
        losses = init_losses(losses, self.logger, **kwargs)
        for i, loss in enumerate(losses):
            try:
                losses[i] = loss(reduction=reduction, **loss_kwargs)
            except (ValueError, TypeError):
                raise TypeError("All loss functions must currently be "
                                "callable and accept the 'reduction' "
                                "parameter specifying a "
                                "tf.keras.losses.Reduction type. If you "
                                "specified a keras loss function such as "
                                "'sparse_categorical_crossentropy', change "
                                "this to its corresponding loss class "
                                "'SparseCategoricalCrossentropy'. If "
                                "you implemented a custom loss function, "
                                "please raise an issue on GitHub.")
        metrics = init_metrics(metrics, self.logger, **kwargs)

        # Compile the model
        self.model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        self.logger("Optimizer:   %s" % optimizer)
        self.logger("Loss funcs:  %s" % losses)
        self.logger("Metrics:     %s" % init_metrics)
        return self

    def fit(self, train, val, batch_size, no_im=False, **fit_kwargs):
        """
        Fit the stored tf.keras Model (self.model) on a set of data.

        The 'fit' method is a wrapper around the hidden '_fit' method. It
        handles KeyboardInterrupts (--> stopping training prematurely), TF
        GPU memory errors (--> batch_size is reduced by 2 and training
        restarted), and other exceptions (--> error logged and training
        terminated).

        Please refer to the self._fit method for 'fit_kwargs' argument details.

        Args:
            train:      TODO
            val:        TODO
            batch_size: (int)  The initial batch size to run training with
            no_im:      TODO
            fit_kwargs: (dict) Keyword arguments passed to self._fit
        """
        # Crop labels?
        if hasattr(self.model, "label_crop"):
            train.label_crop = self.model.label_crop
            if val is not None:
                val.label_crop = self.model.label_crop
        if type(train).__name__ == "MultiTaskSequence":
            self.logger("-- Skipping saving images (not yet implemented for"
                        " MultiTaskSequences).")
            no_im = True
        # Save a few images to disk for inspection
        if no_im:
            self.logger("No images saved (--no_images flag is set)")
        else:
            from mpunet.utils.plotting import save_images
            im_path = os.path.join(self.logger.base_path, "images")
            save_images(train, val, im_path, self.logger)

        # Start fitting
        fitting = True
        while fitting:
            try:
                self._fit(train=train,
                          val=val,
                          batch_size=batch_size,
                          no_im=no_im,
                          **fit_kwargs)
                fitting = False
            except (ResourceExhaustedError, InternalError):
                # Reduce batch size
                batch_size -= 2
                self.logger("\n\n[MEMORY ERROR] Reducing batch size "
                            "by 2 (now %i)" % batch_size)
                if batch_size < 1:
                    self.logger("[ERROR] Batch size negative or zero!")
                    fitting = False
            except KeyboardInterrupt:
                fitting = False
            except Exception as e:
                self.logger(e)
                raise e

        try:
            if train.image_pair_loader.queue:
                train.image_pair_loader.queue.stop()
            if val.image_pair_loader.queue:
                val.image_pair_loader.queue.stop()
        except AttributeError:
            # Multi-tasking, train.image_pair_loader will be a list
            # TODO: Make all sequences store a reference to the queue
            pass

        self.logger("Training stopped.")
        self.logger.print_calling_method = True
        return self.model

    def _fit(self,
             train,
             val,
             batch_size,
             n_epochs,
             callbacks,
             train_im_per_epoch,
             val_im_per_epoch,
             val_ignore_class_zero=True,
             no_im=False,
             verbose=1,
             init_epoch=0,
             use_multiprocessing=False,
             **unused):
        train.batch_size = batch_size

        # Get number of steps per train epoch
        train_steps = get_steps(train, train_im_per_epoch)
        self.logger("Using %i steps per train epoch (total batches=%i)" %
                    (train_steps, len(train)))

        if val is None:
            # No validation to be performed, remove callbacks that might need
            # validation data to function properly
            remove_validation_callbacks(callbacks, self.logger)
        else:
            val.batch_size = batch_size
            val_steps = get_steps(val, val_im_per_epoch)
            self.logger("Using %i steps per val epoch (total batches=%i)" %
                        (val_steps, len(val)))
            # Add validation callback
            # Important: Should be first in callbacks list as other CBs may
            # depend on the validation metrics/loss
            validation = Validation(val,
                                    steps=val_steps,
                                    ignore_class_zero=val_ignore_class_zero,
                                    logger=self.logger,
                                    verbose=verbose)
            callbacks = [validation] + callbacks

        # Add various callbacks for plotting learning curves etc.
        # Get FGBatchBalancer callbacks, etc.
        if hasattr(train, "n_fg_slices"):
            callbacks.append(FGBatchBalancer(train, logger=self.logger))
        if not no_im:
            # Add save images cb
            callbacks.append(SavePredictionImages(train, val))
        callbacks.insert(1, MeanReduceLogArrays())
        # callbacks.insert(1, MemoryConsumption(logger=self.logger))
        callbacks.append(LearningCurve(logger=self.logger))
        callbacks.append(DividerLine(self.logger))

        # Get initialized callback objects
        callbacks, cb_dict = init_callback_objects(callbacks, self.logger)

        # If ModelCheckPointClean is used, set the original model to store
        # the correct weights when using multi-GPU models
        cb = cb_dict.get("ModelCheckPointClean")
        if cb:
            cb.org_model = self.model  # TEMP TODO

        # Init TF dataset with DATA autosharding
        dtypes, shapes = list(zip(*map(lambda x: (x.dtype, x.shape), train[0])))
        train = tf.data.Dataset.from_generator(train, dtypes, shapes)

        # Fit the model
        # is_queued = bool(train.image_pair_loader.queue)
        self.logger.active_log_file = "training"
        self.logger.print_calling_method = False
        self.model.fit(
            train,
            steps_per_epoch=train_steps,
            epochs=n_epochs,
            callbacks=callbacks,
            initial_epoch=init_epoch,
            use_multiprocessing=use_multiprocessing,
            workers=5,
            max_queue_size=5,
            shuffle=False,  # Determined by the chosen Sequence class
            verbose=verbose
        )
