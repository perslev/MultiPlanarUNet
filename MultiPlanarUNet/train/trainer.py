# from MultiPlanarUNet.database import DBConnection
from MultiPlanarUNet.callbacks import init_callback_objects
from MultiPlanarUNet.logging import ScreenLogger
from MultiPlanarUNet.evaluate import loss_functions
from MultiPlanarUNet.evaluate import metrics as custom_metrics
from MultiPlanarUNet.utils import pred_to_class
from MultiPlanarUNet.callbacks import SavePredictionImages, Validation, FGBatchBalancer, DividerLine, SaveOutputAs2DImage, PrintLayerWeights
from tensorflow.keras import optimizers, losses
from tensorflow.keras import metrics as TF_metrics
import matplotlib.pyplot as plt
import os
import numpy as np
from multiprocessing import cpu_count
from tensorflow.python.framework.errors_impl import ResourceExhaustedError


class Trainer(object):
    """
    Handles initialization and logging of model fitting sessions.
    Fits models as implemented in thesis.models
    Stores logs, models etc. in a SQLite3 db as implemented in thesis.database

    USAGE
    -----
    Trainer(project_folder, model, hparams)

    project_folder : string, path to project folder
                     If folder is empty, the folder is used directly.
                     Otherwise a sub-folder is created named based on the
                     passed model & hparams objects.
    model          : Model object, model to fit
    hparams        : HParams object, stores hyperparameters of the fitting
    """
    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger if logger is not None else ScreenLogger()
        self.metadata = None

        # Extra reference to original (non multiple-GPU) model
        # Is set from train.py as needed
        self.org_model = None

    def compile_model(self, optimizer, optimizer_kwargs, loss, metrics,
                      sparse=False, mem_logging=False, **kwargs):
        # Initialize optimizer
        optimizer = optimizers.__dict__[optimizer]
        optimizer = optimizer(**optimizer_kwargs)

        # # Initialize loss
        if loss in losses.__dict__:
            loss = losses.get(loss)
        else:
            import inspect
            loss = loss_functions.__dict__[loss]
            if inspect.isclass(loss):
                loss = loss(logger=self.logger, **kwargs)

        if sparse:
            # Make sure sparse metrics are specified
            for i, m in enumerate(metrics):
                if "sparse" not in m:
                    new = "sparse_" + m
                    self.logger("Note: changing %s --> "
                                "%s (sparse=True passed)" % (m, new))
                    metrics[i] = new

        # Find metrics both from standard keras.metrics module and own custom
        init_metrics = []
        for m in metrics:
            if m in TF_metrics.__dict__:
                init_metrics.append(TF_metrics.get(m))
            else:
                import inspect
                metric = custom_metrics.__dict__[m]
                if inspect.isclass(metric):
                    metric = metric(logger=self.logger, **kwargs)
                init_metrics.append(metric)

        # Compile the model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=init_metrics)

        self.logger("Optimizer:   %s" % optimizer)
        self.logger("Loss:        %s" % loss)
        self.logger("Targets:     %s" % ("Integer" if sparse else "One-Hot"))
        self.logger("Metrics:     %s" % init_metrics)

        return self

    def fit(self, train, val, callbacks, n_epochs, shuffle_batch_order,
            hparams, batch_size=8, verbose=1, init_epoch=0, no_im=False,
            **kwargs):

        # Crop labels?
        if hasattr(self.model, "label_crop"):
            train.label_crop = self.model.label_crop
            val.label_crop = self.model.label_crop

        # Log various info on the data
        self._log_sequences(train, val)

        # Save a few images to disk for inspection
        if no_im:
            self.logger("No images saved (--no_images flag is set)")
        else:
            self.save_images(train, val)

        # Start fitting
        fitting = True
        while fitting:
            try:
                self._fit_loop(train, val, batch_size, n_epochs, verbose,
                               callbacks, shuffle_batch_order, init_epoch,
                               no_im, hparams)
                fitting = False
            except ResourceExhaustedError:
                # Reduce batch size
                batch_size -= 2
                hparams["fit"]["batch_size"] = batch_size
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

        if train.image_pair_loader.queue:
            train.image_pair_loader.queue.stop()
        if val.image_pair_loader.queue:
            val.image_pair_loader.queue.stop()

        self.logger("Training stopped.")
        self.logger.print_calling_method = True
        return self.model

    def _fit_loop(self, train, val, batch_size, n_epochs, verbose, callbacks,
                  shuffle_batch_order, init_epoch, no_im, hparams):

        # Update batch size on generators (needed after OOM error --> reduced
        # batch size)
        train.batch_size = batch_size
        val.batch_size = batch_size

        # Get number of steps per train epoch
        train_steps = min(int(2500/batch_size), len(train))
        val_steps = min(int(3750/batch_size), len(val))
        # train_steps = 1
        # val_steps = 1

        self.logger("Using %i steps per train epoch (N batches=%i)" % (train_steps,
                                                                       len(train)))
        self.logger("Using %i steps per validation epoch (N batches=%i)" % (val_steps,
                                                                            len(val)))

        if train_steps < len(train):
            # Force batch shuffle as each epoch does not cover total number
            # of data points
            shuffle_batch_order = True

        if len(val) == 0:
            # No validation data
            val = None
        else:
            # Add validation callback
            # IMPORTANT: Should be first in callbacks list as other CBs may
            # depend on the validation metrics/loss
            validation = Validation(val, val_steps, logger=self.logger,
                                    verbose=verbose)
            callbacks = [validation] + callbacks

        # Add save layer output
        # callbacks.append(SaveOutputAs2DImage(self.model.layers[8], train,
        #                                      self.model,
        #                                      out_dir="images/layer_output",
        #                                      every=500, logger=self.logger))

        # Print layer weights?
        # callbacks.append(PrintLayerWeights(self.model.layers[7], every=500,
        #                                    first=30, logger=self.logger))

        if not no_im:
            # Add save images cb
            callbacks.append(SavePredictionImages(train, val))

        # Get validation and FGBatchBalancer callbacks, etc.
        FGbalancer = FGBatchBalancer(train, logger=self.logger)
        line = DividerLine(self.logger)
        callbacks = callbacks + [FGbalancer, line]

        # Get initialized callback objects
        callbacks, cb_dict = init_callback_objects(callbacks, self.logger)

        # If ModelCheckPointClean is used, set the original model to store
        # the correct weights when using multi-GPU models
        cb = cb_dict.get("ModelCheckPointClean")
        if cb:
            cb.org_model = self.org_model

        if len(val) == 0:
            # No validation data, remove callbacks dependent on such
            callbacks = [c for c in callbacks if not any("val" in s for s in [str(v) for v in c["kwargs"].values()])]

        # Fit the model
        is_queued = bool(train.image_pair_loader.queue)
        self.logger.active_log_file = "training"
        self.logger.print_calling_method = False

        self.model.fit_generator(generator=train,
                                 steps_per_epoch=train_steps,
                                 epochs=n_epochs,
                                 verbose=verbose,
                                 callbacks=callbacks,
                                 shuffle=shuffle_batch_order,
                                 initial_epoch=init_epoch,
                                 use_multiprocessing=True,
                                 workers=cpu_count()-1,
                                 max_queue_size=3 if is_queued else 10)

    def save_metadata_trace(self, save_path):
        if not self.metadata:
            self.logger("[ERROR] Cannot save metadata trace (not found)")
        else:
            from tensorflow.python.client import timeline

            trace = timeline.Timeline(step_stats=self.metadata.step_stats)
            with open('%s.json' % save_path, 'w') as f:
                f.write(trace.generate_chrome_trace_format())

    def _log_sequences(self, train, val):
        unique, counts = train.count()

        self.logger("Number of samples: %s" % train.n_samples)
        if len(unique) > 1:
            self.logger("Real: %i (weight: %s)" % (counts[1], unique[1]))
            self.logger("Augmented: %i (weight: %s)" % (counts[0], unique[0]))
        else:
            self.logger("Real: %i (weight: %s)" % (counts[0], unique[0]))
            self.logger("Number of batches:", len(train))
        self.logger("Number of validation samples:", val.n_samples)
        self.logger("Number of batches:", len(val))

    def save_images(self, train, val):
        from MultiPlanarUNet.utils.plotting import imshow_with_label_overlay
        # Write a few images to disk
        im_path = os.path.join(self.logger.base_path, "images")
        if not os.path.exists(im_path):
            os.mkdir(im_path)

        training = train[0]
        if val is not None:
            validation = val[0]
            v_len = len(validation[0])
        else:
            validation = None
            v_len = 0

        self.logger("Saving %i sample images in '<project_dir>/images' folder"
                    % ((len(training[0]) + v_len) * 2))
        for rr in range(2):
            for k, temp in enumerate((training, validation)):
                if temp is None:
                    # No validation data
                    continue
                X, Y, W = temp
                for i, (xx, yy, ww) in enumerate(zip(X, Y, W)):
                    # Make figure
                    fig = plt.figure(figsize=(10, 4))
                    ax1 = fig.add_subplot(121)
                    ax2 = fig.add_subplot(122)

                    # Plot image and overlayed labels
                    chnl, view, _ = imshow_with_label_overlay(ax1, xx, yy)

                    # Plot histogram
                    ax2.hist(xx.flatten(), bins=200)

                    # Set labels
                    ax1.set_title("Channel %i - Axis %i - "
                                  "Weight %.3f" % (chnl, view, ww), size=18)

                    # Get path
                    out_path = im_path + "/%s%i.png" % ("train" if k == 0 else
                                                        "val", len(X) * rr + i)

                    with np.testing.suppress_warnings() as sup:
                        sup.filter(UserWarning)
                        fig.savefig(out_path)
                    plt.close(fig)
