import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.callbacks import Callback
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from MultiPlanarUNet.logging import ScreenLogger
from MultiPlanarUNet.evaluate.metrics import dice_all
from MultiPlanarUNet.utils import highlighted, ensure_list_or_tuple


class Validation(Callback):
    """
    Validation computation callback.
    Samples a number of validation batches from a MultiPlanarUNet.sequence object
    and computes for all tasks:
        - Batch-wise validation loss
        - Epoch-wise pr-class and average precision
        - Epoch-wise pr-class and average recall
        - Epoch-wise pr-class and average dice coefficients
    ... and adds all results to the log dict
    Note: The purpose of this callback over the default tf.keras evaluation
    mechanism is to calculate certain metrics over the entire epoch of data as
    opposed to averaged batch-wise computations.
    """
    def __init__(self, val_sequence, steps, logger=None, verbose=True,
                 ignore_class_zero=True):
        """
        Args:
            val_sequence: A MultiPlanarUNet.sequence object from which validation
                          batches can be sampled via its __getitem__ method.
            steps:        Numer of batches to sample from val_sequences in each
                          validation epoch
            logger:       An instance of a MultiPlanar Logger that prints to screen
                          and/or file
            verbose:      Print progress to screen - OBS does not use Logger
        """
        super().__init__()
        self.logger = logger or ScreenLogger()
        self.data = val_sequence
        self.steps = steps
        self.verbose = verbose
        self.ignore_bg = ignore_class_zero

        self.n_classes = self.data.n_classes
        if isinstance(self.n_classes, int):
            self.task_names = [""]
            self.n_classes = [self.n_classes]
        else:
            self.task_names = self.data.task_names

    def predict(self):
        def count_cm_elements_from_queue(queue,
                                         steps,
                                         TPs,
                                         relevant,
                                         selected,
                                         task_names,
                                         n_classes_list,
                                         lock):
            step = 0
            while step < steps:
                # Get prediction and true labels from prediction queue
                step += 1
                pred, true = queue.get(block=True)

                for p, y, task_name, n_classes in zip(pred, true, task_names,
                                                      n_classes_list):
                    # Argmax and CM elements
                    p = p.argmax(-1).ravel()
                    y = y.ravel()

                    # Compute relevant CM elements
                    # We select the number following the largest class integer
                    # when y != pred, then bincount and remove the added dummy
                    # class
                    tps = np.bincount(np.where(y == p, y, n_classes),
                                      minlength=n_classes+1)[:-1]
                    rel = np.bincount(y, minlength=n_classes)
                    sel = np.bincount(p, minlength=n_classes)

                    # Update counts on shared lists
                    lock.acquire()
                    TPs[task_name] += tps.astype(np.uint64)
                    relevant[task_name] += rel.astype(np.uint64)
                    selected[task_name] += sel.astype(np.uint64)
                    lock.release()

        # Fetch some validation images from the generator
        pool = ThreadPoolExecutor(max_workers=7)
        result = pool.map(self.data.__getitem__, np.arange(self.steps))

        # Prepare arrays for CM summary stats
        TPs, relevant, selected = {}, {}, {}
        for task_name, n_classes in zip(self.task_names, self.n_classes):
            TPs[task_name] = np.zeros(shape=(n_classes,), dtype=np.uint64)
            relevant[task_name] = np.zeros(shape=(n_classes,), dtype=np.uint64)
            selected[task_name] = np.zeros(shape=(n_classes,), dtype=np.uint64)

        # Prepare queue and thread for computing counts
        from queue import Queue
        from threading import Thread
        count_queue = Queue(maxsize=self.steps)
        count_thread = Thread(target=count_cm_elements_from_queue,
                              args=[count_queue, self.steps,
                                    TPs, relevant, selected,
                                    self.task_names,
                                    self.n_classes, Lock()])
        count_thread.start()

        # Predict on all
        self.logger("")
        for i, (X, y) in enumerate(result):
            if self.verbose:
                print("   Validation: %i/%i" % (i+1, self.steps),
                      end="\r", flush=True)

            # Predict and put values in the queue for counting
            pred = self.model.predict_on_batch(ensure_list_or_tuple(X))
            count_queue.put([pred, ensure_list_or_tuple(y)])

        # Terminate count thread
        self.logger("Waiting for counting queue to terminate...")
        count_thread.join()
        pool.shutdown()
        return TPs, relevant, selected

    @staticmethod
    def _compute_dice(tp, rel, sel):
        # Get data masks (to avoid div. by zero warnings)
        # We set precision, recall, dice to 0 in for those particular cls.
        sel_mask = sel > 0
        rel_mask = rel > 0

        # prepare arrays
        precisions = np.zeros(shape=tp.shape, dtype=np.float32)
        recalls = np.zeros_like(precisions)
        dices = np.zeros_like(precisions)

        # Compute precisions, recalls
        precisions[sel_mask] = tp[sel_mask] / sel[sel_mask]
        recalls[rel_mask] = tp[rel_mask] / rel[rel_mask]

        # Compute dice
        intrs = (2 * precisions * recalls)
        union = (precisions + recalls)
        dice_mask = union > 0
        dices[dice_mask] = intrs[dice_mask] / union[dice_mask]

        return precisions, recalls, dices

    @staticmethod
    def _print_val_results(precisions, recalls, dices, epoch,
                           name, classes, ignore_bg, logger):
        # Log the results
        # We add them to a pd dataframe just for the pretty print output
        index = ["cls %i" % i for i in classes]
        val_results = pd.DataFrame({
            "precision": [np.nan] + list(precisions) if ignore_bg else precisions,
            "recall": [np.nan] + list(recalls) if ignore_bg else recalls,
            "dice": [np.nan] + list(dices) if ignore_bg else dices,
        }, index=index)
        # Transpose the results to have metrics in rows
        val_results = val_results.T
        # Add mean and set in first row
        means = [precisions.mean(), recalls.mean(), dices.mean()]
        val_results["mean"] = means
        cols = list(val_results.columns)
        cols.insert(0, cols.pop(cols.index('mean')))
        val_results = val_results.ix[:, cols]

        # Print the df to screen
        logger(highlighted("\n" + ("%s Validation Results for "
                                   "Epoch %i" % (name, epoch)).lstrip(" ")))
        logger(val_results.round(4))
        logger("")

    def on_epoch_end(self, epoch, logs={}):

        # Predict and get CM
        TPs, relevant, selected = self.predict()
        for name in self.task_names:
            tp, rel, sel = TPs[name], relevant[name], selected[name]
            precisions, recalls, dices = self._compute_dice(tp=tp, sel=sel, rel=rel)
            classes = np.arange(len(dices))
            if self.ignore_bg:
                precisions = precisions[1:]
                recalls = recalls[1:]
                dices = dices[1:]

            # Add to log
            if name:
                name += "_"
            logs["%sval_dice" % name] = dices.mean()
            logs["%sval_precision" % name] = precisions.mean()
            logs["%sval_recall" % name] = recalls.mean()

            if self.verbose:
                self._print_val_results(precisions=precisions, recalls=recalls,
                                        dices=dices, epoch=epoch, name=name,
                                        classes=classes,
                                        ignore_bg=self.ignore_bg,
                                        logger=self.logger)

        if self.verbose:
            if len(self.task_names) > 1:
                self.logger("\nMean across tasks")
                # If multi-task, compute mean over tasks and add to logs
                fetch = ("val_CE", "val_dice", "val_precision", "val_recall")
                for f in fetch:
                    res = np.mean([logs["%s_%s" % (name, f)] for name in self.task_names])
                    logs[f] = res
                    self.logger(("Mean %s for epoch %d:" %
                                 (f.split("_")[1], epoch)).ljust(30) + "%.4f" % res)
                self.logger("")


class ValDiceScores(Callback):
    """
    Similar to Validation, but working on an array of data instead of
    internally sampling from a validation sequence generator.

    On epoch end computes the mean dice coefficient and adds to following log
    entry:
    logs["val_dice"] = mean_dice
    """
    def __init__(self, validation_data, n_classes, batch_size=8, logger=None):
        """
        Args:
            validation_data: A tuple (X, y) of two ndarrays of validation data
                             and corresponding labels.
                             Any shape accepted by the model.
                             Labels must be integer targets (not one-hot)
            n_classes:       Number of classes, including background
            batch_size:      Batch size used for prediction
            logger:          An instance of a MultiPlanar Logger that prints to screen
                             and/or file
        """
        super().__init__()
        self.logger = logger or ScreenLogger()
        self.X_val, self.y_val = validation_data
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.scores = []

    def eval(self):
        pred = self.model.predict(self.X_val, self.batch_size, verbose=1)
        dices = dice_all(self.y_val,
                         pred.argmax(-1),
                         n_classes=self.n_classes,
                         ignore_zero=True)
        return dices

    def on_epoch_end(self, epoch, logs={}):
        scores = self.eval()
        mean_dice = scores.mean()
        s = "Mean dice for epoch %d: %.4f\nPr. class: %s" % (epoch,
                                                             mean_dice,
                                                             scores)
        self.logger(highlighted(s))
        self.scores.append(mean_dice)

        # Add to log
        logs["val_dice"] = mean_dice
