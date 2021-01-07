import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from collections import defaultdict
from mpunet.logging import ScreenLogger
from mpunet.evaluate.metrics import dice_all
from mpunet.utils import highlighted, ensure_list_or_tuple


class Validation(Callback):
    """
    Validation computation callback.
    Samples a number of validation batches from a mpunet.sequence object
    and computes for all tasks:
        - Batch-wise validation loss + metrics as specified in model.compile
        - Epoch-wise pr-class and average precision
        - Epoch-wise pr-class and average recall
        - Epoch-wise pr-class and average dice coefficients
    ... and adds all results to the log dict
    Note: The purpose of this callback over the default tf.keras evaluation
    mechanism is to calculate certain metrics over the entire epoch of data as
    opposed to averaged batch-wise computations.
    Also, it supports multi-dataset/multi-task evaluation
    """
    def __init__(self, val_sequence, steps, logger=None, verbose=True,
                 ignore_class_zero=True):
        """
        Args:
            val_sequence: A mpunet.sequence object from which validation
                          batches can be sampled via its __getitem__ method.
            steps:        Numer of batches to sample from val_sequences in each
                          validation epoch
            logger:       An instance of a MultiPlanar Logger that prints to screen
                          and/or file
            verbose:      Print progress to screen - OBS does not use Logger
            ignore_class_zero: TODO
        """
        super().__init__()
        self.logger = logger or ScreenLogger()
        self.data = val_sequence
        self.steps = steps
        self.verbose = verbose
        self.ignore_bg = ignore_class_zero
        self.print_round = 3
        self.log_round = 4
        self._supports_tf_logs = True  # ensures correct logs passed from tf.keras

        self.n_classes = self.data.n_classes
        if isinstance(self.n_classes, int):
            self.task_names = [""]
            self.n_classes = [self.n_classes]
        else:
            self.task_names = self.data.task_names

    @staticmethod
    def _compute_dice(tp, rel, sel):
        """
        TODO

        :param tp:
        :param rel:
        :param sel:
        :return:
        """
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
    def _count_cm_elements_from_queue(queue,
                                      steps,
                                      TPs,
                                      relevant,
                                      selected,
                                      task_names,
                                      n_classes_list,
                                      lock):
        """
        TODO

        :param queue:
        :param steps:
        :param TPs:
        :param relevant:
        :param selected:
        :param task_names:
        :param n_classes_list:
        :param lock:
        :return:
        """
        for _ in range(steps):
            # Get prediction and true labels from prediction queue
            pred, true = queue.get(block=True)
            for p, y, task_name, n_classes in zip(pred, true, task_names,
                                                  n_classes_list):
                # Argmax and CM elements
                if not isinstance(p, np.ndarray):
                    p = p.numpy()
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
            queue.task_done()

    def evalaute(self):
        """
        TODO

        :return:
        """
        # Get tensors to run and their names
        if hasattr(self.model, "loss_functions"):
            metrics = self.model.loss_functions + self.model.metrics
        else:
            metrics = self.model.metrics
        metrics_names = self.model.metrics_names
        self.model.reset_metrics()
        assert len(metrics_names) == len(metrics)

        # Prepare dictionaries for storing pr. task metric results
        TPs, relevant, selected, batch_wise_metrics = {}, {}, {}, {}
        for task_name, n_classes in zip(self.task_names, self.n_classes):
            TPs[task_name] = np.zeros(shape=(n_classes,), dtype=np.uint64)
            relevant[task_name] = np.zeros(shape=(n_classes,), dtype=np.uint64)
            selected[task_name] = np.zeros(shape=(n_classes,), dtype=np.uint64)
            batch_wise_metrics[task_name] = defaultdict(list)

        # Prepare queue and thread for computing counts
        from queue import Queue
        from threading import Thread
        count_queue = Queue(maxsize=self.steps)
        count_thread = Thread(target=self._count_cm_elements_from_queue,
                              daemon=True,
                              args=[count_queue, self.steps, TPs, relevant,
                                    selected, self.task_names, self.n_classes,
                                    Lock()])
        count_thread.start()

        # Fetch validation batches from the generator(s)
        pool = ThreadPoolExecutor(max_workers=3)
        batches = pool.map(self.data.__getitem__, np.arange(self.steps))

        # Predict on all
        self.logger("")
        for i, (X, y, _) in enumerate(batches):
            if self.verbose:
                print("   Validation: %i/%i" % (i+1, self.steps),
                      end="\r", flush=True)
            X = ensure_list_or_tuple(X)
            y = ensure_list_or_tuple(y)

            # Predict and put values in the queue for counting
            pred = self.model.predict_on_batch(X)
            pred = ensure_list_or_tuple(pred)
            count_queue.put([pred, y])

            for p_task, y_task, task in zip(pred, y, self.task_names):
                # Run all metrics
                for metric, name in zip(metrics, metrics_names):
                    m = tf.reduce_mean(metric(y_task, p_task))
                    batch_wise_metrics[task][name].append(m.numpy())
        pool.shutdown(wait=True)

        # Compute the mean over batch-wise metrics
        mean_batch_wise_metrics = {}
        for task in self.task_names:
            mean_batch_wise_metrics[task] = {}
            for metric in metrics_names:
                ms = batch_wise_metrics[task][metric]
                mean_batch_wise_metrics[task][metric] = np.mean(ms)
        self.model.reset_metrics()
        self.logger("")

        # Terminate count thread
        print("Waiting for counting queue to terminate...\n")
        count_thread.join()
        count_queue.join()

        # Compute per-class metrics (dice+precision+recall)
        class_wise_metrics = {}
        for task in self.task_names:
            precisions, recalls, dices = self._compute_dice(tp=TPs[task],
                                                            sel=relevant[task],
                                                            rel=selected[task])
            if self.ignore_bg:
                precisions[0] = np.nan
                recalls[0] = np.nan
                dices[0] = np.nan
            class_wise_metrics[task] = {
                "dice": dices,
                "recall": recalls,
                "precision": precisions
            }
        return class_wise_metrics, mean_batch_wise_metrics

    def _print_val_results(self,
                           class_wise_metrics,
                           batch_wise_metrics,
                           epoch,
                           task_name,
                           classes):
        """
        TODO

        :param class_wise_metrics:
        :param batch_wise_metrics:
        :param epoch:
        :param task_name:
        :param classes:
        :return:
        """
        # We add them to a pd dataframe just for the pretty print output
        index = ["mean"] + ["cls %i" % i for i in classes]
        columns = list(batch_wise_metrics.keys()) + list(class_wise_metrics.keys())
        df = pd.DataFrame(data={c: [np.nan] * len(index) for c in columns},
                          index=index)

        # Fill the df with metrics
        for m_name, value in batch_wise_metrics.items():
            df.loc['mean', m_name] = value
        for m_name, values in class_wise_metrics.items():
            values = [np.nanmean(values)] + list(values)
            df.loc[:, m_name] = values

        # Print the df to screen
        s = "Validation Results for epoch %i" % epoch
        self.logger(highlighted((("[%s]" % task_name) if task_name else "") + s))
        print_string = df.round(self.print_round).T.to_string()
        self.logger(print_string.replace("NaN", "  -") + "\n")

    def on_epoch_end(self, epoch, logs={}):
        # Predict and get CM
        class_wise_metrics, mean_batch_wise_metrics = self.evalaute()
        for n_classes, name in zip(self.n_classes, self.task_names):
            classes = np.arange(n_classes)
            n = (name + "_") if len(self.task_names) > 1 else ""

            # Add batch-wise metrics to log
            for m_name, value in mean_batch_wise_metrics[name].items():
                logs[f"{n}val_{m_name}"] = value.round(self.log_round)
            # Add mean of class-wise metrics to log
            for m_name, values in class_wise_metrics[name].items():
                logs[f"{n}val_{m_name}"] = np.nanmean(values)

            if self.verbose:
                self._print_val_results(class_wise_metrics=class_wise_metrics[name],
                                        batch_wise_metrics=mean_batch_wise_metrics[name],
                                        epoch=epoch,
                                        task_name=name,
                                        classes=classes)

        if len(self.task_names) > 1:
            # Print cross-dataset mean values
            if self.verbose:
                self.logger(highlighted(f"[ALL DATASETS] Means Across Classes"
                                        f" for Epoch {epoch}"))
            fetch = ("val_dice", "val_precision", "val_recall")
            m_fetch = tuple(["val_" + s for s in self.model.metrics_names])
            to_print = {}
            for f in fetch + m_fetch:
                scores = [logs["%s_%s" % (name, f)] for name in self.task_names]
                res = np.mean(scores)
                logs[f] = res.round(self.log_round)  # Add to log file
                to_print[f.split("_")[-1]] = list(scores) + [res]
            if self.verbose:
                df = pd.DataFrame(to_print)
                df.index = self.task_names + ["mean"]
                self.logger(df.round(self.print_round))
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
