"""
Mathias Perslev
MSc Bioinformatics

University of Copenhagen
November 2017
"""

import numpy as np
import tensorflow as tf


def dice(y_true, y_pred, smooth=1.0):
    """
    Calculates the Soerensen dice coefficient between two binary sets
    """
    # Flatten and bool the ground truth and predicted labels
    s1 = np.array(y_true).flatten().astype(np.bool)
    s2 = np.array(y_pred).flatten().astype(np.bool)

    # Calculate dice
    return (smooth + 2 * np.logical_and(s1, s2).sum()) \
           / (smooth + s1.sum() + s2.sum())


def dice_all(y_true, y_pred, smooth=1.0, n_classes=None, ignore_zero=True,
             skip_if_no_y=False):
    """
    Calculates the Soerensen dice coefficients for all unique classes
    """
    # Get array of unique classes in true label array
    if n_classes is None:
        classes = np.unique(y_true)
    else:
        classes = np.arange(max(2, n_classes))
    # Ignore background class?
    if ignore_zero:
        classes = classes[np.where(classes != 0)]

    # Calculate dice for all targets
    dice_coeffs = np.empty(shape=classes.shape, dtype=np.float32)
    dice_coeffs.fill(np.nan)
    for idx, _class in enumerate(classes):
        s1 = y_true == _class
        if skip_if_no_y and not np.any(s1):
            continue
        s2 = y_pred == _class

        if np.any(s1) or np.any(s2):
            d = dice(s1, s2, smooth=smooth)
            dice_coeffs[idx] = d
    return dice_coeffs


def class_wise_kappa(true, pred, n_classes=None, ignore_zero=True):
    from sklearn.metrics import cohen_kappa_score
    if n_classes is None:
        classes = np.unique(true)
    else:
        classes = np.arange(max(2, n_classes))
    # Ignore background class?
    if ignore_zero:
        classes = classes[np.where(classes != 0)]

    # Calculate kappa for all targets
    kappa_scores = np.empty(shape=classes.shape, dtype=np.float32)
    kappa_scores.fill(np.nan)
    for idx, _class in enumerate(classes):
        s1 = true == _class
        s2 = pred == _class

        if np.any(s1) or np.any(s2):
            kappa_scores[idx] = cohen_kappa_score(s1, s2)
    return kappa_scores


def one_class_dice(y_true, y_pred, smooth=1.0):
    # Predict
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    return (smooth + 2.0 * tf.reduce_sum(y_true * y_pred)) / (smooth + tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


def sparse_fg_recall(y_true, y_pred, bg_class=0):
    # Get MAP estimates
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.cast(tf.reshape(tf.argmax(y_pred, axis=-1), [-1]), tf.int32)

    # Remove background
    mask = tf.not_equal(y_true, bg_class)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def sparse_mean_fg_f1(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)

    # Get confusion matrix
    cm = tf.confusion_matrix(tf.reshape(y_true, [-1]),
                             tf.reshape(y_pred, [-1]))

    # Get precisions
    TP = tf.diag_part(cm)
    precisions = TP / tf.reduce_sum(cm, axis=0)

    # Get recalls
    TP = tf.diag_part(cm)
    recalls = TP / tf.reduce_sum(cm, axis=1)

    # Get F1s
    f1s = (2 * precisions * recalls) / (precisions + recalls)

    return tf.reduce_mean(f1s[1:])


def sparse_mean_fg_precision(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)

    # Get confusion matrix
    cm = tf.confusion_matrix(tf.reshape(y_true, [-1]),
                             tf.reshape(y_pred, [-1]))

    # Get precisions
    TP = tf.diag_part(cm)
    precisions = TP / tf.reduce_sum(cm, axis=0)

    return tf.reduce_mean(precisions[1:])


def sparse_mean_fg_recall(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)

    # Get confusion matrix
    cm = tf.confusion_matrix(tf.reshape(y_true, [-1]),
                             tf.reshape(y_pred, [-1]))

    # Get precisions
    TP = tf.diag_part(cm)
    recalls = TP / tf.reduce_sum(cm, axis=1)

    return tf.reduce_mean(recalls[1:])


def sparse_fg_precision(y_true, y_pred, bg_class=0):
    # Get MAP estimates
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.cast(tf.reshape(tf.argmax(y_pred, axis=-1), [-1]), tf.int32)

    # Remove background
    mask = tf.not_equal(y_pred, bg_class)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))
