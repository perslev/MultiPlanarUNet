from mpunet.evaluate import loss_functions as custom_losses
from mpunet.evaluate import metrics as custom_metrics
from mpunet.utils import ensure_list_or_tuple
from mpunet.errors.implementation_change_errors import NotSparseError
from tensorflow.keras import optimizers, losses, metrics, activations
from tensorflow_addons import optimizers as addon_optimizers
from tensorflow_addons import activations as addon_activations


# Default error message to raise with non-sparse losses or metrics passed
SPARSE_ERR = NotSparseError("This implementation now requires integer targets "
                            "as opposed to one-hot encoded targets. "
                            "All metrics and loss functions should be named "
                            "'sparse_[org_name]' to reflect this in accordance"
                            " with the naming convention of TensorFlow.keras.")


def ensure_sparse(loss_and_metric_names):
    """
    Checks that 'sparse' is a substring of each string in a list of loss and/or
    metric names. Raises NotSparseError if one or more does not contain the
    substring.
    """
    for i, m in enumerate(loss_and_metric_names):
        if "sparse" not in m:
            raise SPARSE_ERR


def _init(string_list, tf_funcs, custom_funcs, logger=None, **kwargs):
    """
    Helper for 'init_losses' or 'init_metrics'.
    Please refer to their docstrings.

    Args:
        string_list:  (list)   List of strings, each giving a name of a metric
                               or loss to use for training. The name should
                               refer to a function or class in either tf_funcs
                               or custom_funcs modules.
        tf_funcs:     (module) A Tensorflow.keras module of losses or metrics,
                               or a list of various modules to look through.
        custom_funcs: (module) A custom module or losses or metrics
        logger:       (Logger) A Logger object
        **kwargs:     (dict)   Parameters passed to all losses or metrics which
                               are represented by a class (i.e. not a function)

    Returns:
        A list of len(string_list) of initialized classes of losses or metrics
        or references to loss or metric functions.
    """
    initialized = []
    tf_funcs = ensure_list_or_tuple(tf_funcs)
    for func_or_class in ensure_list_or_tuple(string_list):
        modules_found = list(filter(None, [getattr(m, func_or_class, None)
                                           for m in tf_funcs]))
        if modules_found:
            initialized.append(modules_found[0])  # return the first found
        else:
            # Fall back to look in custom module
            initialized.append(getattr(custom_funcs, func_or_class))
    return initialized


def init_losses(loss_string_list, logger=None, **kwargs):
    """
    Takes a list of strings each naming a loss function to return. The string
    name should correspond to a function or class that is an attribute of
    either the tensorflow.keras.losses or mpunet.evaluate.losses
    modules.

    The returned values are either references to the loss functions to use, or
    initialized loss classes for some custom losses (used when the loss
    requires certain parameters to be set).

    Args:
        loss_string_list: (list)   A list of strings each naming a loss to
                                   return
        logger:           (Logger) An optional Logger object
        **kwargs:         (dict)   Parameters that will be passed to all class
                                   loss functions (i.e. not to functions)

    Returns:
        A list of length(loss_string_list) of loss functions or initialized
        classes
    """
    return _init(
        loss_string_list, losses, custom_losses, logger, **kwargs
    )


def init_metrics(metric_string_list, logger=None, **kwargs):
    """
    Same as 'init_losses', but for metrics.
    Please refer to the 'init_losses' docstring.
    """
    return _init(
        metric_string_list, metrics, custom_metrics, logger, **kwargs
    )


def init_optimizer(optimizer_string, logger=None, **kwargs):
    """
    Same as 'init_losses', but for optimizers.
    Please refer to the 'init_losses' docstring.
    """
    optimizer = _init(
        optimizer_string,
        tf_funcs=[optimizers, addon_optimizers],
        custom_funcs=None,
        logger=logger
    )[0]
    return optimizer(**kwargs)


def init_activation(activation_string, logger=None, **kwargs):
    """
    Same as 'init_losses', but for optimizers.
    Please refer to the 'init_losses' docstring.
    """
    activation = _init(
        activation_string,
        tf_funcs=[activations, addon_activations],
        custom_funcs=None,
        logger=logger
    )[0]
    return activation
