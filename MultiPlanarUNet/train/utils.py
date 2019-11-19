from MultiPlanarUNet.evaluate import loss_functions as custom_losses
from MultiPlanarUNet.evaluate import metrics as custom_metrics
from MultiPlanarUNet.utils import ensure_list_or_tuple
from MultiPlanarUNet.errors.implementation_change_errors import NotSparseError
from tensorflow.keras import optimizers, losses
from tensorflow.keras import metrics


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
        tf_funcs:     (module) A Tensorflow.keras module of losses or metrics
        custom_funcs: (module) A custom module or losses or metrics
        logger:       (Logger) A Logger object
        **kwargs:     (dict)   Parameters passed to all losses or metrics which
                               are represented by a class (i.e. not a function)

    Returns:
        A list of len(string_list) of initialized classes of losses or metrics
        or references to loss or metric functions.
    """
    initialized = []
    for func_or_class in ensure_list_or_tuple(string_list):
        if hasattr(tf_funcs, func_or_class):
            initialized.append(getattr(tf_funcs, func_or_class))
        else:
            import inspect
            func_or_class = getattr(custom_funcs, func_or_class)
            if inspect.isclass(func_or_class):
                func_or_class = func_or_class(logger=logger, **kwargs)
            initialized.append(func_or_class)
    return initialized


def init_losses(loss_string_list, logger=None, **kwargs):
    """
    Takes a list of strings each naming a loss function to return. The string
    name should correspond to a function or class that is an attribute of
    either the tensorflow.keras.losses or MultiPlanarUNet.evaluate.losses
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
