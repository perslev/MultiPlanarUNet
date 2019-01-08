from .scaling import apply_scaling, get_scaler
from .input_prep import reshape_add_axis, one_hot_encode_y


def get_preprocessing_func(model):
    """
    Takes a model name (string) and returns a preparation function.

    Args:
        model: String representation of a MultiPlanarUNet.models model class

    Returns:
        A MultiPlanarUNet.preprocessing.data_preparation_funcs function
    """
    from MultiPlanarUNet.models import PREPARATION_FUNCS
    if model in PREPARATION_FUNCS:
        return PREPARATION_FUNCS[model]
    else:
        raise ValueError("Unsupported model type '%s'. "
                         "Supported models: %s" % (model,
                                                   PREPARATION_FUNCS.keys()))
