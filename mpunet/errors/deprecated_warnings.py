from mpunet.logging import ScreenLogger


def warn_sparse_param(logger):
    logger = logger or ScreenLogger
    sparse_err = "mpunet 0.1.3 or higher requires integer targets" \
                 " as opposed to one-hot encoded targets. Setting the 'sparse'" \
                 " parameter no longer has any effect and may not be allowed" \
                 " in future versions."
    logger.warn(sparse_err)
