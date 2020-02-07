import numpy as np


def one_hot_encode_y(y, n_classes=135):
    y = np.asarray(y)
    if n_classes == 1:
        return reshape_add_axis(y, len(y.shape)-1)
    else:
        from keras.utils import to_categorical
        shape = y.shape
        y = to_categorical(y, num_classes=n_classes).astype(np.uint8)
        y = y.reshape(shape + (n_classes,))
    return y


def reshape_add_axis(X, im_dims=2, n_channels=1):
    X = np.asarray(X)
    if X.shape[-1] != n_channels:
        # Reshape
        X = X.reshape(X.shape + (n_channels,))
    if len(X.shape) == im_dims+1:
        X = X.reshape((1,) + X.shape)
    return X
