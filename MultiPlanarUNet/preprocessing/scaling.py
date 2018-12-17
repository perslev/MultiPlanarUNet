import sklearn.preprocessing as preprocessing
import numpy as np


def get_scaler(scaler, *args, **kwargs):
    scaler = preprocessing.__dict__[scaler]
    return MultiChannelScaler(scaler=scaler, *args, **kwargs)


def apply_scaling(X, scaler):

    # Get scaler
    multi_scaler = get_scaler(scaler)

    # Fit and apply transformation
    return multi_scaler.fit_transform(X)


class MultiChannelScaler(object):
    def __init__(self, scaler, *args, **kwargs):
        # Store scaler class and passed parameters
        self.scaler_class = scaler
        self.scaler_args = args
        self.scaler_kwargs = kwargs

        # Store list of initialized scalers fit to each channel
        self.scalers = []

        # Store number of channels
        self.n_channels = None

    def fit(self, X, *args, **kwargs):
        if X.ndim != 4:
            raise ValueError("Invalid shape for X (%s)" % X.shape)

        # Set number of channels
        self.n_channels = X.shape[-1]

        scalers = []
        for i in range(self.n_channels):
            sc = self.scaler_class(*self.scaler_args, **self.scaler_kwargs)
            sc.fit(X[..., i].reshape(-1, 1), *args, **kwargs)
            scalers.append(sc)

        self.scalers = scalers
        return self

    def transform(self, X, *args, **kwargs):
        if X.shape[-1] != self.n_channels:
            raise ValueError("Invalid input of dimension %i, expected "
                             "last axis with %i channels" % (X.ndim,
                                                             self.n_channels))

        # Prepare volume like X to store results
        transformed = np.empty_like(X)
        for i in range(self.n_channels):
            scl = self.scalers[i]
            s = scl.transform(X[..., i].reshape(-1, 1), *args, **kwargs)
            transformed[..., i] = s.reshape(X.shape[:-1])

        return transformed

    def fit_transform(self, X, *args, **kwargs):
        self.fit(X, *args, **kwargs)
        return self.transform(X, *args, **kwargs)
