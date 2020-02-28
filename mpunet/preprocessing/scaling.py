import sklearn.preprocessing as preprocessing
import numpy as np


def assert_scaler(scaler):
    return scaler in preprocessing.__dict__


def get_scaler(scaler, *args, ignore_less_eq=None, **kwargs):
    scaler = preprocessing.__dict__[scaler]
    return MultiChannelScaler(scaler=scaler,
                              ignore_less_eq=ignore_less_eq,
                              *args, **kwargs)


def apply_scaling(X, scaler, ignore_less_eq=None):
    # Get scaler
    multi_scaler = get_scaler(scaler, ignore_less_eq=ignore_less_eq)

    # Fit and apply transformation
    return multi_scaler.fit_transform(X)


class MultiChannelScaler(object):
    def __init__(self, scaler, *args, ignore_less_eq=None, **kwargs):
        # Store scaler class and passed parameters
        self.scaler_class = scaler
        self.scaler_args = args
        self.scaler_kwargs = kwargs
        self.ignore_less_eq = ignore_less_eq

        # Store list of initialized scalers fit to each channel
        self.scalers = []

        # Store number of channels
        self.n_channels = None

    def __str__(self):
        return "MultiChannelScaler(scaler_class='{}', ignore_less_eq={})".format(
            self.scaler_class.__name__,
            self.ignore_less_eq
        )

    def __repr__(self):
        return str(self)

    def fit(self, X, *args, **kwargs):
        if X.ndim != 4:
            raise ValueError("Invalid shape for X (%s)" % X.shape)

        # Set number of channels
        self.n_channels = X.shape[-1]

        if self.ignore_less_eq is not None:
            if not isinstance(self.ignore_less_eq, (list, tuple, np.ndarray)):
                self.ignore_less_eq = [self.ignore_less_eq] * self.n_channels
            if not len(self.ignore_less_eq) == self.n_channels:
                raise ValueError("'ignore_less_eq' should be a list of length "
                                 "'n_channels'. Got {} for n_channels={}".format(
                    self.ignore_less_eq, self.n_channels
                ))

        scalers = []
        for i in range(self.n_channels):
            sc = self.scaler_class(*self.scaler_args, **self.scaler_kwargs)
            xs = X[..., i]
            if self.ignore_less_eq is not None:
                xs = xs[np.where(xs > self.ignore_less_eq[i])]
            sc.fit(xs.reshape(-1, 1), *args, **kwargs)
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
