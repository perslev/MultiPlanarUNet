import numpy as np
from .base_sequence import BaseSequence


class RandomDataFeeder(BaseSequence):
    def __init__(self, X, y, batch_size):
        super().__init__()
        self.X = X
        self.y = y
        self.bs = batch_size
        self.inds = np.arange(len(self.X))

    def __len__(self):
        n_samples = len(self.X)
        return int(np.ceil(n_samples / self.bs))

    def __getitem__(self, item):
        inds = np.random.choice(self.inds, size=self.bs, replace=False)
        return self.X[inds], self.y[inds]
