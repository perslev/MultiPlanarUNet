from tensorflow.keras.utils import Sequence
import numpy as np
from MultiViewUNet.logging import ScreenLogger


class PointSequence(Sequence):
    def __init__(self, X, y, n_classes, batch_size, logger=None):
        self.logger = logger or ScreenLogger()

        self.X = X
        self.y = y
        self.n_classes = n_classes
        self.batch_size = batch_size

        # Get list of foreground pixels
        pred = self.y.argmax(-1) if n_classes > 1 else (self.y > 0.5).astype(np.uint8)
        self.is_fg = np.where(pred != 0)[0]
        self.no_fg = np.delete(np.arange(len(self.X)), self.is_fg, 0)
        self.fg_fraction = len(self.is_fg) / len(self.X)
        self.n_fg = max(int(np.ceil(self.fg_fraction * self.batch_size)), 1) * 10

        self._log()

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        points = np.empty(shape=(self.batch_size,) + self.X.shape[1:],
                          dtype=self.X.dtype)
        target = np.empty(shape=(self.batch_size,) + self.y.shape[1:],
                          dtype=self.y.dtype)

        # Sample fg pixels
        fg_samples = self.is_fg[np.random.randint(0, len(self.is_fg), self.n_fg)]
        points[:self.n_fg] = self.X[fg_samples]
        target[:self.n_fg] = self.y[fg_samples]

        # Sample bg pixels
        bg_samples = self.no_fg[np.random.randint(0, len(self.no_fg),
                                                  self.batch_size-self.n_fg)]
        points[self.n_fg:] = self.X[bg_samples]
        target[self.n_fg:] = self.y[bg_samples]

        return points, target

    def _log(self):
        self.logger("Points shape:    %s" % list(self.X.shape))
        self.logger("Targets shape:   %s" % list(self.y.shape))
        self.logger("FG fraction:     %.3f" % self.fg_fraction)
        self.logger("N fg points:     %i" % self.n_fg)
