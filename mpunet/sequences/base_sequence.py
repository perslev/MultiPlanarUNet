from multiprocessing import current_process
from tensorflow.keras.utils import Sequence
from abc import ABC, abstractmethod
import numpy as np


class BaseSequence(Sequence, ABC):
    def __init__(self):
        super().__init__()

        # A dictionary mapping process names to whether the process has been
        # seeded
        self.is_seeded = {}

    def seed(self):
        # If multiprocessing the processes will inherit the RNG state of the
        # main process - here we reseed each process once so that the batches
        # are randomly generated across processes
        pname = current_process().name
        if pname not in self.is_seeded or not self.is_seeded[pname]:
            # Re-seed this process
            # If threading this will just re-seed MainProcess
            np.random.seed()
            self.is_seeded[pname] = True

    @abstractmethod
    def __len__(self):
        raise NotImplemented

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplemented

    def __call__(self):
        import tensorflow as tf
        def tensor_iter():
            """ Iterates the dataset, converting numpy arrays to tensors """
            for x, y, w in self:
                yield (tf.convert_to_tensor(x),
                       tf.convert_to_tensor(y),
                       tf.convert_to_tensor(w))
        return tensor_iter()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
