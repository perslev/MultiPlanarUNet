from tensorflow.keras.utils import Sequence
from multiprocessing import current_process
import numpy as np


class BaseSequence(Sequence):
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
