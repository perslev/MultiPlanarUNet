from tensorflow.keras.utils import Sequence
from mpunet.logging import ScreenLogger


class MultiTaskSequence(Sequence):
    """
    A Sequence which simply wraps around multiple other Sequences, calling each
    of them to create batches for each Sequence (typically different 'tasks')
    and returns three lists, stroing iamges, labels and weights, each of
    length equal to the number of wrapped Sequencers.
    """
    def __init__(self, sequencers, task_names, logger=None):
        super().__init__()
        self.logger = logger or ScreenLogger()
        self.task_names = task_names
        self.sequencers = sequencers
        self.log()

        # Redirect setattrs to the sub-sequences
        self.redirect = True

    def log(self):
        self.logger("--- MultiTaskSequence sequencer --- ",
                    print_calling_method=True)
        self.logger("N tasks:  %i" % len(self.sequencers))

    def __iter__(self):
        for s in self.sequencers:
            yield s

    @property
    def n_samples(self):
        return sum([s.n_samples for s in self])

    def __getattr__(self, item):
        # Fetch item from all sub-sequences
        return [getattr(s, item) for s in self.sequencers]

    def __setattr__(self, key, value):
        if self.__dict__.get("redirect", False):
            # Fetch item from all sub-sequences
            for s in self:
                # Set on all sub-sequences
                setattr(s, key, value)
        else:
            # Set on self
            self.__dict__[key] = value

    def __len__(self):
        """Number of batch in the Sequence.

        Returns:
            The number of batches in the Sequence.
        """
        return sum([len(seq) for seq in self.sequencers])

    def __getitem__(self, index):
        batches_x, batches_y, batches_w = [], [], []
        for seq in self.sequencers:
            x, y, w = seq[0]
            batches_x.append(x)
            batches_y.append(y)
            batches_w.append(w)

        return batches_x, batches_y, batches_w
