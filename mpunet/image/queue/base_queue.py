import numpy as np
from mpunet.logging.default_logger import ScreenLogger


class BaseQueue:
    """
    The base queue object defines the Queue API and stores basic attributes
    used across all queue objects
    The BaseQueue should normally not be initialized directly.
    """
    def __init__(self, dataset, logger=None):
        """
        TODO
        Args:
            datasets:
            logger:
        """
        self.dataset = dataset
        self.logger = logger or ScreenLogger()

    @property
    def all_loaded(self):
        raise NotImplemented

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i in range(len(self.dataset.images)):
            yield self.get_image_by_idx(i)

    def __getitem__(self, idx):
        return self.get_image_by_idx(idx)

    def get_image_iterator(self, max_load=None):
        load_inds = np.arange(len(self))
        if max_load and max_load < len(self):
            load_inds = np.random.choice(load_inds, max_load, False)
        for idx in load_inds:
            yield self.get_image_by_idx(idx)

    def get_random_image(self):
        return np.random.choice(self.dataset.images, 1)[0]

    def get_image_by_idx(self, image_idx):
        return self.dataset.images[image_idx]

    def get_image_by_id(self, image_id):
        return self.dataset.id_to_image[image_id]
