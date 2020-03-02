from mpunet.image.queue.base_queue import BaseQueue
from contextlib import contextmanager


class LazyQueue(BaseQueue):
    """
    Implements a queue-like object (same API interface as LoadQueue), but one
    that only loads data just-in-time when requested.
    This is useful for wrapping e.g. validation data in an object that behaves
    similar to the training queue object, but without consuming memory before
    needing to do validation.
    """
    def __init__(self, dataset, logger=None, **kwargs):
        """
        TODO
        Args:
            dataset:
            logger:
        """
        super(LazyQueue, self).__init__(
            dataset=dataset,
            logger=logger
        )
        self.logger("'Lazy' queue created:\n"
                    "  Dataset:      {}".format(self.dataset))
        self.logger("Images will be loaded just-in-time and unloaded when "
                    "not in use.".format(len(dataset)))

    @contextmanager
    def get_random_image(self):
        image = super().get_random_image()
        with image.loaded_in_context():
            yield image

    @contextmanager
    def get_image_by_idx(self, image_idx):
        image = super().get_image_by_idx(image_idx)
        with image.loaded_in_context():
            yield image

    @contextmanager
    def get_image_by_id(self, image_id):
        image = super().get_image_by_id(image_id)
        with image.loaded_in_context():
            yield image
