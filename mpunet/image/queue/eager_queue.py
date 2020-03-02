from mpunet.image.queue.base_queue import BaseQueue
from contextlib import contextmanager


class EagerQueue(BaseQueue):
    """
    Implements a queue-like object (same API interface as LoadQueue), but one
    that loads all data immediately when initialized.
    This is useful for wrapping a smaller collection of data in an object that
    behaves similar to the training queue object, but where all data is loaded
    up-front.
    """
    def __init__(self, dataset, logger=None, **kwargs):
        """
        TODO
        Args:
            dataset:
            logger:
        """
        super(EagerQueue, self).__init__(
            dataset=dataset,
            logger=logger
        )
        self.logger("'Eager' queue created:\n"
                    "  Dataset:      {}".format(self.dataset))
        self.logger("Preloading all {} images now... (eager)".format(len(dataset)))
        self.dataset.load()

    @staticmethod
    def check_loaded(image):
        if not image.is_loaded:
            raise RuntimeError("Some process unloaded image '{}'; this "
                               "is unexpected behaviour when using the "
                               "EagerQueue object, which expects all data to "
                               "be loaded at all times")
        return image

    def __iter__(self):
        for i in range(len(self.dataset.images)):
            with self.get_image_by_idx(i) as ss:
                yield ss

    @contextmanager
    def get_random_image(self):
        yield self.check_loaded(super().get_random_image())

    @contextmanager
    def get_image_by_idx(self, image_idx):
        yield self.check_loaded(super().get_image_by_idx(image_idx))

    @contextmanager
    def get_image_by_id(self, image_id):
        yield self.check_loaded(super().get_image_by_id(image_id))
