import numpy as np
from mpunet.image.queue import BaseQueue, LoadingPool
from queue import Queue, Empty
from contextlib import contextmanager


class LimitationQueue(BaseQueue):
    """
    Implements an image loading queue.
    Stores a reference to a ImagePairLoader object.

    Using the methods get_random_image method, this method tracks the number of
    times a ImagePair object has been accessed, and when exceeding a threshold,
    unloads it and loads a random ImagePair from the dataset.
    """
    def __init__(self,
                 dataset,
                 max_loaded=25,
                 num_access_before_reload=50,
                 preload_now=True,
                 await_preload=True,
                 loading_pool=None,
                 n_load_jobs=5,
                 logger=None,
                 **kwargs):
        """
        Initialize a LoadQueue object from a ImagePairLoader object

        Args:
            dataset:                    (list) A ImagePairLoader object
            max_loaded:                 (int)  Number of ImagePair objects in
                                               a dataset that will be loaded
                                               at a given time.
            num_access_before_reload:   (int)  Number of times a ImagePair obj
                                               can be accessed be
                                               get_random_image or
                                               a unload is invoked and a new
                                               data point is loaded.
            preload_now:   TODO
            image_loader:  TODO
            n_load_jobs:   TODO
            logger:        TODO
        """
        super(LimitationQueue, self).__init__(
            dataset=dataset,
            logger=logger
        )
        self.max_loaded = min(max_loaded or len(dataset), len(dataset))
        self.num_access_before_reload = num_access_before_reload or 50

        # Queues of loaded and non-loaded objects
        self.loaded_queue = Queue(maxsize=self.max_loaded)
        self.non_loaded_queue = Queue(maxsize=len(dataset))

        # Fill non-loaded queue in random order
        inds = np.arange(len(dataset))
        np.random.shuffle(inds)
        for i in inds:
            self.non_loaded_queue.put(self.dataset.images[i])

        # Setup load thread pool
        self.loading_pool = loading_pool or LoadingPool(
            n_threads=n_load_jobs
        )
        # Register this dataset to become updated with new loaded images
        # from the StudyLoader thread.
        self.loading_pool.register_dataset(
            dataset_id=self.dataset.identifier,
            load_put_function=self._add_loaded_to_queue,
            error_put_function=self._load_error_callback,
        )

        # Increment counters to random off-set points for the first images
        self.max_offset = int(self.num_access_before_reload * 0.75)
        self.n_offset = self.max_loaded

        self.logger("'Limitation' queue created:\n"
                    "  Dataset:      {}\n"
                    "  Max loaded:   {}\n"
                    "  Reload limit: {}".format(
            self.dataset, self.max_loaded, self.num_access_before_reload
        ))
        if preload_now:
            # Load specified number of obj and populate access count dict
            self.preload(await_preload)

    def preload(self, await_preload=True):
        """
        TODO

        Returns:

        """
        # Set the number of loaded objects to 'max_loaded_per_dataset'
        self.logger("Adding {} ImagePair objects from "
                    "{} to load queue".format(self.max_loaded,
                                              self.dataset.identifier))
        if self.dataset.n_loaded != 0 or self.loaded_queue.qsize() != 0:
            raise RuntimeError("Dataset {} seems to have already been "
                               "loaded. Do not load any data before "
                               "passing the ImagePairLoader object "
                               "to the queue class. Only call "
                               "LoadQueue.preload once."
                               "".format(self.dataset.identifier))
        self._add_images_to_load_queue(num=self.max_loaded)
        if await_preload:
            self.logger("... awaiting preload")
            self.loading_pool.join()
            self.logger("Preload complete.")

    def load_queue_too_full(self, max_fraction=0.33):
        return self.loading_pool.qsize() > \
               self.loading_pool.maxsize*max_fraction

    def _warn_access_limit(self, min_fraction=0.10):
        qsize = self.loaded_queue.qsize()
        if qsize == 0:
            self.logger.warn("Study ID queue for dataset {} seems to"
                             " block. This might indicate a data loading "
                             "bottleneck.".format(self.dataset.identifier))
        elif qsize <= self.max_loaded*min_fraction:
            self.logger.warn("Dataset {}: Loaded queue in too empty "
                             "(qsize={}, maxsize={})"
                             .format(self.dataset.identifier, qsize,
                                     self.max_loaded))

    @contextmanager
    def get_image_by_id(self, image_id):
        raise NotImplementedError

    @contextmanager
    def get_image_by_idx(self, image_idx):
        raise NotImplementedError

    @contextmanager
    def get_random_image(self):
        """
        TODO

        Returns:

        """
        with self.loading_pool.thread_lock:
            self._warn_access_limit()
        timeout_s = 5
        try:
            image_pair, n_accesses = self.loaded_queue.get(timeout=timeout_s)
        except Empty as e:
            raise Empty("Could not get ImagePair from dataset {} with "
                        "timeout of {} seconds. Consider increasing the "
                        "number of load threads / max loaded per dataset /"
                        " access threshold".format(self.dataset.identifier,
                                                   timeout_s)) from e
        try:
            yield image_pair
        finally:
            self._release_image(image_pair, n_accesses)

    def _add_images_to_load_queue(self, num=1):
        """
        TODO

        Args:
            num:

        Returns:

        """
        for _ in range(num):
            image = self.non_loaded_queue.get_nowait()  # Should never block!
            if image.is_loaded:
                raise RuntimeWarning("Image {} in dataset {} seems to be "
                                     "already loaded, but it was fetched from "
                                     "the self.non_loaded_queue queue. This "
                                     "could be an implementation error!"
                                     "".format(image.identifier,
                                               self.dataset.identifier))
            self.loading_pool.add_image_to_load_queue(image, self.dataset.identifier)

    def _add_loaded_to_queue(self, image_pair):
        """

        Args:
            image_id:

        Returns:

        """
        if self.n_offset >= 0:
            offset = np.random.randint(0, self.max_offset)
            self.n_offset -= 1
        else:
            offset = 0
        self.loaded_queue.put((image_pair, offset))

    def _load_error_callback(self, image_pair, *args, **kwargs):
        self.logger.warn("Load error on image {}".format(image_pair))
        self._add_images_to_load_queue(num=1)

    def _release_image(self, image_pair, n_accesses):
        """
        TODO

        Args:
            image_pair:
            n_accesses:

        Returns:

        """
        if n_accesses >= self.num_access_before_reload:
            # Unload, add to unloaded queue, start loading new image
            image_pair.unload()
            self.non_loaded_queue.put(image_pair)
            self._add_images_to_load_queue(num=1)
        else:
            self.loaded_queue.put((image_pair, n_accesses+1))
