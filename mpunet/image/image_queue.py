from threading import Thread, Event
from contextlib import contextmanager
from queue import Queue
import numpy as np
import time


def _start(queue, stop_event):
    """
    Thread target function. Starts the ImageQueue._populate function which runs
    indefinitely until stop_event is set.

    Args:
        queue:      A reference to the ImageQueue object onto which the threads
                    apply.
        stop_event: An even that can be set in the main thread to stop
                    population of the ImageQueue
    """
    while not stop_event.is_set():
        queue._populate()


class ImageQueue(object):
    """
    Queue object handling loading ImagePair data from disk, preprocessing those
    images and adding them to a queue from which mpunet.sequence objects
    can pull them for training

    The queue will store a maximum number of loaded ImagePairs at a given time.
    When the queue is not full, one or more threads will perform the following
    sequence of actions in parallel to populate the queue:

    1) Select a random image from the ImagePairLoader object
    2) An entry function is called on the ImagePair that will perform some
       preprocessing operations
    3) The image is added to the queue

    Whenever 1) is performed, a sample may with a probability managed by
    attribute 'self.load_new_prob' instead re-add an image already loaded,
    processed and added to the queue. This process is much faster, and will be
    performed with a higher probability if the queue is being emptied too fast
    for the population to keep up.

    ImagePairs are pulled from the queue using the context manager method 'get':
    ...
    with image_queue.get() as image:
        # Do something with 'image', an ImagePair object
        f(image)
    ...
    When the with statement exits, one of two things happens in ImageQueue.get:
        1) If the ImagePair no longer exists in the queue, the ImageQueue
           invokes an exit function on the ImagePair. This function will
           usually free memory.
        2) If the ImagePair is still in queue - which may happen when the same
           image is re-added to queue to prevent queue exhaustion - the exit
           function is NOT invoked.

    TODO: Race conditions may occur in updating/referencing the dictionary
          storing the number of times images are currently loaded in queue,
          during executing of entry and exit functions and during calculation
          of load_new_prob.
          Update to a more robust queuing approach.
          Move to a TensorFlow queue based system perhaps?
    """
    def __init__(self, max_queue_size, image_pair_loader, entry_func=None,
                 entry_func_kw=None, exit_func=None, exit_func_kw=None):
        """
        Args:
            max_queue_size:    Int, the maximum number of ImagePair objects
                               to store in the queue at a given time
            image_pair_loader: The ImagePairLoader object from which images are
                               fetched.
            entry_func:        String giving name of method to call on the
                               ImagePair object at queue entry time.
            entry_func_kw:     Dict, keyword arguments to supply to entry_func
            exit_func:         String giving name of method to call on the
                               ImagePair object at queue exit time.
            exit_func_kw:      Dict, keyword arguments to supply to exit_func
        """
        # Reference Queue and ImagePairLoader objects
        self.queue = Queue(maxsize=max_queue_size)
        self.image_pair_loader = image_pair_loader

        # Initialize probability of loading a new (not currently in queue)
        # image to 1.0 (queue is empty at first anyway)
        self.load_new_prob = 1.0

        # Call the entry func when an image is added to the queue and the exit
        # func when the image leaves the queue
        self.entry_func = (entry_func, entry_func_kw or {})
        self.exit_func = (exit_func, exit_func_kw or {})

        # Store reference to all running threads
        self.threads = []

        # Store the number of times each image identifier is currently in the queue
        self.items_in_queue = 0
        self._last = 0
        self.no_new_counter = 0

        # Reference to images not in queue and IDs in queue
        self.num_times_in_queue = {image: 0 for image in self.image_pair_loader}

    @property
    def load_new_prob(self):
        return self._load_new_prob

    @load_new_prob.setter
    def load_new_prob(self, value):
        self._load_new_prob = np.clip(value, 0.05, 1.0)

    def set_entry_func(self, func_str, func_kw=None):
        self.entry_func = (func_str, func_kw or {})

    def set_exit_func(self, func_str, func_kw=None):
        self.exit_func = (func_str, func_kw or {})

    def wait_N(self, N):
        """
        Sleep until N images has been added to the queue

        Args:
            N: Int, number of images to wait for
        """
        cur = self.items_in_queue
        while self.items_in_queue < cur + N-1:
            time.sleep(1)

    @contextmanager
    def get(self):
        """
        Context manager method pulling an image from the queue and yielding it
        At yield return time the exit_func is called upon the image unless it
        has another reference later in the queue

        yields:
            an ImagePair from the queue
        """
        if self.items_in_queue < 0.1 * self.queue.maxsize:
            # If queue is almost empty, halt the main thread a bit
            self.wait_N(N=3)

        # Get the image from the queue
        image = self.queue.get()

        # Check if too high new_prob
        if self._last:
            diff = self.items_in_queue - self._last
            if diff > 0 or self.items_in_queue >= self.queue.maxsize-1:
                # If queue is increasing in size, increase load new prob
                self.load_new_prob *= 1.05
            elif diff < 0:
                # If queue is decreasing in size, decrease load new prob
                self.load_new_prob *= 0.95
        else:
            self._last = self.items_in_queue

        # Yield back
        yield image

        # Update reference attributes
        self.items_in_queue -= 1
        self.num_times_in_queue[image] -= 1

        # Call exit function on the object
        if self.num_times_in_queue[image] == 0:
            # Unload if last in the queue
            getattr(image, self.exit_func[0])(**self.exit_func[1])
            image.load_state = None

    def start(self, n_threads=3):
        """
        Start populating the queue in n_threads

        Args:
            n_threads: Number of threads to spin up
        """
        for _ in range(n_threads):
            stop_event = Event()
            thread = Thread(target=_start, args=(self, stop_event))
            thread.start()
            self.threads.append((thread, stop_event))

    def stop(self):
        """
        Stop populating the queue by invoking the stop event on all threads and
        wait for them to terminate.
        """
        print("Stopping %i threads" % len(self.threads))
        for _, event in self.threads:
            # Make sure no threads keep working after next addition to the Q
            event.set()
        for i, (t, _) in enumerate(self.threads):
            # Wait for the threads to stop
            print("   %i/%i" % (i+1, len(self.threads)), end="\r", flush=True)
            t.join()
        print("")

    @property
    def unique_in_queue(self):
        """
        Returns:
            Int, the current number of unique images in the queue
        """
        return sum([bool(m) for m in self.num_times_in_queue.values()])

    def await_full(self):
        """
        Halt main thread until queue object is populated to its max capacity
        """
        while self.items_in_queue < self.queue.maxsize:
            print("   Data queue being populated %i/%i" % (self.items_in_queue,
                                                           self.queue.maxsize),
                  end='\r', flush=True)
            time.sleep(1)

    def _populate(self):
        """
        Puts a random image into the queue. The ImagePair is either taken from
        the ImagePairLoader in an un-loaded state or from the already loaded,
        processed images stored in the current queue.

        This method should be continuously invoked from one of more threads
        to maintain a populated queue.
        """
        # With load_new_prob probability we chose not to reload a new image
        load_new = np.random.rand() < self.load_new_prob or \
                   (self.unique_in_queue < 0.2 * self.queue.maxsize)

        # Pick random image
        found = False
        while not found:
            image = self.image_pair_loader.images[np.random.randint(len(self.image_pair_loader))]
            already_loaded = bool(self.num_times_in_queue[image])
            found = load_new != already_loaded

        # Increment the image counter
        self.num_times_in_queue[image] += 1

        # If the image is not currently loaded, invoke the entry function
        if getattr(image, "load_state", None) != self.entry_func[0]:
            # Set load_state so that future calls dont try to load and
            # preprocess again
            image.load_state = self.entry_func[0]

            # Call entry function
            getattr(image, self.entry_func[0])(**self.entry_func[1])

        # Add it to the queue, block indefinitely until spot is free
        self.queue.put(image, block=True, timeout=None)

        # Increment in-queue counter
        self.items_in_queue += 1
