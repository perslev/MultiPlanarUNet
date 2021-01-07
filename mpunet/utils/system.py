import time
import os

from multiprocessing import Process, Event, Queue
from mpunet.utils.utils import get_free_gpus, _get_free_gpu, set_gpu
from mpunet.logging import ScreenLogger


class GPUMonitor(Process):
    def __init__(self):
        self.logger = ScreenLogger()

        # Prepare signal
        self.stop_signal = Event()
        self.run_signal = Event()
        self.set_signal = Event()

        # Stores list of available GPUs
        self._free_GPUs = Queue()

        super(GPUMonitor, self).__init__(target=self._monitor)
        self.start()

    def stop(self):
        self.stop_signal.set()

    def _monitor(self):
        while not self.stop_signal.is_set():
            if self.run_signal.is_set():
                # Empty queue, just in case...?
                self._free_GPUs.empty()

                # Get free GPUs
                free = get_free_gpus()

                # Add number of elements that will be put in the queue as first
                # element to be read from main process
                self._free_GPUs.put(len(free))

                # Add available GPUs to queue
                for g in free:
                    self._free_GPUs.put(g)

                # Set the signal that main process can start reading queue
                self.set_signal.set()

                # Stop run signal for this process
                self.run_signal.clear()
            else:
                time.sleep(0.5)
                self.set_signal.clear()

    @property
    def free_GPUs(self):
        self.run_signal.set()
        while not self.set_signal.is_set():
            time.sleep(0.2)

        free = []
        for i in range(self._free_GPUs.get()):
            free.append(self._free_GPUs.get())
        return free

    def get_free_GPUs(self, N):
        return _get_free_gpu(self.free_GPUs, N)

    def await_and_set_free_GPU(self, N=0, sleep_seconds=60, stop_after=False):
        cuda_visible_dev = ""
        if N != 0:
            self.logger("Waiting for free GPU.")
            found_gpu = False
            while not found_gpu:
                cuda_visible_dev = self.get_free_GPUs(N=N)
                if cuda_visible_dev:
                    self.logger("Found free GPU: %s" % cuda_visible_dev)
                    found_gpu = True
                else:
                    self.logger("No available GPUs... Sleeping %i seconds." % sleep_seconds)
                    time.sleep(sleep_seconds)
        else:
            self.logger("Using CPU based computations only!")
        self.set_GPUs = cuda_visible_dev
        if stop_after:
            self.stop()

    @property
    def num_currently_visible(self):
        return len(self.set_GPUs.strip().split(","))

    @property
    def set_GPUs(self):
        try:
            return os.environ["CUDA_VISIBLE_DEVICES"]
        except KeyError:
            return ""

    @set_GPUs.setter
    def set_GPUs(self, GPUs):
        set_gpu(GPUs)

    def set_and_stop(self, GPUs):
        self.set_GPUs = GPUs
        self.stop()
