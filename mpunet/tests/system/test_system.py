import unittest


class TestSystem(unittest.TestCase):

    minimum_mem_gib = 8.0
    minimum_cpu_cores = 4
    minimum_gpus = 1

    def test_enough_memory(self):
        """ Tests if the system has the recommended amount of memory. """
        import psutil
        mem = psutil.virtual_memory()
        mem_total_gib = mem.total * (2**-30)
        self.assertGreaterEqual(mem_total_gib, self.minimum_mem_gib,
                                msg="The total available memory of this system"
                                    " is {} GiB. For most practical "
                                    "application more than {} GiB will be "
                                    "needed to run mpunet optimally. "
                                    "For testing purposes, you may disregard "
                                    "this error.".format(mem_total_gib,
                                                         self.minimum_mem_gib))

    def test_enough_cpu_cores(self):
        """ Tests if the system has the recommended number of CPU cores. """
        import psutil
        cpus = psutil.cpu_count(logical=True)
        self.assertGreaterEqual(cpus, self.minimum_cpu_cores,
                                msg="The number of logical CPU cores of this "
                                    "system is {}. We recommend at least {} "
                                    "for mpunet to run optimally. "
                                    "For testing purposes, you may disregard "
                                    "this error.".format(cpus,
                                                         self.minimum_cpu_cores))

    def test_nvidia_smi_available(self):
        """ Tests if the nvidia-smi is available """
        msg = "nvidia-smi does not seem to be available - " \
              "this could indicate that the nvidia drivers are not installed."
        import shutil
        import os
        path = shutil.which("nvidia-smi")
        self.assertIsNot(path, None, msg)
        self.assertTrue(os.path.exists(path), msg)

    def test_gpu_count(self):
        """ Tests if the system has at least 1 nvidia GPU """
        from subprocess import check_output
        try:
            gpu_list = check_output("nvidia-smi "
                                    "--query-gpu=gpu_name "
                                    "--format=csv".split(" "),
                                    universal_newlines=True).split("\n")[1:]
        except FileNotFoundError as e:
            raise FileNotFoundError("nvidia-smi does not seem to be "
                                    "installed - this could indicate missing "
                                    "GPU drivers.") from e
        n_gpus = len(list(filter(None, gpu_list)))
        self.assertGreaterEqual(n_gpus, self.minimum_gpus,
                                msg="The number of GPUs on this "
                                    "system is {}. We recommend at least {} "
                                    "for mpunet to run optimally. "
                                    "For testing purposes, you may disregard "
                                    "this error.".format(n_gpus,
                                                         self.minimum_gpus))
