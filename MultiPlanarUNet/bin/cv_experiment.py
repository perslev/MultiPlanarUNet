import os
from multiprocessing import Process, Lock, Queue, Event
from MultiPlanarUNet.utils import create_folders
from MultiPlanarUNet.bin.init_project import copy_yaml_and_set_data_dirs
from MultiPlanarUNet.logging import Logger
import argparse
import subprocess


def get_parser():
    parser = argparse.ArgumentParser(description="Prepare a data folder for a"
                                                 "CV experiment setup.")
    parser.add_argument("--CV_dir", type=str, required=True,
                        help="Directory storing split subfolders as output by"
                             " cv_split.py")
    parser.add_argument("--out_dir", type=str, default="./splits",
                        help="Folder in which experiments will be run and "
                             "results stored.")
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help="Number of GPUs to use per process. This also "
                             "defines the number of parallel jobs to run.")
    parser.add_argument("--num_jobs", type=int, default=0,
                        help="OBS: Only in effect when --num_GPUs=0. Sets"
                             " the number of jobs to run in parallel when no"
                             " GPUs are attached to each job.")
    parser.add_argument("--script_prototype", type=str, default="./script",
                        help="Path to text file listing commands and "
                             "arguments to execute under each sub-exp folder.")
    parser.add_argument("--hparams_prototype", type=str,
                        default="./train_hparams.yaml",
                        help="Prototype hyperparameter yaml file from which"
                             " sub-CV files will be made.")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Start from CV split<start_from>. Default 0.")
    parser.add_argument("--wait_for", type=str, default="",
                        help="Waiting for PID to terminate before starting "
                             "training process.")
    parser.add_argument("--monitor_GPUs_every", type=int, default=None,
                        help="If specified, start a background process which"
                             " monitors every 'monitor_GPUs_every' seconds "
                             "whether new GPUs have become available than may"
                             " be included in the CV experiment GPU resource "
                             "pool.")
    return parser


def get_CV_folders(dir):
    key = lambda x: int(x.split("_")[-1])
    return [os.path.join(dir, p) for p in sorted(os.listdir(dir), key=key)]


def _get_GPU_sets(free_gpus, num_GPUs):
    free_gpus = list(map(str, free_gpus))
    return [",".join(free_gpus[x:x + num_GPUs]) for x in range(0, len(free_gpus),
                                                               num_GPUs)]


def get_free_GPU_sets(num_GPUs):
    from MultiPlanarUNet.utils.system import GPUMonitor
    mon = GPUMonitor()
    free_gpus = sorted(mon.free_GPUs, key=lambda x: int(x))
    total_GPUs = len(free_gpus)
    mon.stop()

    if total_GPUs % num_GPUs or not free_gpus:
        if total_GPUs < num_GPUs:
            raise ValueError("Invalid number of GPUs per process '%i' for total "
                             "GPU count of '%i' - must be evenly divisible." %
                             (num_GPUs, total_GPUs))
        else:
            full_sequence = list(map(str, range(0, max(map(int, free_gpus))+1)))
            full_sets = _get_GPU_sets(full_sequence, num_GPUs)
            valid_sets = []
            for s in full_sets:
                ok_len = len(s.split(",")) == num_GPUs
                ok_gpus = all([gpu in free_gpus for gpu in s.split(",")])
                if ok_len and ok_gpus:
                    valid_sets.append(s)
            if not valid_sets:
                raise ValueError("No free GPU sets")
            else:
                return valid_sets
    else:
        return _get_GPU_sets(free_gpus, num_GPUs)


def monitor_GPUs(every, gpu_queue, num_GPUs, current_pool, stop_event):
    import time
    # Make flat version of the list of gpu sets
    current_pool = [gpu for sublist in current_pool for gpu in sublist.split(",")]
    while not stop_event.is_set():
        # Get available GPU sets. Will raise ValueError if no full set is
        # available
        try:
            gpu_sets = get_free_GPU_sets(num_GPUs)
            for gpu_set in gpu_sets:
                if any([g in current_pool for g in gpu_set.split(",")]):
                    # If one or more GPUs are already in use - this may happen
                    # initially as preprocessing occurs in a process before GPU
                    # memory has been allocated - ignore the set
                    continue
                else:
                    gpu_queue.put(gpu_set)
                    current_pool += gpu_set.split(",")
        except ValueError:
            pass
        finally:
            time.sleep(every)


def parse_script(script, GPUs):
    commands = []
    with open(script) as in_file:
        for line in in_file:
            line = line.strip(" \n")
            if not line or line[0] == "#":
                continue
            # Split out in-line comments
            line = line.split("#")[0]
            # Get all arguments, remove if concerning GPU (controlled here)
            cmd = list(filter(lambda x: "gpu" not in x.lower(), line.split()))
            if "python" in line:
                cmd.append("--force_GPU=%s" % GPUs)
            commands.append(cmd)
    return commands


def run_sub_experiment(split_dir, out_dir, script, hparams, GPUs, GPU_queue, lock, logger):

    # Create sub-directory
    split = os.path.split(split_dir)[-1]
    out_dir = os.path.join(out_dir, split)
    out_hparams = os.path.join(out_dir, "train_hparams.yaml")
    create_folders(out_dir)

    # Get list of commands
    commands = parse_script(script, GPUs)

    # Move hparams and script files into folder
    copy_yaml_and_set_data_dirs(in_path=hparams,
                                out_path=out_hparams,
                                data_dir=split_dir)

    # Change directory and file permissions
    os.chdir(out_dir)

    # Log
    lock.acquire()
    s = "[*] Running experiment: %s" % split
    logger("\n%s\n%s" % ("-" * len(s), s))
    logger("Data dir:", split_dir)
    logger("Out dir:", out_dir)
    logger("Using GPUs:", GPUs)
    logger("\nRunning commands:")
    for i, command in enumerate(commands):
        logger(" %i) %s" % (i+1, " ".join(command)))
    logger("-"*len(s))
    lock.release()

    # Run the commands
    run_next_command = True
    for command in commands:
        if not run_next_command:
            break
        lock.acquire()
        logger("[%s - STARTING] %s" % (split, " ".join(command)))
        lock.release()
        p = subprocess.Popen(command, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        _, err = p.communicate()
        rc = p.returncode
        lock.acquire()
        if rc != 0:
            logger("[%s - ERROR - Exit code %i] %s" % (split, rc, " ".join(command)))
            logger("\n----- START error message -----\n%s\n"
                   "----- END error message -----\n" % err.decode("utf-8"))
            run_next_command = False
        else:
            logger("[%s - FINISHED] %s" % (split, " ".join(command)))
        lock.release()

    # Add the GPUs back into the queue
    GPU_queue.put(GPUs)


def entry_func(args=None):
    # Get parser
    parser = vars(get_parser().parse_args(args))

    # Get parser arguments
    cv_dir = os.path.abspath(parser["CV_dir"])
    out_dir = os.path.abspath(parser["out_dir"])
    create_folders(out_dir)
    start_from = parser["start_from"]
    await_PID = parser["wait_for"]
    monitor_GPUs_every = parser["monitor_GPUs_every"]

    # Get a logger object
    logger = Logger(base_path="./", active_file="output",
                    print_calling_method=False, overwrite_existing=True)

    # Wait for PID?
    if await_PID:
        from MultiPlanarUNet.utils import await_PIDs
        await_PIDs(await_PID)

    # Get number of GPUs per process
    num_GPUs = parser["num_GPUs"]

    # Get file paths
    script = os.path.abspath(parser["script_prototype"])
    hparams = os.path.abspath(parser["hparams_prototype"])

    # Get list of folders of CV data to run on
    cv_folders = get_CV_folders(cv_dir)

    # Get GPU sets
    if num_GPUs:
        gpu_sets = get_free_GPU_sets(num_GPUs)
    elif parser["num_jobs"] < 1:
        raise ValueError("Should specify a number of jobs to run in parallel "
                         "with the --num_jobs flag when using 0 GPUs pr. "
                         "process (--num_GPUs=0 was set).")
    else:
        gpu_sets = ["''"] * parser["num_jobs"]

    # Get process pool, lock and GPU queue objects
    lock = Lock()
    gpu_queue = Queue()
    for gpu in gpu_sets:
        gpu_queue.put(gpu)

    procs = []
    if monitor_GPUs_every is not None and monitor_GPUs_every:
        logger("\nOBS: Monitoring GPU pool every %i seconds\n" % monitor_GPUs_every)
        # Start a process monitoring new GPU availability over time
        stop_event = Event()
        t = Process(target=monitor_GPUs, args=(monitor_GPUs_every, gpu_queue,
                                               num_GPUs, gpu_sets, stop_event))
        t.start()
        procs.append(t)
    else:
        stop_event = None
    try:
        for cv_folder in cv_folders[start_from:]:
            gpus = gpu_queue.get()
            t = Process(target=run_sub_experiment,
                        args=(cv_folder, out_dir, script, hparams,
                              gpus, gpu_queue, lock, logger))
            t.start()
            procs.append(t)
            for t in procs:
                if not t.is_alive():
                    t.join()
    except KeyboardInterrupt:
        for t in procs:
            t.terminate()
    if stop_event is not None:
        stop_event.set()
    for t in procs:
        t.join()


if __name__ == "__main__":
    entry_func()
