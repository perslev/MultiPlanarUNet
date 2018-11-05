import os
from multiprocessing import Process, Lock, Queue
from MultiViewUNet.utils import create_folders
from MultiViewUNet.bin.init_project import copy_yaml_and_set_data_dirs
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
    return parser


def get_CV_folders(dir):
    key = lambda x: int(x.split("_")[-1])
    return [os.path.join(dir, p) for p in sorted(os.listdir(dir), key=key)]


def get_GPU_sets(num_GPUs):
    from MultiViewUNet.utils.system import GPUMonitor
    mon = GPUMonitor()
    free_gpus = sorted(mon.free_GPUs, key=lambda x: int(x))
    total_GPUs = len(free_gpus)
    mon.stop()

    if total_GPUs % num_GPUs:
        raise ValueError("Invalid number of GPUs per process '%i' for total "
                         "GPU count of '%i' - must be evenly divisible." %
                         (num_GPUs, total_GPUs))

    splits = [",".join(free_gpus[x:x+num_GPUs]) for x in range(0, total_GPUs,
                                                               num_GPUs)]
    return splits


def parse_script(script, GPUs):
    commands = []
    with open(script) as in_file:
        for line in in_file:
            # Get all arguments, remove if concerning GPU (controlled here)
            cmd = list(filter(lambda x: "gpu" not in x.lower(), line.split()))
            cmd.append("--force_GPU=%s" % GPUs)
            commands.append(cmd)
    return commands


def run_sub_experiment(split_dir, out_dir, script, hparams, GPUs, GPU_queue, lock):

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
    print("\n%s\n%s" % ("-" * len(s), s))
    print("Data dir:", split_dir)
    print("Out dir:", out_dir)
    print("Using GPUs:", GPUs)
    print("\nRunning commands:")
    for i, command in enumerate(commands):
        print(" %i) %s" % (i+1, " ".join(command)))
    print("-"*len(s))
    lock.release()

    # Run the commands
    for command in commands:
        lock.acquire()
        print("[%s - STARTING] %s" % (split, " ".join(command)))
        lock.release()
        p = subprocess.Popen(command, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        _, err = p.communicate()
        rc = p.returncode
        lock.acquire()
        if rc != 0:
            print("[%s - ERROR - Exit code %i] %s" % (split, rc, " ".join(command)))
            print("\n----- START error message -----\n%s\n"
                  "----- END error message -----\n" % err.decode("utf-8"))
            break
        else:
            print("[%s - FINISHED] %s" % (split, " ".join(command)))
        lock.release()

    # Add the GPUs back into the queue
    GPU_queue.put(GPUs)


if __name__ == "__main__":

    # Get parser
    parser = vars(get_parser().parse_args())

    # Get parser arguments
    cv_dir = os.path.abspath(parser["CV_dir"])
    out_dir = os.path.abspath(parser["out_dir"])
    create_folders(out_dir)
    start_from = parser["start_from"]
    await_PID = parser["wait_for"]

    # Wait for PID?
    if await_PID:
        from MultiViewUNet.utils import await_PIDs
        await_PIDs(await_PID)

    # Get number of GPUs per process
    num_GPUs = parser["num_GPUs"]

    # Get file paths
    script = os.path.abspath(parser["script_prototype"])
    hparams = os.path.abspath(parser["hparams_prototype"])

    # Get list of folders of CV data to run on
    cv_folders = get_CV_folders(cv_dir)

    # Get GPU sets
    gpu_sets = get_GPU_sets(num_GPUs)

    # Get process pool, lock and GPU queue objects
    lock = Lock()
    gpu_queue = Queue()
    for gpu in gpu_sets:
        gpu_queue.put(gpu)

    procs = []
    try:
        for cv_folder in cv_folders[start_from:]:
            gpus = gpu_queue.get()
            t = Process(target=run_sub_experiment,
                        args=(cv_folder, out_dir, script, hparams,
                              gpus, gpu_queue, lock))
            t.start()
            procs.append(t)
            for t in procs:
                if not t.is_alive():
                    t.join()
    except KeyboardInterrupt:
        for t in procs:
            t.terminate()
    for t in procs:
        t.join()
