import glob
import os
import sys
from argparse import ArgumentParser
from shutil import copy

from mpunet.train import YAMLHParams
from mpunet.utils import get_best_model, create_folders


def get_parser():
    parser = ArgumentParser(description='Branch a trained multi-task model '
                                        'into separate models for each task.')

    parser.add_argument("--project_folder", type=str,
                        default=os.path.abspath("./"),
                        help="The project folder (defaults to current dir)")
    parser.add_argument("--copy_weights", action="store_true",
                        help="Copy the weights file instead of sym-linking.")
    parser.add_argument("--weights_file", type=str, default="",
                        help="(Optional) Path to the weights file. Will be"
                             " inferred automatically if not specified.")
    parser.add_argument("--out_dir", type=str, default="branches",
                        help="Sub-folder to store the branched projects ("
                             "defaults to 'branches')")

    return parser


def branch(task_name, out_dir, task_hparams, task_hparams_file,
           shared_hparams, weights, copy_weights, views_file):
    create_folders(out_dir)

    # Set the task fields to only the current task under 'build'
    shared_hparams.set_value("build", "task_names",
                             value='["%s"]' % task_name, overwrite=True)
    shared_hparams.delete_group("tasks")  # No longer necessary

    # Create a map defining where each task specific parameter should go
    mapping = {
        "task_specifics/n_classes": "build/n_classes",
        "task_specifics/n_channels": "build/n_channels",
        "task_specifics/dim": ["build/dim", "fit/dim"],
        "task_specifics/out_activation": "build/out_activation",
        "task_specifics/real_space_span": "fit/real_space_span"
    }

    for soruce, targets in mapping.items():
        in_key1, in_key2 = soruce.split("/")
        value = task_hparams[in_key1][in_key2]

        # Set the value at all targets
        targets = [targets] if isinstance(targets, str) else targets
        for target in targets:
            out_key1, out_key2 = target.split("/")
            shared_hparams.set_value(out_key1, out_key2, value, overwrite=True)

    # Add all data folders
    data_folders = ("train_data", "val_data", "test_data", "aug_data")
    for df in data_folders:
        yaml = task_hparams.get_group(df)
        shared_hparams.add_group(yaml_string=yaml)

    # Save the updates parameters to a new location
    shared_hparams.save_current(os.path.join(out_dir, "train_hparams.yaml"))

    # Add weights to folder
    weights_folder = os.path.join(out_dir, "model")
    create_folders(weights_folder)
    out_weights_name = os.path.split(weights)[1]
    out_weights_path = os.path.join(weights_folder, out_weights_name)
    func = copy if copy_weights else os.symlink
    if copy_weights:
        print("Copying weights...")
    else:
        print("Symlinking weights...")
    func(weights, out_weights_path)

    # Add views (check for existence for future compatibility with 3D models)
    if os.path.exists(views_file):
        func(views_file, os.path.join(out_dir, "views.npz"))


def entry_func(args=None):
    parser = get_parser()
    args = vars(parser.parse_args(args))

    # Get arguments
    project_folder = os.path.abspath(args["project_folder"])
    copy_weights = args["copy_weights"]
    weights = os.path.abspath(args["weights_file"] or
                              get_best_model(project_folder + "/model"))
    out_dir = os.path.join(project_folder, args["out_dir"])

    # Get main hyperparamter file and check if correct modelt ype
    hparams = YAMLHParams(project_folder + "/train_hparams.yaml", no_log=True)
    tasks = hparams.get("tasks", False)
    if not tasks:
        print("[ERROR] Project must be of type 'MultiTask'.")
        sys.exit(0)

    # Branch out each sub-task
    create_folders(out_dir)
    for name, hparams_file in zip(tasks["task_names"], tasks["hparam_files"]):
        print("\n[*] Branching task %s" % name)
        # Reload the hparams in each iteration as we overwrite fields each time
        hparams = YAMLHParams(project_folder + "/train_hparams.yaml", no_log=False)
        # Get task specific parameters
        task_hparams = YAMLHParams(project_folder + "/%s" % hparams_file)
        branch(task_name=name,
               out_dir=os.path.join(out_dir, name),
               task_hparams=task_hparams,
               task_hparams_file=hparams_file,
               shared_hparams=hparams,
               weights=weights,
               copy_weights=copy_weights,
               views_file=os.path.join(project_folder, "views.npz"))
