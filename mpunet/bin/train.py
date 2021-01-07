"""
mpunet train script
Optimizes a mpunet model in a specified project folder

Typical usage:
--------------
mp init_project --name my_project --data_dir ~/path/to/data
cd my_project
mp train --num_GPUs=1
--------------
"""


from argparse import ArgumentParser
import os


def get_argparser():
    parser = ArgumentParser(description='Fit a mpunet model defined '
                                        'in a project folder. '
                                        'Invoke "init_project" to start a '
                                        'new project.')
    parser.add_argument("--project_dir", type=str, default='./',
                        help="Path to a mpunet project directory. "
                             "Defaults to the current directory.")
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help="Number of GPUs to use for this job (default=1)")
    parser.add_argument("--force_GPU", type=str, default="",
                        help="Manually set the CUDA_VISIBLE_DEVICES env "
                             "variable to this value "
                             "(force a specific set of GPUs to be used)")
    parser.add_argument("--continue_training", action="store_true",
                        help="Continue the last training session")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite previous training session at the '
                             'project path')
    parser.add_argument("--just_one", action="store_true",
                        help="For testing purposes, run only on the first "
                             "training and validation samples.")
    parser.add_argument("--no_val", action="store_true",
                        help="Do not perform validation (must be set if no "
                             "validation set exists)")
    parser.add_argument("--no_images", action="store_true",
                        help="Do not save sample images during training")
    parser.add_argument("--debug", action="store_true",
                        help="Set tfbg CLI wrapper on the session object")
    parser.add_argument("--wait_for", type=str, default="",
                        help="Wait for PID to terminate before starting the "
                             "training process.")
    parser.add_argument("--train_images_per_epoch", type=int, default=2500,
                        help="Number of training images to sample in each "
                             "epoch")
    parser.add_argument("--val_images_per_epoch", type=int, default=3500,
                        help="Number of training images to sample in each "
                             "epoch")
    parser.add_argument("--max_loaded_images", type=int, default=None,
                        help="Set a maximum number of (training) images to "
                             "keep loaded in memory at a given time. Images "
                             "will be cycled every '--num_access slices.'. "
                             "Default=None (all loaded).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Overwrite the number of epochs as specified in "
                             "the hyperparameters file")
    parser.add_argument("--num_access", type=int, default=50,
                        help="Only effective with --max_loaded_images set. "
                             "Sets the number of times an images stored in "
                             "memory may be accessed (e.g. for sampling an "
                             "image slice) before it is replaced by another "
                             "image. Higher values makes the data loader "
                             "less likely to block. Lower values ensures that "
                             "images are sampled across all images of the "
                             "dataset. Default=50.")
    return parser


def validate_project_dir(project_dir):
    if not os.path.exists(project_dir) or \
            not os.path.exists(os.path.join(project_dir, "train_hparams.yaml")):
        raise RuntimeError("The script was launched from directory:\n'%s'"
                           "\n... but this is not a valid project folder.\n\n"
                           "* Make sure to launch the script from within a "
                           "MultiPlanarNet project directory\n"
                           "* Make sure that the directory contains a "
                           "'train_hparams.yaml' file." % project_dir)


def validate_args(args):
    """
    Checks that the passed commandline arguments are valid

    Args:
        args: argparse arguments
    """
    if args.continue_training and args.overwrite:
        raise ValueError("Cannot both continue training and overwrite the "
                         "previous training session. Remove the --overwrite "
                         "flag if trying to continue a previous training "
                         "session.")
    if args.train_images_per_epoch <= 0:
        raise ValueError("train_images_per_epoch must be a positive integer")
    if args.val_images_per_epoch <= 0:
        raise ValueError("val_images_per_epoch must be a positive integer. "
                         "Use --no_val instead.")
    if args.force_GPU and args.num_GPUs != 1:
        raise ValueError("Should not specify both --force_GPU and --num_GPUs")
    if args.num_GPUs < 0:
        raise ValueError("num_GPUs must be a positive integer")


def validate_hparams(hparams):
    """
    Limited number of checks performed on the validity of the hyperparameters.
    The file is generally considered to follow the semantics of the
    mpunet.bin.defaults hyperparameter files.

    Args:
        hparams: A YAMLHParams object
    """
    # Tests for valid hparams
    if hparams["fit"].get("class_weights") and hparams["fit"]["loss"] not in \
            ("SparseFocalLoss",):
        # Only currently supported losses
        raise ValueError("Invalid loss function '{}' used with the "
                         "'class_weights' "
                         "parameter".format(hparams["fit"]["loss"]))
    if hparams["fit"]["loss"] == "WeightedCrossEntropyWithLogits":
        if not bool(hparams["fit"]["class_weights"]):
            raise ValueError("Must specify 'class_weights' argument with loss"
                             "'WeightedCrossEntropyWithLogits'.")
        if not hparams["build"]["out_activation"] == "linear":
            raise ValueError("Must use out_activation: linear parameter with "
                             "loss 'WeightedCrossEntropyWithLogits'")
    if not hparams["train_data"]["base_dir"]:
        raise ValueError("No training data folder specified in parameter file.")


def remove_previous_session(project_folder):
    """
    Deletes various mpunet project folders and files from
    [project_folder].

    Args:
        project_folder: A path to a mpunet project folder
    """
    import shutil
    # Remove old files and directories of logs, images etc if existing
    paths = [os.path.join(project_folder, p) for p in ("images",
                                                       "logs",
                                                       "tensorboard",
                                                       "views.npz",
                                                       "views.png")]
    for p in filter(os.path.exists, paths):
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def get_logger(project_dir, overwrite_existing):
    """
    Initialises and returns a Logger object for a given project directory.
    If a logfile already exists at the specified location, it will be
    overwritten if continue_training == True, otherwise raises RuntimeError

    Args:
        project_dir: Path to a mpunet project folder
        overwrite_existing: Whether to overwrite existing logfile in project_dir

    Returns:
        A mpunet Logger object initialized in project_dir
    """
    # Define Logger object
    from mpunet.logging import Logger
    try:
        logger = Logger(base_path=project_dir,
                        print_to_screen=True,
                        overwrite_existing=overwrite_existing)
    except OSError as e:
        raise RuntimeError("[*] A training session at '%s' already exists."
                           "\n    Use the --overwrite flag to "
                           "overwrite." % project_dir) from e
    return logger


def get_gpu_monitor(num_GPUs, logger):
    """
    Args:
        num_GPUs: Number of GPUs to train on
        logger: A mpunet logger object that will be passed to
                the GPUMonitor

    Returns:
        If num_GPUs >= 0, returns a GPUMonitor object, otherwise returns None
    """
    if num_GPUs >= 0:
        # Initialize GPUMonitor in separate fork now before memory builds up
        from mpunet.utils.system import GPUMonitor
        gpu_mon = GPUMonitor()
    else:
        gpu_mon = None
    return gpu_mon


def set_gpu(gpu_mon, args):
    """
    Sets the GPU visibility based on the passed arguments. Takes an already
    initialized GPUMonitor object. Sets GPUs according to args.force_GPU, if
    specified, otherwise sets first args.num_GPUs free GPUs on the system.

    Stops the GPUMonitor process once GPUs have been set
    If gpu_mon is None, this function does nothing

    Args:
        gpu_mon: An initialized GPUMonitor object or None
        args: argparse arguments

    Returns: The number of GPUs that was actually set (different from
    args.num_GPUs if args.force_GPU is set to more than 1 GPU)
    """
    num_GPUs = args.num_GPUs
    if gpu_mon is not None:
        if not args.force_GPU:
            gpu_mon.await_and_set_free_GPU(N=num_GPUs, sleep_seconds=120)
        else:
            gpu_mon.set_GPUs = args.force_GPU
            num_GPUs = len(args.force_GPU.split(","))
        gpu_mon.stop()
    return num_GPUs


def get_data_sequences(project_dir, hparams, logger, args):
    """
    Loads training and validation data as specified in the hyperparameter file.
    Returns a batch sequencer object for each dataset, not the  ImagePairLoader
    dataset itself. The preprocessing function may make changes to the hparams
    dictionary.

    Args:
        project_dir: A path to a mpunet project
        hparams: A YAMLHParams object
        logger: A mpunet logging object
        args: argparse arguments

    Returns:
        train: A batch sequencer object for the training data
        val: A batch sequencer object for the validation data,
             or None if --no_val was specified
        hparams: The YAMLHParams object
    """
    from mpunet.preprocessing import get_preprocessing_func
    func = get_preprocessing_func(hparams["build"].get("model_class_name"))
    hparams['fit']['flatten_y'] = True
    hparams['fit']['max_loaded'] = args.max_loaded_images
    hparams['fit']['num_access'] = args.num_access
    train, val, hparams = func(hparams=hparams,
                               logger=logger,
                               just_one=args.just_one,
                               no_val=args.no_val,
                               continue_training=args.continue_training,
                               base_path=project_dir)
    return train, val, hparams


def get_model(project_dir, train_seq, hparams, logger, args):
    """
    Initializes a tf.keras Model from mpunet.models as specified in
    hparams['build']. If args.continue_training, the best previous model stored
    in [project_dir]/models will be loaded.

    If hparams["build"]["biased_output_layer"] is True, sets the bias weights
    on the final conv. layer so that a zero-input gives an output of class
    probabilities equal to the class frequencies of the training set.

    Args:
        project_dir: A path to a mpunet project folder
        train_seq: A mpunet.sequences object for the training data
        hparams: A mpunet YAMLHParams object
        logger: A mpunet logging object
        args: argparse arguments

    Returns:
        model: The model to fit
        org_model: The original, non-GPU-distributed model
                   (Same as model if num_GPUs==1)
    """
    from mpunet.models import model_initializer
    # Build new model (or continue training an existing one)
    hparams["build"]['flatten_output'] = True
    model = model_initializer(hparams=hparams,
                              continue_training=args.continue_training,
                              project_dir=project_dir,
                              logger=logger)
    # Initialize weights in final layer?
    if not args.continue_training and hparams["build"].get("biased_output_layer"):
        from mpunet.utils.utils import set_bias_weights_on_all_outputs
        set_bias_weights_on_all_outputs(model,
                                        train_seq.image_pair_queue,
                                        hparams,
                                        logger)
    return model


def save_final_weights(model, project_dir, logger=None):
    """
    Saves the weights of 'model' to [project_dir]/model/model_weights.h5

    Args:
        model: A tf.keras Model object
        project_dir: A path to a mpunet project
        logger: mpunet logging object, or None
    """
    if not os.path.exists("%s/model" % project_dir):
        os.mkdir("%s/model" % project_dir)
    model_path = "%s/model/model_weights.h5" % project_dir
    if logger:
        logger("Saving current model to: %s" % model_path)
    model.save_weights(model_path)


def run(project_dir, gpu_mon, logger, args):
    """
    Runs training of a model in a mpunet project directory.

    Args:
        project_dir: A path to a mpunet project
        gpu_mon: An initialized GPUMonitor object
        logger: A mpunet logging object
        args: argparse arguments
    """
    # Read in hyperparameters from YAML file
    from mpunet.hyperparameters import YAMLHParams
    hparams = YAMLHParams(project_dir + "/train_hparams.yaml", logger=logger)
    validate_hparams(hparams)

    # Wait for PID to terminate before continuing?
    if args.wait_for:
        from mpunet.utils import await_PIDs
        await_PIDs(args.wait_for)

    # Prepare sequence generators and potential model specific hparam changes
    train, val, hparams = get_data_sequences(project_dir=project_dir,
                                             hparams=hparams,
                                             logger=logger,
                                             args=args)

    # Set GPU visibility and create model with MirroredStrategy
    set_gpu(gpu_mon, args)
    import tensorflow as tf
    with tf.distribute.MirroredStrategy().scope():
        model = get_model(project_dir=project_dir, train_seq=train,
                          hparams=hparams, logger=logger, args=args)

        # Get trainer and compile model
        from mpunet.train import Trainer
        trainer = Trainer(model, logger=logger)
        trainer.compile_model(n_classes=hparams["build"].get("n_classes"),
                              reduction=tf.keras.losses.Reduction.NONE,
                              **hparams["fit"])

    # Debug mode?
    if args.debug:
        from tensorflow.python import debug as tfdbg
        from tensorflow.keras import backend as K
        K.set_session(tfdbg.LocalCLIDebugWrapperSession(K.get_session()))

    # Fit the model
    hparams["fit"]["n_epochs"] = args.epochs or hparams["fit"]["n_epochs"]
    try:
        _ = trainer.fit(train=train, val=val,
                        train_im_per_epoch=args.train_images_per_epoch,
                        val_im_per_epoch=args.val_images_per_epoch,
                        hparams=hparams, no_im=args.no_images, **hparams["fit"])
    except KeyboardInterrupt:
        gpu_mon.stop()
    finally:
        save_final_weights(model, project_dir, logger)


def entry_func(args=None):
    """
    Function called from mp to init training
    1) Parses command-line arguments
    2) Validation command-line arguments
    3) Checks and potentially deletes a preious version of the project folder
    4) Initializes a logger and a GPUMonitor object
    5) Calls run() to start training

    Args:
        args: None or arguments passed from mp
    """
    # Get and check args
    args = get_argparser().parse_args(args)
    validate_args(args)

    # Check for project dir
    project_dir = os.path.abspath(args.project_dir)
    validate_project_dir(project_dir)
    os.chdir(project_dir)

    # Get project folder and remove previous session if --overwrite
    if args.overwrite:
        remove_previous_session(project_dir)

    # Get logger object. Overwrites previous logfile if args.continue_training
    logger = get_logger(project_dir, args.continue_training)
    logger("Fitting model in path:\n%s" % project_dir)

    # Start GPU monitor process, if num_GPUs > 0
    gpu_mon = get_gpu_monitor(args.num_GPUs, logger)

    try:
        run(project_dir=project_dir, gpu_mon=gpu_mon, logger=logger, args=args)
    except Exception as e:
        if gpu_mon is not None:
            gpu_mon.stop()
        raise e


if __name__ == "__main__":
    entry_func()
