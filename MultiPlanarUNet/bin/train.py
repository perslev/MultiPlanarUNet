from argparse import ArgumentParser
import sys
import os


def get_argparser():
    parser = ArgumentParser(description='Fit a MultiPlanarUNet model defined in a project folder. '
                                        'Invoke "init_project" to start a new project.')
    parser.add_argument("--project_dir", type=str, default="./",
                        help='path to MultiPlanarUNet project folder')
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help="Number of GPUs to use for this job (default=1)")
    parser.add_argument("--force_GPU", type=str, default="")
    parser.add_argument("--continue_training", action="store_true",
                        help="Continue the last training session")
    parser.add_argument("--overwrite", action='store_true',
                        help='overwrite previous training session in the project path')
    parser.add_argument("--just_one", action="store_true",
                        help="For testing purposes, run only on the first "
                             "training and validation samples.")
    parser.add_argument("--no_val", action="store_true",
                        help="For testing purposes, do not perform validation.")
    parser.add_argument("--no_images", action="store_true",
                        help="Do not save images during training")
    parser.add_argument("--debug", action="store_true",
                        help="Set tfbg CLI wrapper on session object")
    parser.add_argument("--wait_for", type=str, default="",
                        help="Waiting for PID to terminate before starting "
                             "training process.")
    parser.add_argument("--train_images_per_epoch", type=int, default=2500,
                        help="Number of training images to sample in each "
                             "epoch")
    parser.add_argument("--val_images_per_epoch", type=int, default=3500,
                        help="Number of training images to sample in each "
                             "epoch")
    return parser


def validate_path(base_path):
    if not os.path.exists(base_path) or \
            not os.path.exists(os.path.join(base_path, "train_hparams.yaml")):
        print("Path: %s is not a valid project folder.\n"
              "Make sure the folder exists and contains a "
              "'train_hparams.yaml' file." % base_path)
        import sys
        sys.exit(0)


def validate_hparams(hparams):
    # Tests for valid hparams

    if hparams["fit"]["class_weights"]:
        # Only currently supported loss
        assert hparams["fit"]["loss"] in ("WeightedSemanticCCE",
                                          "GeneralizedDiceLoss",
                                          "WeightedCrossEntropyWithLogits",
                                          "FocalLoss")

    if hparams["fit"]["loss"] == "WeightedCrossEntropyWithLogits":
        assert bool(hparams["fit"]["class_weights"])
        assert hparams["build"]["out_activation"] == "linear"


def run(base_path, gpu_mon, num_GPUs, continue_training, force_GPU, just_one,
        no_val, no_images, debug, wait_for, logger, train_images_per_epoch,
            val_images_per_epoch, **kwargs):

    from MultiPlanarUNet.train import Trainer, YAMLHParams
    from MultiPlanarUNet.models import model_initializer
    from MultiPlanarUNet.preprocessing import get_preprocessing_func

    # Read in hyperparameters from YAML file
    hparams = YAMLHParams(base_path + "/train_hparams.yaml", logger=logger)
    validate_hparams(hparams)

    # Wait for PID?
    if wait_for:
        from MultiPlanarUNet.utils import await_PIDs
        await_PIDs(wait_for)

    # Prepare Sequence generators and potential model specific hparam changes
    f = get_preprocessing_func(hparams["build"].get("model_class_name"))
    train, val, hparams = f(hparams, logger=logger, just_one=just_one,
                            no_val=no_val, continue_training=continue_training,
                            base_path=base_path)

    if gpu_mon:
        # Wait for free GPU
        if not force_GPU:
            gpu_mon.await_and_set_free_GPU(N=num_GPUs, sleep_seconds=120)
        else:
            gpu_mon.set_GPUs = force_GPU
            num_GPUs = len(force_GPU.split(","))
        gpu_mon.stop()

    # Build new model (or continue training an existing one)
    org_model = model_initializer(hparams=hparams,
                                  continue_training=continue_training,
                                  project_dir=base_path,
                                  logger=logger)

    # Initialize weights in final layer?
    if not continue_training and hparams["build"].get("biased_output_layer"):
        from MultiPlanarUNet.utils.utils import set_bias_weights_on_all_outputs
        set_bias_weights_on_all_outputs(org_model, train, hparams, logger)

    # Multi-GPU?
    if num_GPUs > 1:
        from tensorflow.keras.utils import multi_gpu_model
        model = multi_gpu_model(org_model, gpus=num_GPUs,
                                cpu_merge=False, cpu_relocation=False)
        logger("Creating multi-GPU model: N=%i" % num_GPUs)
    else:
        model = org_model

    # Init trainer
    trainer = Trainer(model, logger=logger)
    trainer.org_model = org_model

    # Compile model
    trainer.compile_model(n_classes=hparams["build"].get("n_classes"),
                          **hparams["fit"])

    # Debug mode?
    if debug:
        from tensorflow.python import debug as tfdbg
        from tensorflow.keras import backend as k
        k.set_session(tfdbg.LocalCLIDebugWrapperSession(k.get_session()))

    # Fit the model
    _ = trainer.fit(train=train, val=val,
                    train_im_per_epoch=train_images_per_epoch,
                    val_im_per_epoch=val_images_per_epoch,
                    hparams=hparams, no_im=no_images, **hparams["fit"])

    # Save final model weights (usually not used, but maybe....?)
    if not os.path.exists("%s/model" % base_path):
        os.mkdir("%s/model" % base_path)
    model_path = "%s/model/model_weights.h5" % base_path
    logger("Saving current model to: %s" % model_path)
    org_model.save_weights(model_path)


def remove_previous_session(base_path):
    import shutil
    # Remove old files and directories of logs, images etc if existing
    paths = [os.path.join(base_path, p) for p in ("images", "logs",
                                                  "tensorboard", "views.npz",
                                                  "views.png",
                                                  "auditor.pickle")]
    for p in filter(os.path.exists, paths):
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def entry_func(args=None):
    # Get settings
    # Project base path etc.
    args = vars(get_argparser().parse_args(args))
    base_path = os.path.abspath(args["project_dir"])
    overwrite = args["overwrite"]
    continue_training = args["continue_training"]
    num_GPUs = args["num_GPUs"]

    if continue_training and overwrite:
        raise ValueError("Cannot both continue training and overwrite the "
                         "previous training session. Remove the --overwrite "
                         "flag if trying to continue a previous training "
                         "session.")

    # Check path
    validate_path(base_path)
    if overwrite:
        remove_previous_session(base_path)

    # Define Logger object
    # Also checks if the model in the project folder has already been fit
    from MultiPlanarUNet.logging import Logger
    try:
        logger = Logger(base_path, print_to_screen=True,
                        overwrite_existing=continue_training)
    except OSError:
        print("\n[*] A training session at '%s' already exists."
              "\n    Use the --overwrite flag to overwrite." % base_path)
        sys.exit(0)
    logger("Fitting model in path:\n%s" % base_path)

    if num_GPUs >= 0:
        # Initialize GPUMonitor in separate fork now before memory builds up
        from MultiPlanarUNet.utils.system import GPUMonitor
        gpu_mon = GPUMonitor(logger)
    else:
        gpu_mon = None

    try:
        run(base_path=base_path, gpu_mon=gpu_mon, logger=logger, **args)
    except Exception as e:
        gpu_mon.stop()
        raise e


if __name__ == "__main__":
    entry_func()
