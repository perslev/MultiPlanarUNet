from argparse import ArgumentParser
import sys
import os


def get_argparser():
    parser = ArgumentParser(description='Fit a MultiViewUNet model defined in a project folder. '
                                        'Invoke "init_project" to start a new project.')
    parser.add_argument("--project_dir", type=str, default="./",
                        help='path to MultiViewUNet project folder')
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
    return parser


def validate_path(path):
    if not os.path.exists(path) or \
            not os.path.exists(os.path.join(path, "train_hparams.yaml")):
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
                                          "WeightedCrossEntropyWithLogitsAndSparseTargets",
                                          "FocalLoss")

    if hparams["fit"]["loss"] == "WeightedCrossEntropyWithLogitsAndSparseTargets":
        assert bool(hparams["fit"]["class_weights"])
        assert hparams["fit"]["sparse"] is True
        assert hparams["build"]["out_activation"] == "linear"


def get_preprocessing_func(model):
    if model in ("UNet", "AutofocusUNet2D"):
        from MultiViewUNet.preprocessing import prepare_for_multi_view_unet
        return prepare_for_multi_view_unet
    elif model in ("UNet3D", "STUNet3D", "AutofocusUNet3D"):
        from MultiViewUNet.preprocessing import prepare_for_3d_unet
        return prepare_for_3d_unet
    else:
        raise ValueError("Unsupported model type '%s'" % model)


def run(base_path, gpu_mon, num_GPUs):
    # Extract command line settings
    continue_training = args["continue_training"]
    force_gpu = args["force_GPU"]
    just_one = args["just_one"]
    no_val = args["no_val"]
    no_im = args["no_images"]
    debug = args["debug"]
    await_PID = args["wait_for"]

    from MultiViewUNet.train import Trainer, YAMLHParams
    from MultiViewUNet.models import model_initializer

    # Read in hyperparameters from YAML file
    hparams = YAMLHParams(base_path + "/train_hparams.yaml", logger=logger)
    validate_hparams(hparams)

    # Wait for PID?
    if await_PID:
        from MultiViewUNet.utils import await_PIDs
        await_PIDs(await_PID)

    # Prepare Sequence generators and potential model specific hparam changes
    f = get_preprocessing_func(hparams["build"].get("model_class_name"))
    train, val, hparams = f(hparams, logger=logger, just_one=just_one,
                            no_val=no_val, continue_training=continue_training,
                            base_path=base_path)

    if gpu_mon:
        # Wait for free GPU
        if not force_gpu:
            gpu_mon.await_and_set_free_GPU(N=num_GPUs, sleep_seconds=120)
        else:
            gpu_mon.set_GPUs = force_gpu
            num_GPUs = len(force_gpu.split(","))
        gpu_mon.stop()

    # Build new model (or continue training an existing one)
    org_model = model_initializer(hparams, continue_training, base_path,
                                  logger)

    # Initialize weights in final layer?
    # This will bias the softmax output layer to output class confidences
    # equal to the class frequency
    if hparams["build"].get("biased_output_layer"):
        from MultiViewUNet.utils.utils import set_bias_weights
        set_bias_weights(layer=org_model.layers[-1],
                         train_loader=train.image_pair_loader,
                         class_counts=hparams.get("class_counts"),
                         logger=logger)

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
    _ = trainer.fit(train, val, hparams=hparams, no_im=no_im, **hparams["fit"])

    # Save final model weights (usually not used, but maybe....?)
    if not os.path.exists("%s/model" % base_path):
        os.mkdir("%s/model" % base_path)
    model_path = "%s/model/model_weights.h5" % base_path
    logger("Saving current model to: %s" % model_path)
    org_model.save_weights(model_path)

    # Plot learning curves
    from MultiViewUNet.utils.plotting import plot_training_curves
    try:
        plot_training_curves(os.path.join(base_path, "logs", "training.csv"),
                             os.path.join(base_path, "logs",
                                          "learning_curve.png"),
                             logy=True)
    except Exception as e:
        logger("Could not plot learning curves due to error:")
        logger(e)


if __name__ == "__main__":

    # Get settings
    # Project base path etc.
    args = vars(get_argparser().parse_args())
    base_path = os.path.abspath(args["project_dir"])
    overwrite = args["overwrite"]
    num_GPUs = args["num_GPUs"]

    # Check path
    validate_path(base_path)
    print("Fitting model in path:\n%s" % base_path)

    # Define Logger object
    # Also checks if the model in the project folder has already been fit
    from MultiViewUNet.logging import Logger
    try:
        logger = Logger(base_path, print_to_screen=True, overwrite_existing=overwrite)
    except OSError:
        print("\n[*] A training session at '%s' already exists."
              "\n    Use the --overwrite flag to overwrite." % base_path)
        sys.exit(0)

    # Import GPUMonitor and start process (forks, therefor called early)
    from MultiViewUNet.utils.system import GPUMonitor

    # Initialize GPUMonitor in separate fork now before memory builds up
    if num_GPUs >= 0:
        gpu_mon = GPUMonitor(logger)
    else:
        gpu_mon = None

    try:
        run(base_path, gpu_mon, num_GPUs)
    except Exception as e:
        gpu_mon.stop()
        raise e
