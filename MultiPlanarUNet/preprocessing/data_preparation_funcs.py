from MultiPlanarUNet.logging import ScreenLogger
from MultiPlanarUNet.image.auditor import Auditor
from MultiPlanarUNet.image import ImagePairLoader
import numpy as np
import os


"""
A collection of functions that prepares data for feeding to various models in
the MultiPlanarUNet.models packages. All functions should follow the following
specification:

f(hparams, just_one, no_val, logger, mtype, base_path), 
--> return train, val, hparams
See function docstrings for further description 

and return an object of type keras.Sequence feeding valid train and validation
inputs to the model. Should also return the hparams object for clarity.
"""


def _base_loader_func(hparams, just_one, no_val, logger, mtype):
    """
    Base loader function used for all models. This function performs a series
    of actions:

    1) Loads train, val and test data according to hparams
    2) Performs a hparam audit on the training + validation images
    3) If any audited parameters were not manually specified, updates the
       hparams dict with the audited values and updates the YAML file on disk
    4) If just_one, discards all but the first training and validation images
    5) Initializes a ImageQueue object on the training and validation data
       if needed.

    Args:
        hparams:   A MultiPlanarUNet.train.YAMLHParams object
        just_one:  A bool specifying whether to keep only the first train and
                   validation samples (for quick testing purposes)
        no_val:    A bool specifying whether to omit validation data entirely
                   Note: This setting applies even if validation data is
                   specified in the YAMLHparams object
        logger:    A MultiPlanarUNet.logger object
        mtype:     A string identifier for the dimensionality of the model,
                   currently either '2d', '3d'
                   (upper/lower ignored)

    Returns:
        train_data: An ImagePairLoader object storing the training images
        val_data:   An ImagePairLoader object storing the validation images, or
                    an 'empty' ImagePairLoader storing no images if no_val=True
        logger:     The passed logger object or a ScreenLogger object
        auditor:    An auditor object storing statistics on the training data
    """

    # Get basic ScreenLogger if no logger is passed
    logger = logger or ScreenLogger()
    logger("Looking for images...")

    # Get data loaders
    train_data = ImagePairLoader(logger=logger, **hparams["train_data"])
    val_data = ImagePairLoader(logger=logger, **hparams["val_data"])

    # Audit
    lab_paths = train_data.label_paths + val_data.label_paths
    auditor = Auditor(train_data.image_paths + val_data.image_paths,
                      nii_lab_paths=lab_paths, logger=logger,
                      dim_3d=hparams.get_from_anywhere("dim") or 64,
                      hparams=hparams)

    # Fill hparams with audited values, if not specified manually
    auditor.fill(hparams, mtype)

    # Add augmented data?
    if hparams.get("aug_data"):
        aug_data = hparams["aug_data"]
        if "include" not in aug_data:
            logger.warn("Found 'aug_data' group, but the group does not "
                        "contain the key 'include', which is required in "
                        "version 2.0 and above. OBS: Not including aug data!")
        elif aug_data["include"]:
            logger("\n[*] Adding augmented data with weight ", aug_data["sample_weight"])
            train_data.add_augmented_images(ImagePairLoader(logger=logger, **aug_data))

    if just_one:
        # For testing purposes, run only on one train and one val image?
        logger("[**NOTTICE**] Only running on first train & val samples.")
        train_data.images = [train_data.images[0]]
        val_data.images = [val_data.images[0]]
    if no_val:
        # Run without performing validation (even if specified in param file)
        val_data.images = []

    # Set queue object if necessary
    train_data.set_queue(hparams["train_data"].get("max_load"))
    val_data.set_queue(hparams["val_data"].get("max_load"))

    return train_data, val_data, logger, auditor


def add_class_weights_to_hparams(train_data, hparams):
    if hparams["fit"]["class_weights"] is True:
        # If train data is queued, unload each image after class counting
        unload = bool(train_data.queue)

        # Compute the class weights and also return the counts
        weights, counts = train_data.get_class_weights(as_array=True,
                                                       return_counts=True,
                                                       unload=unload)
        hparams["fit"]["class_weights"] = weights
        hparams["fit"]["class_counts"] = counts


def load_or_create_views(hparams, continue_training, logger, base_path, auditor):
    views = hparams["fit"]["views"]
    if not continue_training:
        if isinstance(views, int):
            from MultiPlanarUNet.interpolation.sample_grid import sample_random_views_with_angle_restriction
            views = sample_random_views_with_angle_restriction(views, 60,
                                                               auditor=auditor,
                                                               logger=logger)
            hparams["fit"]["views"] = views
        elif isinstance(views, (list, tuple)):
            if not hparams["fit"]["intrp_style"] == "iso_live":
                logger("[Note] Pre-adding noise to views (SD: %s)" %
                       hparams["fit"]["noise_sd"])
                # Apply noise to views
                from MultiPlanarUNet.utils import add_noise_to_views
                hparams["fit"]["views"] = add_noise_to_views(hparams["fit"]["views"],
                                                             hparams["fit"]["noise_sd"])
                hparams["fit"]["noise_sd"] = False
        else:
            raise ValueError("Invalid 'views' input '%s'. Must be list or "
                             "single integer" % views)
        logger("View SD:     %s" % hparams["fit"].get("noise_sd"))

        # Save views
        np.savez(os.path.join(base_path, "views"), hparams["fit"]["views"])

        # Plot views
        from MultiPlanarUNet.utils.plotting import plot_views
        plot_views(views, os.path.join(base_path, "views.png"))
    else:
        # Fetch views from last session
        view_path = os.path.join(base_path, "views.npz")
        hparams["fit"]["views"] = np.load(view_path)["arr_0"]


def prepare_for_multi_view_unet(hparams, just_one=False, no_val=False,
                                continue_training=False, logger=None,
                                base_path='./'):

    # Load the data
    train_data, val_data, logger, auditor = _base_loader_func(hparams, just_one,
                                                              no_val, logger, "2d")

    # Load or create a set of views (determined by 'continue_training')
    # This function will add the views to hparams["fit"]["views"] and
    # store the views on disk at base_path/views.npz.
    load_or_create_views(hparams=hparams,
                         continue_training=continue_training,
                         logger=logger,
                         base_path=base_path,
                         auditor=auditor)

    # Print views in use
    logger("Views:       N=%i" % len(hparams["fit"]["views"]))
    logger("             %s" % ((" " * 13).join([str(v) + "\n" for v in hparams["fit"]["views"]])))

    # Get keras.Sequence generators for training images
    logger("Preparing views...")
    train = train_data.get_sequencer(n_classes=hparams["build"]["n_classes"],
                                     dim=hparams["build"]["dim"],
                                     is_validation=False, **hparams["fit"])
    val = val_data.get_sequencer(n_classes=hparams["build"]["n_classes"],
                                 dim=hparams["build"]["dim"],
                                 is_validation=True, **hparams["fit"])

    # Compute class weights if specified, added to hparams
    add_class_weights_to_hparams(train_data, hparams)
    logger("Class weights: %s" % hparams["fit"].get("class_weights"))
    logger("Class counts: %s" % hparams["fit"].get("class_counts"))

    return train, val, hparams


def prepare_for_3d_unet(hparams, just_one=False, no_val=False, logger=None,
                        continue_training=None, base_path="./"):

    # Load the data
    train_data, val_data, logger, auditor = _base_loader_func(hparams, just_one,
                                                              no_val, logger, "3d")

    # Get 3D patch sequence generators
    train = train_data.get_sequencer(n_classes=hparams["build"]["n_classes"],
                                     dim=hparams["build"]["dim"],
                                     **hparams["fit"])
    val = val_data.get_sequencer(n_classes=hparams["build"]["n_classes"],
                                 dim=hparams["build"]["dim"],
                                 is_validation=True, **hparams["fit"])

    # Compute class weights if specified, added to hparams
    add_class_weights_to_hparams(train_data, hparams)
    logger("Class weights: %s" % hparams["fit"].get("class_weights"))
    logger("Class counts: %s" % hparams["fit"].get("class_counts"))

    return train, val, hparams


def prepare_for_multi_task_2d(hparams, just_one=False, no_val=False, logger=None,
                              continue_training=None, base_path="./"):
    from MultiPlanarUNet.hyperparameters import YAMLHParams
    # Get image loaders for all tasks
    tasks = []
    for name, task_hparam_file in zip(*hparams["tasks"].values()):
        task_hparams = YAMLHParams(task_hparam_file)
        type_ = 'multi_task_2d'
        train_data, val_data, logger, auditor = _base_loader_func(task_hparams,
                                                                  just_one,
                                                                  no_val,
                                                                  logger,
                                                                  mtype=type_)

        task = {
            "name": name,
            "hparams": task_hparams,
            "train": train_data,
            "val": val_data
        }
        tasks.append(task)

    # Set various build hparams
    fetch = ("n_classes", "dim", "n_channels",
             "out_activation", "biased_output_layer")
    field = "task_specifics"
    for f in fetch:
        hparams["build"][f] = tuple([t["hparams"][field][f] for t in tasks])

    # Add task names to build dir
    hparams["build"]["task_names"] = hparams["tasks"]["task_names"]

    # Load or create a set of views (determined by 'continue_training')
    # This function will add the views to hparams["fit"]["views"] and
    # store the views on disk at base_path/views.npz.
    load_or_create_views(hparams=hparams,
                         continue_training=continue_training,
                         logger=logger,
                         base_path=base_path,
                         auditor=None)

    # Get per-task sequences
    train_seqs = []
    val_seqs = []
    for task in tasks:
        logger("Fetching sequences for task %s" % task["name"])

        # Create hparams dict that combines the common hparams and
        # task-specific hparams
        task_hparams = dict(hparams["fit"])
        task_hparams.update(task["hparams"]["task_specifics"])

        # Get sequences for training and validation
        train = task["train"].get_sequencer(is_validation=False, **task_hparams)
        val = task["val"].get_sequencer(is_validation=True, **task_hparams)

        # Add to lists
        train_seqs.append(train)
        val_seqs.append(val)

    # Create the training and validation sequences
    # These will produce batches shared across the N tasks
    from MultiPlanarUNet.sequences import MultiTaskSequence
    train = MultiTaskSequence(train_seqs, hparams["build"]["task_names"])
    val = MultiTaskSequence(val_seqs, hparams["build"]["task_names"])

    return train, val, hparams
