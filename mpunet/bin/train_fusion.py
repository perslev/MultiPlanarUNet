"""
mpunet fusion model training script
Optimizes a mpunet fusion model in a specified project folder

The fusion model takes a set of class-probability vectors, one from each view
of a mpunet model, and merges them into a single class-probability
vector. The fusion model learns to apply additional or reduced weight to each
class as seen in each view of the MultiPlanar model.

The final fusion computes a simple function, but training is relatively compute
intensive as the mpunet model needs to be evaluated over all the
training images in all views. To keep memory load down, the model is optimized
over sub-sets of the training data one-by-one, controlled by the
--images_per_round flag.

Typical usage:
--------------
cd my_project
mp train_fusion
--------------
"""

import random
import numpy as np
import os
from mpunet.image import ImagePairLoader
from mpunet.models import FusionModel
from mpunet.models.model_init import init_model
from mpunet.hyperparameters import YAMLHParams
from mpunet.utils import await_and_set_free_gpu, get_best_model, \
                                  create_folders, highlighted, set_gpu
from mpunet.utils.fusion import predict_and_map, stack_collections
from mpunet.interpolation.sample_grid import get_voxel_grid_real_space
from mpunet.logging import Logger
from mpunet.callbacks import ValDiceScores, PrintLayerWeights
from mpunet.evaluate.metrics import sparse_fg_precision, \
                                             sparse_fg_recall

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from sklearn.utils import shuffle
from argparse import ArgumentParser


def get_argparser():
    parser = ArgumentParser(description='Fit a fusion model for a '
                                        'mpunet project')
    parser.add_argument("--project_dir", type=str, default="./",
                        help='path to mpunet project folder')
    parser.add_argument("--overwrite", action='store_true',
                        help='overwrite previous fusion weights')
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help='Number of GPUs to assign to this job')
    parser.add_argument("--images_per_round", type=int, default=5,
                        help="Number of images to train on in each sub-round."
                             " Larger numbers should be preferred but "
                             "requires potentially large amounts of memory."
                             " Defaults to 5.")
    parser.add_argument("--batch_size", type=int, default=2**17,
                        help="Sets the batch size used in the fusion model "
                             "training process. Lowering this number may "
                             "resolve GPU memory issues. Defaults to 2**17.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train on each subset of "
                             "images (default=30)")
    parser.add_argument("--early_stopping", type=int, default=3,
                        help="Number of non-improving epochs before premature "
                             "training round termination is invoked. "
                             "(default=3)")
    parser.add_argument("--continue_training", action='store_true')
    parser.add_argument("--force_GPU", type=str, default="")
    parser.add_argument("--eval_prob", type=float, default=1.0,
                        help="Perform evaluation on only a fraction of the"
                             " computed views (to speed up run-time)")
    parser.add_argument("--wait_for", type=str, default="",
                        help="Waiting for PID to terminate before starting "
                             "training process.")
    parser.add_argument("--dice_weight", type=str, default="uniform")
    return parser


def log(logger, hparams, views, weights, fusion_weights):
    logger("N classes:       %s" % hparams["build"].get("n_classes"))
    logger("Scaler:          %s" % hparams["fit"].get("scaler"))
    logger("Crop:            %s" % hparams["fit"].get("crop_to"))
    logger("Downsample:      %s" % hparams["fit"].get("downsample_to"))
    logger("CF factor:       %s" % hparams["build"].get("complexity_factor"))
    logger("Views:           %s" % views)
    logger("Weights:         %s" % weights)
    logger("Fusion weights:  %s" % fusion_weights)


def contains_all_images(sets, images):
    """
    TODO
    """
    l = [i for s in sets for i in s]
    return all([m in l for m in images])


def make_sets(images, sub_size, N):
    """
    TODO
    """
    sets = []
    for i in range(N):
        sets.append(set(np.random.choice(images, sub_size, replace=False)))
    return sets


def _run_fusion_training(sets, logger, hparams, min_val_images, is_validation,
                         views, n_classes, unet, fusion_model, early_stopping,
                         fm_batch_size, epochs, eval_prob, fusion_weights_path):
    """
    TODO
    """

    for _round, _set in enumerate(sets):
        s = "Set %i/%i:\n%s" % (_round + 1, len(sets), _set)
        logger("\n%s" % highlighted(s))

        # Reload data
        images = ImagePairLoader(**hparams["val_data"])
        if len(images) < min_val_images:
            images.add_images(ImagePairLoader(**hparams["train_data"]))

        # Get list of ImagePair objects to run on
        image_set_dict = {m.identifier: m for m in images if m.identifier in _set}

        # Set scaler and bg values
        images.set_scaler_and_bg_values(
            bg_value=hparams.get_from_anywhere('bg_value'),
            scaler=hparams.get_from_anywhere('scaler'),
            compute_now=False
        )

        # Init LazyQueue and get its sequencer
        from mpunet.sequences.utils import get_sequence
        seq = get_sequence(data_queue=images,
                           is_validation=True,
                           views=views,
                           **hparams["fit"], **hparams["build"])

        # Fetch points from the set images
        points_collection = []
        targets_collection = []
        N_im = len(image_set_dict)
        for num_im, image_id in enumerate(list(image_set_dict.keys())):
            logger("")
            logger(
                highlighted("(%i/%i) Running on %s (%s)" % (num_im + 1, N_im,
                                                            image_id, "val" if
                                                            is_validation[
                                                                image_id] else "train")))

            with seq.image_pair_queue.get_image_by_id(image_id) as image:
                # Get voxel grid in real space
                voxel_grid_real_space = get_voxel_grid_real_space(image)

                # Get array to store predictions across all views
                targets = image.labels.reshape(-1, 1)
                points = np.empty(shape=(len(targets), len(views), n_classes),
                                  dtype=np.float32)
                points.fill(np.nan)

                # Predict on all views
                for k, v in enumerate(views):
                    print("\n%s" % "View: %s" % v)
                    points[:, k, :] = predict_and_map(model=unet,
                                                      seq=seq,
                                                      image=image,
                                                      view=v,
                                                      voxel_grid_real_space=voxel_grid_real_space,
                                                      n_planes='same+20',
                                                      targets=targets,
                                                      eval_prob=eval_prob).reshape(-1, n_classes)

                # add to collections
                points_collection.append(points)
                targets_collection.append(targets)
            print(image.is_loaded)

        # Stack points into one matrix
        logger("Stacking points...")
        X, y = stack_collections(points_collection, targets_collection)

        # Shuffle train
        print("Shuffling points...")
        X, y = shuffle(X, y)

        print("Getting validation set...")
        val_ind = int(0.20*X.shape[0])
        X_val, y_val = X[:val_ind], y[:val_ind]
        X, y = X[val_ind:], y[val_ind:]

        # Prepare dice score callback for validation data
        val_cb = ValDiceScores((X_val, y_val), n_classes, 50000, logger)

        # Callbacks
        cbs = [val_cb,
               CSVLogger(filename="logs/fusion_training.csv",
                         separator=",", append=True),
               PrintLayerWeights(fusion_model.layers[-1], every=1,
                                 first=1000, per_epoch=True, logger=logger)]

        es = EarlyStopping(monitor='val_dice', min_delta=0.0,
                           patience=early_stopping, verbose=1, mode='max')
        cbs.append(es)

        # Start training
        try:
            fusion_model.fit(X, y, batch_size=fm_batch_size,
                             epochs=epochs, callbacks=cbs, verbose=1)
        except KeyboardInterrupt:
            pass
        fusion_model.save_weights(fusion_weights_path)


def entry_func(args=None):

    # Project base path
    args = vars(get_argparser().parse_args(args))
    basedir = os.path.abspath(args["project_dir"])
    overwrite = args["overwrite"]
    continue_training = args["continue_training"]
    eval_prob = args["eval_prob"]
    await_PID = args["wait_for"]
    dice_weight = args["dice_weight"]
    print("Fitting fusion model for project-folder: %s" % basedir)

    # Minimum images in validation set before also using training images
    min_val_images = 15

    # Fusion model training params
    epochs = args['epochs']
    fm_batch_size = args["batch_size"]

    # Early stopping params
    early_stopping = args["early_stopping"]

    # Wait for PID?
    if await_PID:
        from mpunet.utils import await_PIDs
        await_PIDs(await_PID)

    # Fetch GPU(s)
    num_GPUs = args["num_GPUs"]
    force_gpu = args["force_GPU"]
    # Wait for free GPU
    if not force_gpu:
        await_and_set_free_gpu(N=num_GPUs, sleep_seconds=120)
    else:
        set_gpu(force_gpu)

    # Get logger
    logger = Logger(base_path=basedir, active_file="train_fusion",
                    overwrite_existing=overwrite)

    # Get YAML hyperparameters
    hparams = YAMLHParams(os.path.join(basedir, "train_hparams.yaml"))

    # Get some key settings
    n_classes = hparams["build"]["n_classes"]

    if hparams["build"]["out_activation"] == "linear":
        # Trained with logit targets?
        hparams["build"]["out_activation"] = "softmax" if n_classes > 1 else "sigmoid"

    # Get views
    views = np.load("%s/views.npz" % basedir)["arr_0"]
    del hparams["fit"]["views"]

    # Get weights and set fusion (output) path
    weights = get_best_model("%s/model" % basedir)
    weights_name = os.path.splitext(os.path.split(weights)[-1])[0]
    fusion_weights = "%s/model/fusion_weights/" \
                     "%s_fusion_weights.h5" % (basedir, weights_name)
    create_folders(os.path.split(fusion_weights)[0])

    # Log a few things
    log(logger, hparams, views, weights, fusion_weights)

    # Check if exists already...
    if not overwrite and os.path.exists(fusion_weights):
        from sys import exit
        print("\n[*] A fusion weights file already exists at '%s'."
              "\n    Use the --overwrite flag to overwrite." % fusion_weights)
        exit(0)

    # Load validation data
    images = ImagePairLoader(**hparams["val_data"], logger=logger)
    is_validation = {m.identifier: True for m in images}

    # Define random sets of images to train on simul. (cant be all due
    # to memory constraints)
    image_IDs = [m.identifier for m in images]

    if len(images) < min_val_images:
        # Pick N random training images
        diff = min_val_images - len(images)
        logger("Adding %i training images to set" % diff)

        # Load the training data and pick diff images
        train = ImagePairLoader(**hparams["train_data"], logger=logger)
        indx = np.random.choice(np.arange(len(train)), diff, replace=diff > len(train))

        # Add the images to the image set set
        train_add = [train[i] for i in indx]
        for m in train_add:
            is_validation[m.identifier] = False
            image_IDs.append(m.identifier)
        images.add_images(train_add)

    # Append to length % sub_size == 0
    sub_size = args["images_per_round"]
    rest = int(sub_size*np.ceil(len(image_IDs)/sub_size)) - len(image_IDs)
    if rest:
        image_IDs += list(np.random.choice(image_IDs, rest, replace=False))

    # Shuffle and split
    random.shuffle(image_IDs)
    sets = [set(s) for s in np.array_split(image_IDs, len(image_IDs)/sub_size)]
    assert(contains_all_images(sets, image_IDs))

    # Define fusion model (named 'org' to store reference to orgiginal model if
    # multi gpu model is created below)
    fusion_model = FusionModel(n_inputs=len(views), n_classes=n_classes,
                               weight=dice_weight,
                               logger=logger, verbose=False)

    if continue_training:
        fusion_model.load_weights(fusion_weights)
        print("\n[OBS] CONTINUED TRAINING FROM:\n", fusion_weights)

    import tensorflow as tf
    with tf.distribute.MirroredStrategy().scope():
        # Define model
        unet = init_model(hparams["build"], logger)
        print("\n[*] Loading weights: %s\n" % weights)
        unet.load_weights(weights, by_name=True)

    # Compile the model
    logger("Compiling...")
    metrics = ["sparse_categorical_accuracy", sparse_fg_precision, sparse_fg_recall]
    fusion_model.compile(optimizer=Adam(lr=1e-3),
                         loss=fusion_model.loss,
                         metrics=metrics)
    fusion_model._log()

    try:
        _run_fusion_training(sets, logger, hparams, min_val_images,
                             is_validation, views, n_classes, unet,
                             fusion_model, early_stopping, fm_batch_size,
                             epochs, eval_prob, fusion_weights)
    except KeyboardInterrupt:
        pass
    finally:
        if not os.path.exists(os.path.split(fusion_weights)[0]):
            os.mkdir(os.path.split(fusion_weights)[0])
        # Save fusion model weights
        # OBS: Must be original model if multi-gpu is performed!
        fusion_model.save_weights(fusion_weights)


if __name__ == "__main__":
    entry_func()
