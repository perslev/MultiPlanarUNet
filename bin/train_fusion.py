from MultiViewUNet.image import ImagePairLoader
from MultiViewUNet.models import UNet, FusionModel
from MultiViewUNet.train import YAMLHParams
from MultiViewUNet.utils import await_and_set_free_gpu, get_best_model, \
                                create_folders, highlighted, set_gpu
from MultiViewUNet.utils.fusion import predict_volume, map_real_space_pred
from MultiViewUNet.interpolation.sample_grid import get_voxel_grid_real_space
from MultiViewUNet.logging import Logger
from MultiViewUNet.evaluate import dice_all
from MultiViewUNet.callbacks import ValDiceScores, PrintLayerWeights
from tensorflow.keras.optimizers import Adam
from MultiViewUNet.evaluate.metrics import sparse_fg_precision, sparse_fg_recall

from sklearn.utils import shuffle

from keras.callbacks import CSVLogger, EarlyStopping

from argparse import ArgumentParser
import random
import numpy as np
import os


def get_argparser():
    parser = ArgumentParser(description='Fit the fusion model stage of a '
                                        'MultiViewUNet project')
    parser.add_argument("--project_dir", type=str, default="./",
                        help='path to MultiViewUNet project folder')
    parser.add_argument("--overwrite", action='store_true',
                        help='overwrite previous fusion weights')
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help='Number of GPUs to assign to this job')
    parser.add_argument("--continue_training", action='store_true')
    parser.add_argument("--force_GPU", type=str, default="")
    parser.add_argument("--eval_prob", type=float, default=1.0,
                        help="Perform evaluation on only a fraction of the"
                             " computed views (to speed up run-time)")
    parser.add_argument("--wait_for", type=str, default="",
                        help="Waiting for PID to terminate before starting "
                             "training process.")
    parser.add_argument("--dice_weight", type=str, default="Simple")
    return parser


def log():
    logger("N classes:       %s" % hparams["build"].get("n_classes"))
    logger("Scaler:          %s" % hparams["fit"].get("scaler"))
    logger("Crop:            %s" % hparams["fit"].get("crop_to"))
    logger("Downsample:      %s" % hparams["fit"].get("downsample_to"))
    logger("CF factor:       %s" % hparams["build"].get("complexity_factor"))
    logger("Views:           %s" % views)
    logger("Weights:         %s" % weights)
    logger("Fusion weights:  %s" % fusion_weights)


def contains_all_images(sets, images):
    l = [i for s in sets for i in s]
    return all([m in l for m in images])


def make_sets(images, sub_size, N):
    sets = []
    for i in range(N):
        sets.append(set(np.random.choice(images, sub_size, replace=False)))
    return sets


if __name__ == "__main__":

    # Minimum images in validation set before also using training images
    min_val_images = 15

    # Approximate number of images in each sub-split of data
    sub_size = 20

    # Fusion model training params
    epochs = 10
    fm_batch_size = 1000000

    # Early stopping params
    early_stopping = 4
    improve_delta = 0.0

    # Project base path
    args = vars(get_argparser().parse_args())
    basedir = os.path.abspath(args["project_dir"])
    overwrite = args["overwrite"]
    continue_training = args["continue_training"]
    eval_prob = args["eval_prob"]
    await_PID = args["wait_for"]
    dice_weight = args["dice_weight"]
    print("Fitting fusion model for project-folder: %s" % basedir)

    # Wait for PID?
    if await_PID:
        from MultiViewUNet.utils import await_PIDs
        await_PIDs(await_PID)

    # Fetch GPU(s)
    num_GPUs = args["num_GPUs"]
    force_gpu = args["force_GPU"]
    # Wait for free GPU
    if not force_gpu:
        await_and_set_free_gpu(N=num_GPUs, sleep_seconds=120)
        num_GPUs = 1
    else:
        set_gpu(force_gpu)
        num_GPUs = len(force_gpu.split(","))

    # Get logger
    logger = Logger(base_path=basedir, active_file="train_fusion",
                    overwrite_existing=overwrite)

    # Get YAML hyperparameters
    hparams = YAMLHParams(os.path.join(basedir, "train_hparams.yaml"))

    # Get some key settings
    n_classes = hparams["build"]["n_classes"]

    if hparams["build"]["out_activation"] == "linear":
        # Trained with sparse, logit targets?
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
    log()

    # Check if exists already...
    if not overwrite and os.path.exists(fusion_weights):
        from sys import exit
        print("\n[*] A fusion weights file already exists at '%s'."
              "\n    Use the --overwrite flag to overwrite." % fusion_weights)
        exit(0)

    # Load validation data
    images = ImagePairLoader(**hparams["val_data"], logger=logger)
    is_validation = {m.id: True for m in images}

    # Define random sets of images to train on simul. (cant be all due
    # to memory constraints)
    image_IDs = [m.id for m in images]

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
            is_validation[m.id] = False
            image_IDs.append(m.id)
        images.add_images(train_add)

    # Append to length % sub_size == 0
    rest = int(sub_size*np.ceil(len(image_IDs)/sub_size)) - len(image_IDs)
    if rest:
        image_IDs += list(np.random.choice(image_IDs, rest, replace=False))

    # Shuffle and split
    random.shuffle(image_IDs)
    sets = [set(s) for s in np.array_split(image_IDs, len(image_IDs)/sub_size)]
    assert(contains_all_images(sets, image_IDs))

    # Define fusion model (named 'org' to store reference to orgiginal model if
    # multi gpu model is created below)
    fusion_model_org = FusionModel(n_inputs=len(views), n_classes=n_classes,
                                   weight=dice_weight,
                                   logger=logger, verbose=False)

    # from MultiViewUNet.utils.utils import set_bias_weights
    # set_bias_weights(layer=fusion_model_org.layers[-1],
    #                  train_loader=images,
    #                  class_counts=hparams.get("class_counts"),
    #                  logger=logger)

    if continue_training:
        fusion_model_org.load_weights(fusion_weights)
        print("\n[OBS] CONTINUED TRAINING FROM:\n", fusion_weights)

    # Define model
    unet = UNet(**hparams["build"])
    print("\n[*] Loading weights: %s\n" % weights)
    unet.load_weights(weights)

    if num_GPUs > 1:
        from tensorflow.keras.utils import multi_gpu_model

        # Set for predictor model
        n_classes = unet.n_classes
        unet = multi_gpu_model(unet, gpus=num_GPUs)
        unet.n_classes = n_classes

        # Set for fusion model
        fusion_model = multi_gpu_model(fusion_model_org, gpus=num_GPUs)
    else:
        fusion_model = fusion_model_org

    # Compile the model
    logger("Compiling...")
    metrics = ["sparse_categorical_accuracy", sparse_fg_precision, sparse_fg_recall]
    fusion_model.compile(optimizer=Adam(lr=1e-3), loss=fusion_model_org.loss, metrics=metrics)
    fusion_model_org._log()

    try:
        for _round, _set in enumerate(sets):
            s = "Set %i/%i:\n%s" % (_round+1, len(sets), _set)
            logger("\n%s" % highlighted(s))

            # Reload data
            images = ImagePairLoader(**hparams["val_data"])
            if len(images) < min_val_images:
                images.add_images(ImagePairLoader(**hparams["train_data"]))

            # Get list of ImagePair objects to run on
            image_set_dict = {m.id: m for m in images if m.id in _set}

            # Fetch points from the set images
            points_collection = []
            targets_collection = []
            N_im = len(image_set_dict)
            for num_im, image_id in enumerate(list(image_set_dict.keys())):
                logger("")
                logger(highlighted("(%i/%i) Running on %s (%s)" % (num_im+1, N_im,
                                                                   image_id, "val" if is_validation[image_id] else "train")))

                # Set the current ImagePair
                image = image_set_dict[image_id]
                images.images = [image]

                # Load views
                kwargs = hparams["fit"]
                kwargs.update(hparams["build"])
                seq = images.get_views(views=views, **kwargs)

                # Get voxel grid in real space
                voxel_grid_real_space = get_voxel_grid_real_space(image)

                # Get array to store predictions across all views
                targets = image.labels.reshape(-1, 1)
                points = np.empty(shape=(len(targets), len(views), n_classes),
                                  dtype=np.float32)
                points.fill(np.nan)

                # Predict on all views
                for k, v in enumerate(views):
                    logger("\n%s" % highlighted("View: %s" % v))

                    # Sample planes from the image at grid_real_space grid
                    # in real space (scanner RAS) coordinates.
                    X, y, grid, inv_basis = seq.get_view_from(image.id, v,
                                                              n_planes='same+20')

                    # Predict on volume using model
                    pred = predict_volume(unet, X, axis=2,
                                          batch_size=seq.batch_size)

                    # Map the real space coordiante predictions to nearest
                    # real space coordinates defined on voxel grid
                    points[:, k, :] = map_real_space_pred(pred, grid, inv_basis,
                                                          voxel_grid_real_space,
                                                          method="nearest").reshape(-1, n_classes)

                    if np.random.rand() <= eval_prob:
                        # Print dice scores
                        logger("Computing evaluations...")
                        logger("View dice scores:   ", dice_all(y, pred.argmax(-1),
                                                                ignore_zero=False))
                        logger("Mapped dice scores: ", dice_all(targets,
                                                                points[:, k, :].argmax(-1),
                                                                ignore_zero=False))
                    else:
                        logger("Skipping evaluation for this view... "
                               "(eval_prob=%.3f)" % eval_prob)

                # Clean up a bit
                del image_set_dict[image_id]
                del image  # Should be GC at this point anyway

                # add to collections
                points_collection.append(points)
                targets_collection.append(targets)

            # Stack points into one matrix
            logger("Stacking points...")
            n_points = sum([x.shape[0] for x in points_collection])
            X = np.empty(shape=(n_points, len(views), n_classes),
                         dtype=points_collection[0].dtype)
            y = np.empty(shape=(n_points, 1),
                         dtype=targets_collection[0].dtype)

            c = 0
            len_collection = len(points_collection)
            for i in range(len_collection):
                print("  %i/%i" % (i+1, len_collection),
                      end="\r", flush=True)
                Xs = points_collection.pop()
                X[c:c+len(Xs)] = Xs
                y[c:c+len(Xs)] = targets_collection.pop()
                c += len(Xs)
            print("")

            # Take random split of validation data
            n_val = int(n_points*0.20)
            val_ind = np.random.choice(np.arange(n_points), size=n_val)
            X_val, y_val = X[val_ind], y[val_ind]

            # Get inverse for training
            print("Getting validation set...")
            X, y = np.delete(X, val_ind, axis=0), np.delete(y, val_ind, axis=0)

            # Shuffle train
            print("Shuffling...")
            X, y = shuffle(X, y)

            # Prepare dice score callback for validation data
            val_cb = ValDiceScores((X_val, y_val), n_classes, 50000, logger)

            # Callbacks
            cbs = [val_cb,
                   CSVLogger(filename="logs/fusion_training.csv",
                             separator=",", append=True),
                   PrintLayerWeights(fusion_model_org.layers[-1], every=1,
                                     first=1000, per_epoch=True, logger=logger)]

            es = EarlyStopping(monitor='val_dice', min_delta=improve_delta,
                               patience=early_stopping, verbose=1, mode='max')
            cbs.append(es)

            # Start training
            try:
                fusion_model.fit(X, y, batch_size=fm_batch_size,
                                 epochs=epochs, callbacks=cbs, verbose=1)
            except KeyboardInterrupt:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        if not os.path.exists(os.path.split(fusion_weights)[0]):
            os.mkdir(os.path.split(fusion_weights)[0])
        # Save fusion model weights
        # OBS: Must be original model if multi-gpu is performed!
        fusion_model_org.save_weights(fusion_weights)
