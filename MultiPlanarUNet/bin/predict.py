"""
Prediction/evaluation script

Mathias Perslev
March 2018
"""

import os
from argparse import ArgumentParser
import readline

readline.parse_and_bind('tab: complete')


def get_argparser():
    parser = ArgumentParser(description='Predict using a MultiPlanarUNet model.')
    parser.add_argument("--project_dir", type=str, default="./",
                        help='Path to MultiPlanarUNet project folder')
    parser.add_argument("-f", help="Predict on a single file")
    parser.add_argument("-l", help="Optional single label file to use with -f")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory storing data. "
                             "Must contain sub-folder 'images'. Optional, "
                             "otherwise test data folder from "
                             "train_parameters.yaml is used.")
    parser.add_argument("--out_dir", type=str, default="predictions",
                        help="Output folder to store results")
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help="Number of GPUs to use for this job")
    parser.add_argument("--analytical", action='store_true',
                        help="Use analytically derived fusion weights "
                             "over fusion layer approach "
                             "(for backwards compatibility)")
    parser.add_argument("--majority", action="store_true",
                        help="No fusion model, sum softmax probabilities into"
                             " one volume and argmax in the end. Exclusive "
                             "with --analytical.")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite previous results at the output folder')
    parser.add_argument("--no_eval", action="store_true",
                        help="Perform no evaluation of the prediction performance. "
                             "No label files loaded when this flag applies.")
    parser.add_argument("--eval_prob", type=float, default=1.0,
                        help="Perform evaluation on only a fraction of the"
                             " computed views (to speed up run-time). OBS: "
                             "always performs evaluation on the combined "
                             "predictions.")
    parser.add_argument("--force_GPU", type=str, default="")
    parser.add_argument("--save_input_files", action="store_true",
                        help="Save in addition to the predicted volume the "
                             "input image and label files to the output dir)")
    parser.add_argument("--no_argmax", action="store_true",
                        help="Do not argmax prediction volume prior to save.")
    parser.add_argument("--on_val", action="store_true",
                        help="Evaluate on the validation set instead of test")
    parser.add_argument("--wait_for", type=str, default="",
                        help="Waiting for PID to terminate before starting "
                             "training process.")
    return parser


def validate_folders(base_dir, out_dir, overwrite):

    # Check base (model) dir contains required files
    must_exist = ("train_hparams.yaml", "views.npz",
                  "model")
    for p in must_exist:
        p = os.path.join(base_dir, p)
        if not os.path.exists(p):
            from sys import exit
            print("[*] Invalid MultiPlanarUNet project folder: '%s'"
                  "\n    Needed file/folder '%s' not found." % (base_dir, p))
            exit(0)

    # Check if output folder already exists
    if not overwrite and os.path.exists(out_dir):
        from sys import exit
        print("[*] Output directory already exists at: '%s'"
              "\n  Use --overwrite to overwrite" % out_dir)
        exit(0)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


def save_nii_files(combined, image, nii_res_dir, save_input_files):
    from MultiPlanarUNet.utils import create_folders
    import nibabel as nib
    import os

    # Extract data if nii files
    try:
        combined = combined.get_data()
    except AttributeError:
        combined = nib.Nifti1Image(combined, affine=image.affine)

    volumes = [combined, image.image_obj, image.labels_obj]
    labels = ["%s_PRED.nii.gz" % image.id, "%s_IMAGE.nii.gz" % image.id,
              "%s_LABELS.nii.gz" % image.id]

    if not save_input_files:
        volumes = volumes[:1]
        labels = labels[:1]
        p = os.path.abspath(nii_res_dir)  # Save file directly in nii_res_dir
    else:
        # Create sub-folder under nii_res_dir
        p = os.path.join(nii_res_dir, image.id)
    create_folders(p)

    # Save
    for nii, fname in zip(volumes, labels):
        try:
            nib.save(nii, "%s/%s" % (p, fname))
        except AttributeError:
            # No labels file?
            pass


def entry_func(args=None):

    # Get command line arguments
    args = vars(get_argparser().parse_args(args))
    base_dir = os.path.abspath(args["project_dir"])
    analytical = args["analytical"]
    majority = args["majority"]
    _file = args["f"]
    label = args["l"]
    await_PID = args["wait_for"]
    eval_prob = args["eval_prob"]
    if analytical and majority:
        raise ValueError("Cannot specify both --analytical and --majority.")

    # Get settings from YAML file
    from MultiPlanarUNet.train.hparams import YAMLHParams
    hparams = YAMLHParams(os.path.join(base_dir, "train_hparams.yaml"))

    if not _file:
        try:
            # Data specified from command line?
            data_dir = os.path.abspath(args["data_dir"])

            # Set with default sub dirs
            hparams["test_data"] = {"base_dir": data_dir, "img_subdir": "images",
                                    "label_subdir": "labels"}
        except (AttributeError, TypeError):
            data_dir = hparams["test_data"]["base_dir"]
    else:
        data_dir = False
    out_dir = os.path.abspath(args["out_dir"])
    overwrite = args["overwrite"]
    predict_mode = args["no_eval"]
    save_input_files = args["save_input_files"]
    no_argmax = args["no_argmax"]
    on_val = args["on_val"]

    # Check if valid dir structures
    validate_folders(base_dir, out_dir, overwrite)

    # Import all needed modules (folder is valid at this point)
    import numpy as np
    from MultiPlanarUNet.image import ImagePairLoader, ImagePair
    from MultiPlanarUNet.models import FusionModel
    from MultiPlanarUNet.models.model_init import init_model
    from MultiPlanarUNet.utils import await_and_set_free_gpu, get_best_model, \
                                    create_folders, pred_to_class, set_gpu
    from MultiPlanarUNet.logging import init_result_dicts, save_all
    from MultiPlanarUNet.evaluate import dice_all
    from MultiPlanarUNet.utils.fusion import predict_volume, map_real_space_pred
    from MultiPlanarUNet.interpolation.sample_grid import get_voxel_grid_real_space

    # Wait for PID?
    if await_PID:
        from MultiPlanarUNet.utils import await_PIDs
        await_PIDs(await_PID)

    # Set GPU device
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

    # Read settings from the project hyperparameter file
    n_classes = hparams["build"]["n_classes"]

    # Get views
    views = np.load("%s/views.npz" % base_dir)["arr_0"]

    # Force settings
    hparams["fit"]["max_background"] = 1
    hparams["fit"]["test_mode"] = True
    hparams["fit"]["mix_planes"] = False
    hparams["fit"]["live_intrp"] = False
    if "use_bounds" in hparams["fit"]:
        del hparams["fit"]["use_bounds"]
    del hparams["fit"]["views"]

    if hparams["build"]["out_activation"] == "linear":
        # Trained with sparse, logit targets?
        hparams["build"]["out_activation"] = "softmax" if n_classes > 1 else "sigmoid"

    # Set ImagePairLoader object
    if not _file:
        data = "test_data" if not on_val else "val_data"
        image_pair_loader = ImagePairLoader(predict_mode=predict_mode, **hparams[data])
    else:
        predict_mode = not bool(label)
        image_pair_loader = ImagePairLoader(predict_mode=predict_mode,
                                            single_file_mode=True)
        image_pair_loader.add_image(ImagePair(_file, label))

    # Put them into a dict and remove from image_pair_loader to gain more control with
    # garbage collection
    all_images = {image.id: image for image in image_pair_loader.images}
    image_pair_loader.images = None

    """ Define UNet model """
    model_path = get_best_model(base_dir + "/model")
    unet = init_model(hparams["build"])
    unet.load_weights(model_path, by_name=True)

    if num_GPUs > 1:
        from tensorflow.keras.utils import multi_gpu_model
        n_classes = unet.n_classes
        unet = multi_gpu_model(unet, gpus=num_GPUs)
        unet.n_classes = n_classes

    weights_name = os.path.splitext(os.path.split(model_path)[1])[0]
    if not analytical and not majority:
        # Get Fusion model
        fm = FusionModel(n_inputs=len(views), n_classes=n_classes)

        weights = base_dir + "/model/fusion_weights/%s_fusion_weights.h5" % weights_name
        print("\n[*] Loading weights:\n", weights)

        # Load fusion weights
        fm.load_weights(weights)
        print("\nLoaded weights:\n\n%s\n%s\n---" % tuple(fm.layers[-1].get_weights()))

        # Multi-gpu?
        if num_GPUs > 1:
            print("Using multi-GPU model (%i GPUs)" % num_GPUs)
            fm = multi_gpu_model(fm, gpus=num_GPUs)

    # Evaluate?
    if not predict_mode:
        # Prepare dictionary to store results in pd df
        results, detailed_res = init_result_dicts(views, all_images, n_classes)

        # Save to check correct format
        save_all(results, detailed_res, out_dir)

    # Define result paths
    nii_res_dir = os.path.join(out_dir, "nii_files")
    create_folders(nii_res_dir)

    """
    Finally predict on the images
    """
    image_ids = sorted(all_images)
    N_images = len(image_ids)
    for n_image, image_id in enumerate(image_ids):
        print("\n[*] (%i/%s) Running on: %s" % (n_image+1, N_images, image_id))

        # Set image_pair_loader object with only the given file
        image = all_images[image_id]
        image_pair_loader.images = [image]

        # Load views
        kwargs = hparams["fit"]
        kwargs.update(hparams["build"])
        seq = image_pair_loader.get_sequencer(views=views, **kwargs)

        # Get voxel grid in real space
        voxel_grid_real_space = get_voxel_grid_real_space(image)

        # Prepare tensor to store combined prediction
        d = image.image.shape[:-1]
        if not majority:
            combined = np.empty(shape=(len(views), d[0], d[1], d[2], n_classes),
                                dtype=np.float32)
        else:
            combined = np.empty(shape=(d[0], d[1], d[2], n_classes), dtype=np.float32)
        print("Predicting on brain hyper-volume of shape:", combined.shape)

        # Predict for each view
        for n_view, v in enumerate(views):
            print("\n[*] (%i/%i) View: %s" % (n_view+1, len(views), v))
            # for each view, predict on all voxels and map the predictions
            # back into the original coordinate system

            # Sample planes from the image at grid_real_space grid
            # in real space (scanner RAS) coordinates.
            X, y, grid, inv_basis = seq.get_view_from(image.id, v,
                                                      n_planes="same+20")

            # Predict on volume using model
            pred = predict_volume(unet, X, axis=2, batch_size=seq.batch_size)

            # Map the real space coordiante predictions to nearest
            # real space coordinates defined on voxel grid
            mapped_pred = map_real_space_pred(pred, grid, inv_basis,
                                              voxel_grid_real_space,
                                              method="nearest")
            if not majority:
                combined[n_view] = mapped_pred
            else:
                combined += mapped_pred

            if n_classes == 1:
                # Set to background if outside pred domain
                combined[n_view][np.isnan(combined[n_view])] = 0.

            if not predict_mode and np.random.rand() <= eval_prob:
                view_dices = dice_all(y, pred_to_class(pred, img_dims=3,
                                                       has_batch_dim=False),
                                      ignore_zero=False, n_classes=n_classes,
                                      skip_if_no_y = False)
                mapped_dices = dice_all(image.labels,
                                        pred_to_class(mapped_pred, img_dims=3,
                                                      has_batch_dim=False),
                                        ignore_zero=False, n_classes=n_classes,
                                        skip_if_no_y=False)

                # Print dice scores
                print("View dice scores:   ", view_dices)
                print("Mapped dice scores: ", mapped_dices)
                print("Mean dice: ", end="", flush=True)
                mean_dices = mapped_dices[~np.isnan(mapped_dices)][1:].mean()
                print("%s (n=%i)" % (mean_dices, len(mapped_dices)-1))

                # Add to results
                results[str(v)][n_image] = mean_dices
                detailed_res[str(v)][image_id] = mapped_dices[1:]

                # Overwrite with so-far results
                save_all(results, detailed_res, out_dir)
            else:
                print("Skipping evaluation for this view... "
                      "(eval_prob=%.3f, predict_mode=%s)" % (eval_prob,
                                                             predict_mode))

        if not analytical and not majority:
            # Combine predictions across views using Fusion model
            print("\nFusing views...")
            combined = np.moveaxis(combined, 0, -2).reshape((-1, len(views), n_classes))
            combined = fm.predict(combined, batch_size=10**4, verbose=1).reshape((d[0], d[1], d[2], n_classes))
        elif analytical:
            print("\nFusing views (analytical)...")
            combined = np.sum(combined, axis=0)

        if not no_argmax:
            print("\nComputing majority vote...")
            combined = pred_to_class(combined.squeeze(), img_dims=3).astype(np.uint8)

        if not predict_mode:
            if no_argmax:
                # MAP only for dice calculation
                c_temp = pred_to_class(combined, img_dims=3).astype(np.uint8)
            else:
                c_temp = combined

            # Calculate combined prediction dice
            dices = dice_all(image.labels, c_temp, n_classes=n_classes,
                             ignore_zero=True, skip_if_no_y=False)
            mean_dice = dices[~np.isnan(dices)].mean()
            detailed_res["MJ"][image_id] = dices

            print("Combined dices: ", dices)
            print("Combined mean dice: ", mean_dice)
            results["MJ"][n_image] = mean_dice

            # Overwrite with so-far results
            save_all(results, detailed_res, out_dir)

        # Save combined prediction volume as .nii file
        print("Saving .nii files...")
        save_nii_files(combined, image, nii_res_dir, save_input_files)

        # Remove image from dictionary and image_pair_loader to free memory
        del all_images[image_id]
        image_pair_loader.images.remove(image)

    if not predict_mode:
        # Write final results
        save_all(results, detailed_res, out_dir)


if __name__ == "__main__":
    entry_func()
