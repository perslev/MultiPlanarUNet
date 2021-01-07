"""
Prediction/evaluation script

Mathias Perslev
March 2018
"""

import os
from argparse import ArgumentParser


def get_argparser():
    parser = ArgumentParser(description='Predict using a mpunet model.')
    parser.add_argument("--project_dir", type=str, default="./",
                        help='Path to mpunet project folder')
    parser.add_argument("-f", help="Predict on a single file")
    parser.add_argument("-l", help="Optional single label file to use with -f")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory storing data. "
                             "Must contain sub-folder 'images'")
    parser.add_argument("--out_dir", type=str, default="predictions",
                        help="Output folder to store results")
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help="Number of GPUs to use for this job")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite previous results at the output folder')
    parser.add_argument("--no_eval", action="store_true",
                        help="Perform no evaluation of the prediction performance. "
                             "No label files loaded when this flag applies.")
    parser.add_argument("--strides", type=int, default=None,
                        help="Predict on strided overlapping boxes "
                             "instead of only non-overlapping.")
    parser.add_argument("--extra", default="2x",
                        help="Sample N extra patches to perform majority voting")
    parser.add_argument("--force_GPU", type=int, default=-1)
    parser.add_argument("--save_only_pred", action="store_true",
                        help="Save only the predicted volume as .nii files ("
                             "do not save image and labels)")
    return parser


def validate_folders(base_dir, data_dir, out_dir, overwrite):

    # Check base (model) dir contains required files
    must_exist = ("train_hparams.yaml", "model")
    for p in must_exist:
        p = os.path.join(base_dir, p)
        if not os.path.exists(p):
            from sys import exit
            print("[*] Invalid mpunet project folder: '%s'"
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


def entry_func(args=None):

    # Get command line arguments
    args = vars(get_argparser().parse_args(args))
    base_dir = os.path.abspath(args["project_dir"])
    _file = args["f"]
    label = args["l"]
    N_extra = args["extra"]
    try:
        N_extra = int(N_extra)
    except ValueError:
        pass

    # Get settings from YAML file
    from mpunet.hyperparameters import YAMLHParams
    hparams = YAMLHParams(os.path.join(base_dir, "train_hparams.yaml"))

    # Set strides
    hparams["fit"]["strides"] = args["strides"]

    if not _file:
        try:
            # Data specified from command line?
            data_dir = os.path.abspath(args["data_dir"])

            # Set with default sub dirs
            hparams["test_data"] = {"base_dir": 
                                        data_dir, "img_subdir": "images",
                                    "label_subdir": "labels"}
        except (AttributeError, TypeError):
            data_dir = hparams["test_data"]["base_dir"]
    else:
        data_dir = False
    out_dir = os.path.abspath(args["out_dir"])
    overwrite = args["overwrite"]
    predict_mode = args["no_eval"]
    save_only_pred = args["save_only_pred"]

    # Check if valid dir structures
    validate_folders(base_dir, data_dir, out_dir, overwrite)

    # Import all needed modules (folder is valid at this point)
    import numpy as np
    from mpunet.image import ImagePairLoader, ImagePair
    from mpunet.utils import get_best_model, create_folders, \
                                    pred_to_class, await_and_set_free_gpu, set_gpu
    from mpunet.utils.fusion import predict_3D_patches, predict_3D_patches_binary, pred_3D_iso
    from mpunet.logging import init_result_dict_3D, save_all_3D
    from mpunet.evaluate import dice_all
    from mpunet.bin.predict import save_nii_files

    # Fetch GPU(s)
    num_GPUs = args["num_GPUs"]
    force_gpu = args["force_GPU"]
    # Wait for free GPU
    if force_gpu == -1:
        await_and_set_free_gpu(N=num_GPUs, sleep_seconds=240)
    else:
        set_gpu(force_gpu)

    # Read settings from the project hyperparameter file
    dim = hparams["build"]["dim"]
    n_classes = hparams["build"]["n_classes"]
    mode = hparams["fit"]["intrp_style"]

    # Set ImagePairLoader object
    if not _file:
        image_pair_loader = ImagePairLoader(predict_mode=predict_mode, **hparams["test_data"])
    else:
        predict_mode = not bool(label)
        image_pair_loader = ImagePairLoader(predict_mode=predict_mode,
                                            initialize_empty=True)
        image_pair_loader.add_image(ImagePair(_file, label))
    all_images = {image.identifier: image for image in image_pair_loader.images}

    # Set scaler and bg values
    image_pair_loader.set_scaler_and_bg_values(
        bg_value=hparams.get_from_anywhere('bg_value'),
        scaler=hparams.get_from_anywhere('scaler'),
        compute_now=False
    )

    # Init LazyQueue and get its sequencer
    from mpunet.sequences.utils import get_sequence
    seq = get_sequence(data_queue=image_pair_loader,
                       is_validation=True,
                       **hparams["fit"], **hparams["build"])

    """ Define UNet model """
    from mpunet.models import model_initializer
    hparams["build"]["batch_size"] = 1
    unet = model_initializer(hparams, False, base_dir)
    model_path = get_best_model(base_dir + "/model")
    unet.load_weights(model_path)

    # Evaluate?
    if not predict_mode:
        # Prepare dictionary to store results in pd df
        results, detailed_res = init_result_dict_3D(all_images, n_classes)

        # Save to check correct format
        save_all_3D(results, detailed_res, out_dir)

    # Define result paths
    nii_res_dir = os.path.join(out_dir, "nii_files")
    create_folders(nii_res_dir)

    image_ids = sorted(all_images)
    for n_image, image_id in enumerate(image_ids):
        print("\n[*] Running on: %s" % image_id)

        with seq.image_pair_queue.get_image_by_id(image_id) as image_pair:
            if mode.lower() == "iso_live_3d":
                pred = pred_3D_iso(model=unet,
                                   sequence=seq,
                                   image=image_pair,
                                   extra_boxes=N_extra,
                                   min_coverage=None)
            else:
                # Predict on volume using model
                if n_classes > 1:
                    pred = predict_3D_patches(model=unet,
                                              patches=seq,
                                              image=image_pair,
                                              N_extra=N_extra)
                else:
                    pred = predict_3D_patches_binary(model=unet,
                                                     patches=seq,
                                                     image_id=image_id,
                                                     N_extra=N_extra)

            if not predict_mode:
                # Get patches for the current image
                y = image_pair.labels

                # Calculate dice score
                print("Mean dice: ", end="", flush=True)
                p = pred_to_class(pred, img_dims=3, has_batch_dim=False)
                dices = dice_all(y, p, n_classes=n_classes, ignore_zero=True)
                mean_dice = dices[~np.isnan(dices)].mean()
                print("Dices: ", dices)
                print("%s (n=%i)" % (mean_dice, len(dices)))

                # Add to results
                results[image_id] = [mean_dice]
                detailed_res[image_id] = dices

                # Overwrite with so-far results
                save_all_3D(results, detailed_res, out_dir)

                # Save results
                save_nii_files(p, image_pair, nii_res_dir, save_only_pred)

    if not predict_mode:
        # Write final results
        save_all_3D(results, detailed_res, out_dir)


if __name__ == "__main__":
    entry_func()
