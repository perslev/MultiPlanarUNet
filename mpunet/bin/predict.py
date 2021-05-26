"""
Prediction/evaluation script

Mathias Perslev
March 2018
"""

import os
import numpy as np
import nibabel as nib

from mpunet.utils.utils import (create_folders, get_best_model,
                                pred_to_class, await_PIDs)
from mpunet.logging.log_results import save_all
from mpunet.evaluate.metrics import dice_all
from argparse import ArgumentParser


def get_argparser():
    parser = ArgumentParser(description='Predict using a mpunet model.')
    parser.add_argument("--project_dir", type=str, default="./",
                        help='Path to mpunet project folder')
    parser.add_argument("-f", help="Predict on a single file")
    parser.add_argument("-l", help="Optional single label file to use with -f")
    parser.add_argument("--dataset", type=str, default="test",
                        help="Which dataset of those stored in the hparams "
                             "file the evaluation should be performed on. "
                             "Has no effect if a single file is specified "
                             "with -f.")
    parser.add_argument("--out_dir", type=str, default="predictions",
                        help="Output folder to store results")
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help="Number of GPUs to use for this job")
    parser.add_argument("--sum_fusion", action="store_true",
                        help="Fuse the mutliple segmentation volumes into one"
                             " by summing over the probability axis instead "
                             "of applying a learned fusion model.")
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
    parser.add_argument("--continue", action="store_true",
                        help="Continue from a previsous, non-finished "
                             "prediction session at 'out_dir'.")
    return parser


def validate_folders(base_dir, out_dir, overwrite, _continue):
    """
    TODO
    """
    # Check base (model) dir contains required files
    must_exist = ("train_hparams.yaml", "views.npz",
                  "model")
    for p in must_exist:
        p = os.path.join(base_dir, p)
        if not os.path.exists(p):
            from sys import exit
            print("[*] Invalid mpunet project folder: '%s'"
                  "\n    Needed file/folder '%s' not found." % (base_dir, p))
            exit(0)

    # Check if output folder already exists
    if not (overwrite or _continue) and os.path.exists(out_dir):
        from sys import exit
        print("[*] Output directory already exists at: '%s'"
              "\n  Use --overwrite to overwrite or --continue to continue" % out_dir)
        exit(0)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


def save_nii_files(merged, image_pair, nii_res_dir, save_input_files):
    """
    TODO
    """
    # Extract data if nii files
    try:
        merged = merged.get_data()
    except AttributeError:
        merged = nib.Nifti1Image(merged, affine=image_pair.affine)
    volumes = [merged, image_pair.image_obj, image_pair.labels_obj]
    labels = ["%s_PRED.nii.gz" % image_pair.identifier,
              "%s_IMAGE.nii.gz" % image_pair.identifier,
              "%s_LABELS.nii.gz" % image_pair.identifier]
    if not save_input_files:
        volumes = volumes[:1]
        labels = labels[:1]
        p = os.path.abspath(nii_res_dir)  # Save file directly in nii_res_dir
    else:
        # Create sub-folder under nii_res_dir
        p = os.path.join(nii_res_dir, image_pair.identifier)
    create_folders(p)

    for nii, fname in zip(volumes, labels):
        try:
            nib.save(nii, "%s/%s" % (p, fname))
        except AttributeError:
            # No labels file?
            pass


def remove_already_predicted(all_images, out_dir):
    """
    TODO
    """
    nii_dir = os.path.join(out_dir, "nii_files")
    already_pred = [i.replace("_PRED", "").split(".")[0]
                    for i in filter(None, os.listdir(nii_dir))]
    print("[OBS] Not predicting on images: {} "
          "(--continue mode)".format(already_pred))
    return {k: v for k, v in all_images.items() if k not in already_pred}


def load_hparams(base_dir):
    """
    TODO
    """
    from mpunet.hyperparameters import YAMLHParams
    return YAMLHParams(os.path.join(base_dir, "train_hparams.yaml"))


def set_test_set(hparams, dataset):
    """
    TODO
    """
    hparams['test_dataset'] = hparams[dataset.strip("_dataset") + "_dataset"]


def set_gpu_vis(args):
    """
    TODO
    """
    force_gpu = args.force_GPU
    if not force_gpu:
        # Wait for free GPU
        from mpunet.utils import await_and_set_free_gpu
        await_and_set_free_gpu(N=args.num_GPUs, sleep_seconds=120)
        num_GPUs = args.num_GPUs
    else:
        from mpunet.utils import set_gpu
        set_gpu(force_gpu)
        num_GPUs = len(force_gpu.split(","))
    return num_GPUs


def get_image_pair_loader(args, hparams, out_dir):
    """
    TODO
    """
    from mpunet.image import ImagePairLoader, ImagePair
    if not args.f:
        # No single file was specified with -f flag, load the desired dataset
        dataset = args.dataset.replace("_data", "") + "_data"
        image_pair_loader = ImagePairLoader(predict_mode=args.no_eval,
                                            **hparams[dataset])
    else:
        predict_mode = not bool(args.l)
        image_pair_loader = ImagePairLoader(predict_mode=predict_mode,
                                            initialize_empty=True)
        image_pair_loader.add_image(ImagePair(args.f, args.l))

    # Put image pairs into a dict and remove from image_pair_loader to gain
    # more control with garbage collection
    image_pair_dict = {image.identifier: image for image in image_pair_loader.images}
    if vars(args)["continue"]:
        # Remove images that were already predicted
        image_pair_dict = remove_already_predicted(image_pair_dict, out_dir)
    return image_pair_loader, image_pair_dict


def get_results_dicts(out_dir, views, image_pairs_dict, n_classes, _continue):
    """
    TODO
    """
    from mpunet.logging import init_result_dicts, save_all, load_result_dicts
    if _continue:
        csv_dir = os.path.join(out_dir, "csv")
        results, detailed_res = load_result_dicts(csv_dir=csv_dir, views=views)
    else:
        # Prepare dictionary to store results in pd df
        results, detailed_res = init_result_dicts(views, image_pairs_dict, n_classes)
    # Save to check correct format
    save_all(results, detailed_res, out_dir)
    return results, detailed_res


def get_model(project_dir, build_hparams):
    """
    TODO
    """
    from mpunet.models.model_init import init_model
    model_path = get_best_model(project_dir + "/model")
    weights_name = os.path.splitext(os.path.split(model_path)[1])[0]
    print("\n[*] Loading model weights:\n", model_path)
    import tensorflow as tf
    with tf.distribute.MirroredStrategy().scope():
        model = init_model(build_hparams)
        model.load_weights(model_path, by_name=True)
    return model, weights_name


def get_fusion_model(n_views, n_classes, project_dir, weights_name):
    """
    TODO
    """
    from mpunet.models import FusionModel
    fm = FusionModel(n_inputs=n_views, n_classes=n_classes)
    # Load fusion weights
    weights = project_dir + "/model/fusion_weights/%s_fusion_" \
                            "weights.h5" % weights_name
    print("\n[*] Loading fusion model weights:\n", weights)
    fm.load_weights(weights)
    print("\nLoaded weights:\n\n%s\n%s\n---" % tuple(
        fm.layers[-1].get_weights()))
    return fm


def evaluate(pred, true, n_classes, ignore_zero=False):
    """
    TODO
    """
    pred = pred_to_class(pred, img_dims=3, has_batch_dim=False)
    return dice_all(y_true=true,
                    y_pred=pred,
                    ignore_zero=ignore_zero,
                    n_classes=n_classes,
                    skip_if_no_y=False)


def _per_view_evaluation(image_id, pred, true, mapped_pred, mapped_true, view,
                         n_classes, results, per_view_results, out_dir, args):
    """
    TODO
    """
    if np.random.rand() > args.eval_prob:
        print("Skipping evaluation for view %s... "
              "(eval_prob=%.3f)" % (view, args.eval_prob))
        return

    # Evaluate the raw view performance
    view_dices = evaluate(pred, true, n_classes)
    mapped_dices = evaluate(mapped_pred, mapped_true, n_classes)
    mean_dice = mapped_dices[~np.isnan(mapped_dices)][1:].mean()

    # Print dice scores
    print("View dice scores:   ", view_dices)
    print("Mapped dice scores: ", mapped_dices)
    print("Mean dice (n=%i): " % (len(mapped_dices) - 1), mean_dice)

    # Add to results
    results.loc[image_id, str(view)] = mean_dice
    per_view_results[str(view)][image_id] = mapped_dices[1:]

    # Overwrite with so-far results
    save_all(results, per_view_results, out_dir)


def _merged_eval(image_id, pred, true, n_classes, results,
                 per_view_results, out_dir):
    """
    TODO
    """
    # Calculate combined prediction dice
    dices = evaluate(pred, true, n_classes, ignore_zero=True)
    mean_dice = dices[~np.isnan(dices)].mean()
    per_view_results["MJ"][image_id] = dices

    print("Combined dices: ", dices)
    print("Combined mean dice: ", mean_dice)
    results.loc[image_id, "MJ"] = mean_dice

    # Overwrite with so-far results
    save_all(results, per_view_results, out_dir)


def _multi_view_predict_on(image_pair, seq, model, views, results,
                           per_view_results, out_dir, args):
    """
    TODO
    """
    from mpunet.utils.fusion import predict_volume, map_real_space_pred
    from mpunet.interpolation.sample_grid import get_voxel_grid_real_space

    # Get voxel grid in real space
    voxel_grid_real_space = get_voxel_grid_real_space(image_pair)

    # Prepare tensor to store combined prediction
    d = image_pair.image.shape[:-1]
    combined = np.empty(
        shape=(len(views), d[0], d[1], d[2], seq.n_classes),
        dtype=np.float32
    )
    print("Predicting on brain hyper-volume of shape:", combined.shape)

    # Predict for each view
    for n_view, view in enumerate(views):
        print("\n[*] (%i/%i) View: %s" % (n_view + 1, len(views), view))
        # for each view, predict on all voxels and map the predictions
        # back into the original coordinate system

        # Sample planes from the image at grid_real_space grid
        # in real space (scanner RAS) coordinates.
        X, y, grid, inv_basis = seq.get_view_from(image_pair, view,
                                                  n_planes="same+20")

        # Predict on volume using model
        pred = predict_volume(model, X, axis=2, batch_size=seq.batch_size)

        # Map the real space coordiante predictions to nearest
        # real space coordinates defined on voxel grid
        mapped_pred = map_real_space_pred(pred, grid, inv_basis,
                                          voxel_grid_real_space,
                                          method="nearest")
        combined[n_view] = mapped_pred

        if not args.no_eval:
            _per_view_evaluation(image_id=image_pair.identifier,
                                 pred=pred,
                                 true=y,
                                 mapped_pred=mapped_pred,
                                 mapped_true=image_pair.labels,
                                 view=view,
                                 n_classes=seq.n_classes,
                                 results=results,
                                 per_view_results=per_view_results,
                                 out_dir=out_dir,
                                 args=args)
    return combined


def merge_multi_view_preds(multi_view_preds, fusion_model, args):
    """
    TODO
    """
    fm = fusion_model
    if not args.sum_fusion:
        # Combine predictions across views using Fusion model
        print("\nFusing views (fusion model)...")
        d = multi_view_preds.shape
        multi_view_preds = np.moveaxis(multi_view_preds, 0, -2)
        multi_view_preds = multi_view_preds.reshape((-1, fm.n_inputs, fm.n_classes))
        merged = fm.predict(multi_view_preds, batch_size=10**4, verbose=1)
        merged = merged.reshape((d[1], d[2], d[3], fm.n_classes))
    else:
        print("\nFusion views (sum)...")
        merged = np.sum(multi_view_preds, axis=0)
    merged_map = pred_to_class(merged.squeeze(), img_dims=3).astype(np.uint8)
    return merged, merged_map


def run_predictions_and_eval(image_pair_loader, image_pair_dict, model,
                             fusion_model, views, hparams, args, results,
                             per_view_results, out_dir, nii_res_dir):
    """
    TODO
    """
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
                       views=views,
                       **hparams["fit"], **hparams["build"])

    image_ids = sorted(image_pair_dict)
    n_images = len(image_ids)
    for n_image, image_id in enumerate(image_ids):
        print("\n[*] (%i/%s) Running on: %s" % (n_image + 1, n_images, image_id))

        with seq.image_pair_queue.get_image_by_id(image_id) as image_pair:
            # Get prediction through all views
            multi_view_preds = _multi_view_predict_on(
                image_pair=image_pair,
                seq=seq,
                model=model,
                views=views,
                results=results,
                per_view_results=per_view_results,
                out_dir=out_dir,
                args=args
            )

            # Merge the multi view predictions into a final segmentation
            merged, merged_map = merge_multi_view_preds(multi_view_preds,
                                                        fusion_model, args)
            if not args.no_eval:
                _merged_eval(
                    image_id=image_id,
                    pred=merged_map,
                    true=image_pair.labels,
                    n_classes=hparams["build"]["n_classes"],
                    results=results,
                    per_view_results=per_view_results,
                    out_dir=out_dir
                )

            # Save combined prediction volume as .nii file
            print("Saving .nii files...")
            save_nii_files(merged=merged_map if not args.no_argmax else merged,
                           image_pair=image_pair,
                           nii_res_dir=nii_res_dir,
                           save_input_files=args.save_input_files)


def assert_args(args):
    pass


def entry_func(args=None):
    # Get command line arguments
    args = get_argparser().parse_args(args)
    assert_args(args)

    # Get most important paths
    project_dir = os.path.abspath(args.project_dir)
    out_dir = os.path.abspath(args.out_dir)

    # Check if valid dir structures
    validate_folders(project_dir, out_dir,
                     overwrite=args.overwrite,
                     _continue=vars(args)["continue"])
    nii_res_dir = os.path.join(out_dir, "nii_files")
    create_folders(nii_res_dir, create_deep=True)

    # Get settings from YAML file
    hparams = load_hparams(project_dir)

    # Get dataset
    image_pair_loader, image_pair_dict = get_image_pair_loader(args, hparams,
                                                               out_dir)

    # Wait for PID to terminate before continuing, if specified
    if args.wait_for:
        await_PIDs(args.wait_for, check_every=120)

    # Set GPU device
    set_gpu_vis(args)

    # Get views
    views = np.load("%s/views.npz" % project_dir)["arr_0"]
    del hparams['fit']['views']

    # Prepare result dicts
    results, per_view_results = None, None
    if not args.no_eval:
        results, per_view_results = get_results_dicts(out_dir, views,
                                                      image_pair_dict,
                                                      hparams["build"]["n_classes"],
                                                      vars(args)["continue"])

    # Get model and load weights, assign to one or more GPUs
    model, weights_name = get_model(project_dir, hparams['build'])
    fusion_model = None
    if not args.sum_fusion:
        fusion_model = get_fusion_model(n_views=len(views),
                                        n_classes=hparams["build"]["n_classes"],
                                        project_dir=project_dir,
                                        weights_name=weights_name)

    run_predictions_and_eval(
        image_pair_loader=image_pair_loader,
        image_pair_dict=image_pair_dict,
        model=model,
        fusion_model=fusion_model,
        views=views,
        hparams=hparams,
        args=args,
        results=results,
        per_view_results=per_view_results,
        out_dir=out_dir,
        nii_res_dir=nii_res_dir
    )
    if not args.no_eval:
        # Write final results
        save_all(results, per_view_results, out_dir)


if __name__ == "__main__":
    entry_func()
