import numpy as np
from mpunet.interpolation.regular_grid_interpolator import RegularGridInterpolator

from mpunet.preprocessing import reshape_add_axis
from mpunet.interpolation.linalg import mgrid_to_points, points_to_mgrid
from mpunet.interpolation.sample_grid import get_voxel_grid, get_voxel_axes_real_space, get_voxel_grid_real_space
from mpunet.sequences.utils import get_sequence


def predict_single(image, model, hparams, verbose=1):
    """
    A generic prediction function that sets up a ImagePairLoader object for the
    given image, prepares the image and predicts.

    Note that this function should only be used for convinience in scripts that
    work on single images at a time anyway, as batch-preparing the entire
    ImagePairLoader object prior to prediction is faster.

    NOTE: Only works with iso_live intrp modes at this time
    """
    mode = hparams["fit"]["intrp_style"].lower()
    assert mode in ("iso_live", "iso_live_3d")

    # Create a ImagePairLoader with only the given file
    from mpunet.image import ImagePairLoader
    image_pair_loader = ImagePairLoader(predict_mode=True,
                                        initialize_empty=True,
                                        no_log=bool(verbose))
    image_pair_loader.add_image(image)

    # Set scaler and bg values
    image_pair_loader.set_scaler_and_bg_values(
        bg_value=hparams.get_from_anywhere('bg_value'),
        scaler=hparams.get_from_anywhere('scaler'),
        compute_now=False
    )

    if mode == "iso_live":
        # Init LazyQueue and get its sequencer
        seq = get_sequence(data_queue=image_pair_loader,
                           views=np.load(hparams.project_path + "/views.npz")["arr_0"],
                           is_validation=True,
                           **hparams["fit"], **hparams["build"])

        with seq.image_pair_queue.get_image_by_id(image.identifier) as image:
            # Get voxel grid in real space
            voxel_grid_real_space = get_voxel_grid_real_space(image)

            # Prepare tensor to store combined prediction
            d = image.image.shape
            predicted = np.empty(shape=(len(seq.views), d[0], d[1], d[2], seq.n_classes),
                                 dtype=np.float32)
            print("Predicting on brain hyper-volume of shape:", predicted.shape)

            for n_view, v in enumerate(seq.views):
                print("\nView %i/%i: %s" % (n_view+1, len(seq.views), v))
                # Sample the volume along the view
                X, y, grid, inv_basis = seq.get_view_from(image, v,
                                                          n_planes="same+20")

                # Predict on volume using model
                pred = predict_volume(model, X, axis=2)

                # Map the real space coordiante predictions to nearest
                # real space coordinates defined on voxel grid
                predicted[n_view] = map_real_space_pred(pred, grid, inv_basis,
                                                        voxel_grid_real_space,
                                                        method="nearest")
    else:
        # Init LazyQueue and get its sequencer
        seq = get_sequence(data_queue=image_pair_loader,
                           is_validation=True,
                           **hparams["fit"], **hparams["build"])
        predicted = pred_3D_iso(model=model, sequence=seq,
                                image=image, extra_boxes="3x",
                                min_coverage=None)

    return predicted


def predict_volume(model, X, batch_size=8, axis=0):
    """
    TODO
    """
    print("Predicting...")
    # Move axis to front
    X = np.moveaxis(X, source=axis, destination=0)
    pred = model.predict(X, batch_size=batch_size, verbose=1)
    return np.moveaxis(pred, source=0, destination=axis)


def map_real_space_pred(pred, grid, inv_basis, voxel_grid_real_space, method="nearest"):
    """
    TODO
    """
    print("Mapping to real coordinate space...")

    # Prepare fill value vector, we set this to 1.0 background
    fill = np.zeros(shape=pred.shape[-1], dtype=np.float32)
    fill[0] = 1.0

    # Initialize interpolator object
    intrp = RegularGridInterpolator(grid, pred, fill_value=fill,
                                    bounds_error=False, method=method)

    points = inv_basis.dot(mgrid_to_points(voxel_grid_real_space).T).T
    transformed_grid = points_to_mgrid(points, voxel_grid_real_space[0].shape)

    # Prepare mapped pred volume
    mapped = np.empty(transformed_grid[0].shape + (pred.shape[-1],),
                      dtype=pred.dtype)

    # Prepare interpolation function
    def _do(xs, ys, zs, index):
        return intrp((xs, ys, zs)), index

    # Prepare thread pool of 10 workers
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=7)

    # Perform interpolation async.
    inds = np.arange(transformed_grid.shape[1])
    result = pool.map(_do, transformed_grid[0], transformed_grid[1],
                      transformed_grid[2], inds)

    i = 1
    for map, ind in result:
        # Print status
        print("  %i/%i" % (i, inds[-1]+1), end="\r", flush=True)
        i += 1

        # Map the interpolation results into the volume
        mapped[ind] = map

    print("")
    pool.shutdown()
    return mapped


def predict_3D_patches_binary(model, patches, image_id, N_extra=0, logger=None):
    """
    TODO
    """
    # Get box dim and image dim
    d = patches.dim
    i1, i2, i3 = patches.im_dim

    # Prepare reconstruction volume. Predictions will be summed in this volume.
    recon = np.zeros(shape=(i1, i2, i3, 2), dtype=np.uint32)

    # Predict on base patches + N extra randomly
    # sampled patches from the volume
    for patch, (i, k, v), status in patches.get_patches_from(image_id, N_extra):
        # Log the status of the generator
        print(status, end="\r", flush=True)

        # Predict on patch
        pred = model.predict_on_batch(reshape_add_axis(patch, im_dims=3)).numpy().squeeze()
        mask = pred > 0.5

        # Add prediction to reconstructed volume
        recon[i:i+d, k:k+d, v:v+d, 0] += ~mask
        recon[i:i+d, k:k+d, v:v+d, 1] += mask
    print("")

    total = np.sum(recon, axis=-1)
    return (recon[..., 1] > (0.20 * total)).astype(np.uint8)


def predict_3D_patches(model, patches, image, N_extra=0, logger=None):
    """
    TODO
    """
    # Get box dim and image dim
    d = patches.dim
    i1, i2, i3 = image.shape[:3]

    # Prepare reconstruction volume. Predictions will be summed in this volume.
    recon = np.zeros(shape=(i1, i2, i3, model.n_classes), dtype=np.float32)

    # Predict on base patches + N extra randomly
    # sampled patches from the volume
    for patch, (i, k, v), status in patches.get_patches_from(image, N_extra):
        # Log the status of the generator
        print(status, end="\r", flush=True)

        # Predict on patch
        pred = model.predict_on_batch(reshape_add_axis(patch, im_dims=3)).numpy()

        # Add prediction to reconstructed volume
        recon[i:i+d, k:k+d, v:v+d] += pred.squeeze()
    print("")

    # Normalize
    recon /= np.sum(recon, axis=-1, keepdims=True)

    return recon


def pred_3D_iso(model, sequence, image, extra_boxes, min_coverage=None):
    """
    TODO
    """
    total_extra_boxes = extra_boxes

    # Get reference to the image
    n_classes = sequence.n_classes
    pred_shape = tuple(image.shape[:3]) + (n_classes,)
    vox_shape = tuple(image.shape[:3]) + (3,)

    # Prepare interpolator object
    vox_grid = get_voxel_grid(image, as_points=False)

    # Get voxel regular grid centered in real space
    g_all, basis, _ = get_voxel_axes_real_space(image.image, image.affine,
                                                return_basis=True)
    g_all = list(g_all)

    # Flip axes? Must be strictly increasing
    flip = np.sign(np.diagonal(basis)) == -1
    for i, (g, f) in enumerate(zip(g_all, flip)):
        if f:
            g_all[i] = np.flip(g, 0)
            vox_grid = np.flip(vox_grid, i+1)
    vox_points = mgrid_to_points(vox_grid).reshape(vox_shape).astype(np.float32)

    # Setup interpolator - takes a point in the scanner space and returns
    # the nearest voxel coordinate
    intrp = RegularGridInterpolator(tuple(g_all), vox_points,
                                    method="nearest", bounds_error=False,
                                    fill_value=np.nan, dtype=np.float32)

    # Prepare prediction volume
    pred_vol = np.zeros(shape=pred_shape, dtype=np.float32)

    # Predict on base patches first
    base_patches = sequence.get_base_patches_from(image, return_y=False)

    # Sample boxes and predict --> sum into pred_vol
    is_covered, base_reached, extra_reached, N_base, N_extra = not min_coverage, False, False, 0, 0

    while not is_covered or not base_reached or not extra_reached:
        try:
            im, rgrid, _, _, total_base = next(base_patches)
            N_base += 1

            if isinstance(total_extra_boxes, str):
                # Number specified in string format '2x', '2.5x' etc. as a
                # multiplier of number of base patches
                total_extra_boxes = int(float(total_extra_boxes.split("x")[0]) * total_base)

        except StopIteration:
            p = sequence.get_N_random_patches_from(image, 1, return_y=False)
            im, rgrid, _, _ = next(p)
            N_extra += 1

        # Predict on the box
        pred = model.predict_on_batch(np.expand_dims(im, 0))[0].numpy()

        # Apply rotation if needed
        rgrid = image.interpolator.apply_rotation(rgrid)

        # Interpolate to nearest vox grid positions
        vox_inds = intrp(tuple(rgrid)).reshape(-1, 3)

        # Flatten and mask results
        mask = np.logical_not(np.all(np.isnan(vox_inds), axis=-1))
        vox_inds = [i for i in vox_inds[mask].astype(np.int).T]

        # Add to volume
        pred_vol[tuple(vox_inds)] += pred.reshape(-1, n_classes)[mask]

        # Check coverage fraction
        if min_coverage:
            covered = np.logical_not(np.all(np.isclose(pred_vol, 0), axis=-1))
            coverage = np.sum(covered) / np.prod(pred_vol.shape[:3])
            cov_string = "%.3f/%.3f" % coverage, min_coverage
            is_covered = coverage >= min_coverage
        else:
            cov_string = "[Not calculated]"

        print("   N base patches: %i/%i --- N extra patches %i/%i --- "
              "Coverage: %s" % (
                N_base, total_base, N_extra, total_extra_boxes, cov_string),
              end="\r", flush=True)

        # Check convergence
        base_reached = N_base >= total_base
        extra_reached = N_extra >= total_extra_boxes
    print("")

    # Return prediction volume - OBS not normalized
    return pred_vol
