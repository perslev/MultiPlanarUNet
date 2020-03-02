import numpy as np

from mpunet.utils.fusion import map_real_space_pred
from mpunet.evaluate import dice_all


def stack_collections(points_collection, targets_collection):
    """

    Args:
        points_collection:
        targets_collection:

    Returns:

    """
    if len(points_collection) == 1 and len(targets_collection) == 1:
        return points_collection[0], targets_collection[0]
    n_points = sum([x.shape[0] for x in points_collection])
    n_views, n_classes = points_collection[0].shape[1:]

    X = np.empty(shape=(n_points, n_views, n_classes),
                 dtype=points_collection[0].dtype)
    y = np.empty(shape=(n_points, 1),
                 dtype=targets_collection[0].dtype)

    c = 0
    len_collection = len(points_collection)
    for i in range(len_collection):
        print("  %i/%i" % (i + 1, len_collection),
              end="\r", flush=True)
        Xs = points_collection.pop()
        X[c:c + len(Xs)] = Xs
        y[c:c + len(Xs)] = targets_collection.pop()
        c += len(Xs)
    print("")
    return X, y


def predict_and_map(model, seq, image, view, batch_size=None,
                    voxel_grid_real_space=None, targets=None, eval_prob=1.0,
                    n_planes='same+20'):
    """


    Args:
        model:
        seq:
        image:
        view:
        batch_size:
        voxel_grid_real_space:
        targets:
        n_planes:

    Returns:

    """

    # Sample planes from the image at grid_real_space grid
    # in real space (scanner RAS) coordinates.
    X, y, grid, inv_basis = seq.get_view_from(image, view, n_planes=n_planes)

    # Predict on volume using model
    bs = seq.batch_size if batch_size is None else batch_size
    from mpunet.utils.fusion import predict_volume
    pred = predict_volume(model, X, axis=2, batch_size=bs)

    # Map the real space coordiante predictions to nearest
    # real space coordinates defined on voxel grid
    if voxel_grid_real_space is None:
        from mpunet.interpolation.sample_grid import get_voxel_grid_real_space
        voxel_grid_real_space = get_voxel_grid_real_space(image)

    # Map the predicted volume to real space
    mapped = map_real_space_pred(pred, grid, inv_basis, voxel_grid_real_space)

    # Print dice scores
    if targets is not None and np.random.rand(1)[0] <= eval_prob:
        print("Computing evaluations...")
        print("View dice scores:   ", dice_all(y, pred.argmax(-1),
                                               ignore_zero=False))
        print("Mapped dice scores: ", dice_all(targets,
                                               mapped.argmax(-1).reshape(-1, 1),
                                               ignore_zero=False))
    else:
        print("-- Skipping evaluation")

    return mapped
