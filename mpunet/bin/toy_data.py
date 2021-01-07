import os
import numpy as np
import nibabel as nib

from argparse import ArgumentParser
from scipy.ndimage.filters import gaussian_filter
from mpunet.utils.utils import create_folders


def get_argparser():
    parser = ArgumentParser(description='Create a toy dataset - Useful '
                                        'for testing purposes')
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Path to a dir in which the toy data will be "
                             "stored. Must be a non-existing or empty dir.")
    parser.add_argument("--N", type=int, default=10, required=False,
                        help='The number of train, val and test images to '
                             'create.')
    parser.add_argument("--image_size", type=int, default=128,
                        help="Voxel size of image.")
    parser.add_argument("--N_train", type=int, default=0, required=False,
                        help='The number of train images to create. '
                             'Overrides the --N flag.')
    parser.add_argument("--N_val", type=int, default=0, required=False,
                        help='The number of val images to create. '
                             'Overrides the --N flag.')
    parser.add_argument("--N_test", type=int, default=0, required=False,
                        help='The number of test images to create. '
                             'Overrides the --N flag.')
    parser.add_argument("--image_subdir", type=str, default="images",
                        help="Optional name of subdir to store image files")
    parser.add_argument("--label_subdir", type=str, default="labels",
                        help="Optional name of subdir to store label files")
    parser.add_argument("--seed", type=int, default=0,
                        help="Use a specific seed for random number "
                             "generation. Useful for debugging purposes.")
    return parser


def _elastic_deform(xx, yy, zz, sigma, alpha):
    dx = gaussian_filter((np.random.rand(*xx.shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha
    dy = gaussian_filter((np.random.rand(*yy.shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha
    dz = gaussian_filter((np.random.rand(*zz.shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha
    return xx+dx, yy+dy, zz+dz


def _get_center(img_size):
    min_, max_ = img_size * 0.05, img_size * 0.95
    range_ = max_ - min_
    center = np.random.rand(3) * range_ - min_
    return center.astype(np.int)


def create_toy_image(img_size):
    # Create empty image
    im = np.zeros(shape=[img_size]*3, dtype=np.float64)

    # Create grid
    xx, yy, zz = np.mgrid[:img_size, :img_size, :img_size]
    xx_d, yy_d, zz_d = _elastic_deform(xx, yy, zz,
                                       sigma=img_size/6,
                                       alpha=img_size*60)

    # Add background
    im += np.sin(0.05 + np.random.rand() * xx/img_size) + \
          np.power(np.cos(0.05 + np.random.rand() * yy/img_size), 0.5) + \
          np.power(np.sin(0.05 + np.random.rand() * zz/img_size), 2)

    # Scale 0-0.75
    im -= im.min()
    im /= im.max() * (1/0.75)

    # Add sphere
    radius = img_size/8 + np.random.rand()*img_size/2
    center = _get_center(img_size)
    sphere_mask = np.power(xx_d-center[0], 2) + \
                  np.power(yy_d-center[1], 2) + \
                  np.power(zz_d-center[2], 2) <= np.power(radius, 2)
    sphere_vals = np.sin((xx[sphere_mask]-center[0])/img_size) + \
                  np.sin((yy[sphere_mask]-center[1])/img_size*5) + \
                  np.sin((zz[sphere_mask]-center[2])/img_size*10)

    # Add and scale 0.2-1.0
    im[sphere_mask] += sphere_vals
    im[sphere_mask] -= im[sphere_mask].min()
    im[sphere_mask] /= im[sphere_mask].max() * 1/0.8
    im[sphere_mask] += 0.2

    # Add elastic square
    size = img_size/4 + np.random.rand()*img_size/2
    center = _get_center(img_size)
    square_mask = (np.abs(xx_d-center[0]) < size/2) & \
                  (np.abs(yy_d-center[1]) < size/2) & \
                  (np.abs(zz_d-center[2]) < size/2)
    square_vals = np.power(xx[square_mask]-center[0], 2) + \
                  np.power(yy[square_mask]-center[1], 2) + \
                  np.power(zz[square_mask]-center[2], 2)

    # Add and scale 0.2-1.0
    im[square_mask] += square_vals
    im[square_mask] += square_vals
    im[square_mask] -= im[square_mask].min()
    im[square_mask] /= im[square_mask].max() * 1/0.8
    im[square_mask] += 0.2

    # Add torus
    center_radius = img_size/16 + np.random.rand()*img_size/6
    tube_radius = img_size/32 + np.random.rand()*img_size/12
    center = _get_center(img_size)
    torus_mask_1 = np.power(center_radius-np.sqrt(np.power(xx_d-center[0], 2) +
                                                  np.power(yy_d-center[1], 2)), 2) + \
                   np.power(zz_d-center[2], 2) <= np.power(tube_radius, 2)
    torus_mask_2 = np.power(center_radius-np.sqrt(np.power(zz_d-center[2], 2) +
                                                  np.power(yy_d-center[1], 2)), 2) + \
                   np.power(xx_d-center[0], 2) <= np.power(tube_radius, 2)
    torus_mask = np.logical_or(torus_mask_1, torus_mask_2)
    torus_vals = np.random.poisson(im[torus_mask])

    # Add and scale 0.2-1.0
    im[torus_mask] += torus_vals
    im[torus_mask] -= im[torus_mask].min()
    im[torus_mask] /= im[torus_mask].max() * 1 / 0.8
    im[torus_mask] += 0.2

    # Add noise to image
    im += np.random.poisson(lam=10, size=im.shape) * np.random.randn(*im.shape) * 0.0005

    # Scale overall image to [0, ..., 1]
    im -= im.min()
    im /= im.max()

    # Scale to 16 bit int
    im *= np.iinfo(np.uint16).max
    return im.astype(np.uint16), sphere_mask, square_mask, torus_mask


def create_toy_labels(*masks):
    assert len(masks) < np.iinfo(np.uint8).max
    labels = np.zeros(shape=masks[0].shape, dtype=np.uint8)
    for i, mask in enumerate(masks):
        labels[mask] = i+1
    return labels


def create_toy_data_point(img_size):
    im, m1, m2, m3 = create_toy_image(img_size)
    lab = create_toy_labels(m1, m2, m3)
    return im, lab


def create_toy_dataset(N, image_size, out_dir, image_subdir, label_subdir):
    image_out_dir = os.path.join(out_dir, image_subdir)
    label_out_dir = os.path.join(out_dir, label_subdir)
    create_folders([out_dir, image_out_dir, label_out_dir])

    for img_id in range(N):
        print("--", img_id)
        file_name = "toy_data_{}.nii.gz".format(img_id)
        im_path = os.path.join(image_out_dir, file_name)
        lab_path = os.path.join(label_out_dir, file_name)
        image, label = create_toy_data_point(image_size)

        affine = np.eye(4)
        nib.save(nib.Nifti1Image(image, affine=affine), im_path)
        nib.save(nib.Nifti1Image(label, affine=affine), lab_path)


def entry_func(args=None):

    args = get_argparser().parse_args(args)
    out_dir = os.path.abspath(os.path.abspath(args.out_dir))
    if os.path.exists(out_dir) and os.listdir(out_dir):
        from sys import exit
        print("Path '{}' already exists and is not empty!".format(args.out_dir))
        exit(1)
    create_folders([out_dir], create_deep=False)

    n_train, n_val, n_test = [args.N] * 3
    n_train = args.N_train or n_train
    n_val = args.N_val or n_val
    n_test = args.N_test or n_test

    np.random.seed(args.seed)
    for dataset, N in zip(["train", "val", "test"], [n_train, n_val, n_test]):
        if not N:
            continue
        else:
            print("[*] Creating dataset '{}' of {} samples".format(
                dataset, N
            ))
            create_toy_dataset(N=N,
                               image_size=args.image_size,
                               out_dir=os.path.join(out_dir, dataset),
                               image_subdir=args.image_subdir,
                               label_subdir=args.label_subdir)


if __name__ == "__main__":
    entry_func()
