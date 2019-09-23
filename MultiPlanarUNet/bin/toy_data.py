import os
from argparse import ArgumentParser
import readline
import numpy as np
import pandas as pd
import nibabel as nib

from MultiPlanarUNet.utils.utils import create_folders

readline.parse_and_bind('tab: complete')


def get_argparser():
    parser = ArgumentParser(description='Create a toy dataset - Useful '
                                        'for testing purposes')
    parser.add_argument("--out_dir", type=str,
                        help="Path to a dir in which the toy data will be "
                             "stored. Must be a non-existing or empty dir.")
    parser.add_argument("--N", type=int, default=10, required=False,
                        help='The number of train, val and test images to '
                             'create.')
    parser.add_argument("--image_size", type=int, default=192,
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


def create_toy_data_point(img_size):
    im = np.random.randn(*[img_size]*3).astype(np.float32)
    lab = np.random.randint(0, 5, size=[img_size*3]).astype(np.uint8)
    return im, lab


def create_toy_dataset(N, image_size, out_dir, image_subdir, label_subdir):
    image_out_dir = os.path.join(out_dir, image_subdir)
    label_out_dir = os.path.join(out_dir, label_subdir)
    create_folders([out_dir, image_out_dir, label_out_dir])

    for img_id in range(N):
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
