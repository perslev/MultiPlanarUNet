import os
from glob import glob
from argparse import ArgumentParser


def get_argparser():
    parser = ArgumentParser(description='Replace .nii(.gz) files matching a '
                                        'GLOB regex in-place to keep only 1 '
                                        'out of multiple channels')
    parser.add_argument("-d", type=str, default="./",
                        help='root dir')
    parser.add_argument("-e", type=str, required=True,
                        help="GLOB regex matching files")
    parser.add_argument("-c", type=int, default=0, help="Keep only this channel")

    return parser


def entry_func(args=None):
    args = vars(get_argparser().parse_args(args))
    base_path = os.path.abspath(args["d"])
    match = os.path.join(base_path, args["e"])
    c = args["c"]

    files = glob(match, recursive=True)

    if files:
        print("Matching %i files:" % len(files))
        for f in files:
            print(f)

        answer = input("\n[OBS] Replace IN-PLACE with files "
                       "containing only channel %i? (y/N) " % c)

        if answer.lower() in ("y", "yes"):
            import nibabel as nib
            import numpy as np

            for f in files:
                im = nib.load(f)
                new_im = im.get_data()[..., c].astype(np.float32)
                nib.save(nib.Nifti1Image(new_im, affine=im.affine), f)
    else:
        print("No matches to GLOB: '%s'" % match)


if __name__ == "__main__":
    entry_func()
