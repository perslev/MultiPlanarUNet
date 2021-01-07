"""
TODO: Add description
"""

import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from argparse import ArgumentParser


def get_argparser():
    parser = ArgumentParser(description='Fit a mpunet model defined in a project folder. '
                                        'Invoke "init_project" to start a new project.')
    parser.add_argument("--root_dir", type=str, default="./",
                        help='Starting point from which csv '
                             'folders will be searched for.')
    parser.add_argument("--pred_subdir", type=str, default="predictions",
                        help="Subdirectory storing the 'csv' subfolder.")
    parser.add_argument("--round", default=3, type=int)
    return parser


def print_mj_or_detailed(df, round_):
    classes = df.get("class")
    if classes is not None:
        ddf = df.drop(["class"], axis=1)
    else:
        classes = np.arange(1, df.shape[0]+1)
        ddf = df
    means = np.nanmean(ddf.values, axis=1)
    stds = np.nanstd(ddf.values, axis=1)
    mins = np.nanmin(ddf.values, axis=1)
    maxs = np.nanmax(ddf.values, axis=1)
    N = np.sum(~np.isnan(ddf.values), axis=1)

    print("\nPer class:\n--------------------------------")
    print_df = pd.DataFrame(
        {c: [m, std, min, max, n] for c, m, std, min, max, n in zip(classes, means, stds, mins, maxs, N)}).T
    print_df.columns = ["Mean dice by class", "+/- STD", "min", "max", "N"]

    print(np.round(print_df, round_))
    print("\nOverall mean: %s +- %s" % (
        np.nanmean(ddf.values).round(round_),
        np.nanstd(ddf.values).round(round_)))
    print("--------------------------------")


def print_res(df, round_):
    to_drop = ("Unnamed: 0", "identifier", "MJ")
    for d in to_drop:
        try:
            df = df.drop([d], axis=1)
        except KeyError:
            pass
    print("\nBy views:\n--------------------------------")

    longest = max([len(v) for v in df])
    for v in df:
        mean = np.nanmean(df[v]).round(round_)
        print(("%s" % v).ljust(longest + 7) + "%s" % mean)
    print("--------------------------------")


def print_results(results, folder, round_):
    print("\n[***] SUMMARY REPORT FOR FOLDER [***]\n%s\n" % folder)

    for file_ in results:
        df = results[file_]

        if file_ in ("MJ.csv", "detailed.csv"):
            print_mj_or_detailed(df, round_)
        elif file_ == "results.csv":
            print_res(df, round_)
        else:
            raise ValueError("Unknown file type '%s'" % file_)


def parse_folder(folder, look_for=("MJ", "results", "detailed")):
    import pandas as pd
    dfs = {}
    files = os.listdir(folder)
    for tag in look_for:
        for f in files:
            if tag in f:
                dfs[f] = pd.read_csv(os.path.join(folder, f))

    return dfs


def pool_results(results):
    import pandas as pd
    pooled = {}
    for folder in results:
        for file_ in results[folder]:
            df = results[folder][file_]
            if pooled.get(file_) is not None:
                if file_ in ("MJ.csv", "detailed.csv"):
                    pooled[file_] = pd.merge(pooled[file_], df,
                                             left_index=True,
                                             right_index=True)
                elif file_ == "results.csv":
                    pooled[file_] = pd.concat([pooled[file_], df],
                                              axis=0,
                                              sort=True)
                else:
                    raise ValueError("Unknown file type '%s'" % file_)
            else:
                pooled[file_] = df
    return pooled


def parse_results(csv_folders, round_=3):
    results = {}
    for folder in csv_folders:
        results[folder] = parse_folder(folder)
    results = pool_results(results)
    if len(csv_folders) > 1:
        folder = "Pool of %i folders" % len(csv_folders)
    else:
        folder = csv_folders[0]

    print_results(results, folder, round_)


def entry_func(args=None):

    args = vars(get_argparser().parse_args(args))
    dir_ = os.path.abspath(args["root_dir"])
    p_dir = args["pred_subdir"]
    round_ = args["round"]

    # Get folder/folders - 3 levels possible
    csv_folders = glob("%s/csv" % dir_)
    if not csv_folders:
        csv_folders = glob("%s/%s/csv/" % (dir_, p_dir))
        if not csv_folders:
            csv_folders = glob(dir_ + "/**/%s/csv/" % p_dir)
    if not csv_folders:
        print("Could not locate result csv files.")
        sys.exit(0)
    else:
        print("Found %i 'csv' folders under prediction "
              "dirs '%s':" % (len(csv_folders), p_dir))
        csv_folders.sort()
        for d in csv_folders:
            print("-- " + os.path.abspath(d))
        in_ = input("\nCorrect? (Y/n) ")
        if in_.lower() not in ("n", "no"):
            parse_results(csv_folders, round_)


if __name__ == "__main__":
    entry_func()
