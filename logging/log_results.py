import os
import copy
import numpy as np
import pandas as pd

from MultiViewUNet.utils import create_folders


def init_result_dict_3D(all_images, n_classes):
    if n_classes == 1:
        n_classes = 2

    # Prepare dictionary of per class results
    detailed_results = {"class": range(1, n_classes)}
    for image_id in all_images:
        detailed_results.update({image_id: [np.nan for _ in range(1, n_classes)]})

    return {image_id: [np.nan] for image_id in all_images}, detailed_results


def save_all_3D(results, detailed_res, out_dir):
    txt_res_dir = os.path.join(out_dir, "txt")
    csv_res_dir = os.path.join(out_dir, "csv")

    # Create folders
    create_folders([txt_res_dir, csv_res_dir])

    # Save main text/csv results files
    results_to_txt(results, txt_res_dir, transpose=True)
    results_to_csv(results, csv_res_dir, transpose=True)
    results_to_txt(detailed_res, txt_res_dir, fname="detailed")
    results_to_csv(detailed_res, csv_res_dir, fname="detailed")


def init_result_dicts(views, all_images, n_classes):
    if n_classes == 1:
        n_classes = 2

    results = {"ID": [image_id for image_id in sorted(all_images)]}
    results.update({str(v): [np.nan for _ in all_images] for v in views})
    results.update({"MJ": [np.nan for _ in all_images]})

    # Prepare dictionary of per class results
    inner = {"class": range(1, n_classes)}
    for image_id in all_images:
        inner.update({image_id: [np.nan for _ in range(1, n_classes)]})
    pc_results = {str(v): copy.deepcopy(inner) for v in views}
    pc_results.update({"MJ": copy.deepcopy(inner)})

    return results, pc_results


def to_df(results, transpose=False):
    df = pd.DataFrame(results)
    try:
        df = df.set_index("class")
    except KeyError:
        pass
    if transpose:
        df = df.transpose()
    return df


def results_to_csv(results, res_path, fname=None, transpose=False):
    if fname is None:
        fname = "results"

    # Save results
    df = to_df(results, transpose)
    with open(os.path.join(res_path, "%s.csv" % fname), "w") as out_file:
        out_file.write(df.to_csv() + "\n")


def results_to_txt(results, res_path, fname=None, transpose=False):
    if fname is None:
        fname = "results"

    # Save results
    df = to_df(results, transpose)
    with open(os.path.join(res_path, "%s.txt" % fname), "w") as out_file:
        out_file.write(df.to_string() + "\n")


def save_all(results, pc_results, out_dir):

    # Get output paths
    txt_res_dir = os.path.join(out_dir, "txt")
    csv_res_dir = os.path.join(out_dir, "csv")

    # Create folders
    create_folders([txt_res_dir, csv_res_dir])

    # Save main text/csv results files
    results_to_txt(results, txt_res_dir)
    results_to_csv(results, csv_res_dir)

    # Write detailed results
    for view in pc_results:
        r = pc_results[view]
        view_str = str(view).replace("[", "").replace("]", "").replace(" ", "_")
        results_to_txt(r, txt_res_dir, fname=view_str)
        results_to_csv(r, csv_res_dir, fname=view_str)
