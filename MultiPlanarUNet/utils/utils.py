import re
import os
import numpy as np
import glob
import contextlib


def get_free_gpus():
    from subprocess import check_output

    try:
        output = check_output(["nvidia-smi"], universal_newlines=True)
        gpus = np.array(re.findall(r"[|][ ]{1,5}(\d)[ ]{1,5}", output))
        free = list(map(lambda x: int(x) is 0, re.findall(r"(\d+)MiB[ ]?\/[ ]?\d+MiB", output)))
        return list(gpus[free])
    except FileNotFoundError as e:
        print("[ERROR] nvidia-smi is not installed.")
        return []


def _get_free_gpu(free_GPUs, N=1):
    try:
        free_gpu = ",".join(free_GPUs[0:N])
    except IndexError as e:
        raise OSError("No GPU available.") from e
    return free_gpu


def get_free_gpu(N=1):
    free = get_free_gpus()
    return _get_free_gpu(free, N=N)


def await_and_set_free_gpu(N=1, sleep_seconds=60, logger=None):
    gpu = ""
    if N != 0:
        from time import sleep
        logger = logger or print
        logger("Waiting for free GPU.")
        found_gpu = False
        while not found_gpu:
            gpu = get_free_gpu(N=N)
            if gpu:
                logger("Found free GPU: %s" % gpu)
                found_gpu = True
            else:
                logger("No available GPUs... Sleeping %i seconds." % sleep_seconds)
                sleep(sleep_seconds)
    set_gpu(gpu)


def set_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def add_noise_to_views(views, sd):
    # Add Gaussian noise to views specified in parameter file
    return [np.array(v, dtype=np.float32) + np.random.normal(scale=sd, size=3)
            for v in views]


def get_best_model(model_dir):
    val_models = glob.glob(os.path.join(model_dir, "@epoch*val_loss*"))
    dice_models = glob.glob(os.path.join(model_dir, "@epoch*val_dice*"))
    if val_models and dice_models:
        raise RuntimeError("Found both val_loss and dice_loss weight files."
                           " Which to use is undefined.")

    if val_models:
        metric = np.argmin
        models = val_models
    else:
        metric = np.argmax
        models = dice_models

    scores = []
    for m in models:
        scores.append(re.findall(r"(\d+[.]\d+)", m)[0])

    if scores:
        return os.path.abspath(models[metric(np.array(scores))])
    else:
        return os.path.abspath(os.path.join(model_dir, "model_weights.h5"))


def get_last_model(model_dir):
    models = glob.glob(os.path.join(model_dir, "@epoch*"))
    epochs = []
    for m in models:
        epochs.append(re.findall(r"@epoch_(\d+)_", m)[0])

    if epochs:
        last = np.argmax(epochs)
        return os.path.abspath(models[last]), int(epochs[int(last)])
    else:
        return os.path.join(model_dir, "model_weights.h5"), 0


def get_lr_at_epoch(epoch, log_dir):
    log_path = os.path.join(log_dir, "training.csv")
    if not os.path.exists(log_path):
        raise OSError("No training.csv file found at %s" % log_dir)
    import pandas as pd
    df = pd.read_csv(log_path)
    possible_names = ("lr", "LR", "learning_rate", "LearningRate")
    in_df = [l in df.columns for l in possible_names].index(True)
    col_name = possible_names[in_df]
    return float(df[col_name][int(epoch)]), col_name


def clear_csv_after_epoch(epoch, csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)
    # Remove any trailing runs and remove after 'epoch'
    df = df[np.flatnonzero(df["epoch"] == 0)[-1]:]
    df = df[:epoch+1]
    # Save again
    with open(csv_file, "w") as out_f:
        out_f.write(df.to_csv(index=False))


def set_bias_weights_on_all_outputs(model, train, hparams, logger):
    # This will bias the softmax output layer to output class confidences
    # equal to the class frequency
    if hasattr(model, "out_layers"):
        # Multiple output layers, set bias for each
        layers = model.out_layers
        loaders = [t.image_pair_loader for t in train]
    else:
        layers = [model.layers[-1]]
        loaders = [train.image_pair_loader]
    for layer, loader in zip(layers, loaders):
        set_bias_weights(layer=layer,
                         train_loader=loader,
                         class_counts=hparams.get("class_counts"),
                         logger=logger)


def set_bias_weights(layer, train_loader, class_counts=None, logger=None):
    if layer.activation.__name__ != "softmax":
        raise ValueError("Setting output layer bias currently only supported "
                         "with softmax activation functions. Output layer has "
                         "'%s'" % layer.activation.__name__)

    # Calculate counts if not specified
    if class_counts is None:
        class_counts = train_loader.get_class_weights(return_counts=True,
                                                      unload=bool(train_loader.queue))[1]

    # Compute frequencies
    freq = np.asarray(class_counts/np.sum(class_counts))

    # Compute bias weights
    bias = np.log(freq * np.sum(np.exp(freq)))

    # Get original and set new weights
    weights = layer.get_weights()
    s = weights[-1].shape
    if len(weights) != 2:
        raise ValueError("Output layer does not have bias weights.")
    weights[-1] = bias.reshape(s)
    layer.set_weights(weights)

    if logger:
        logger("Setting bias weights on output layer to:\n%s" % bias)


def get_confidence_dict(path, views):
    # Calculate linearly distributed per class weights between the models
    conf_paths = [path + "/validation_confidence_%s.npz" % v for v in views]
    confs = np.column_stack([np.load(x)["arr_0"] for x in conf_paths])
    confs = (confs/np.sum(confs, axis=1).reshape((confs.shape[0], 1))).T

    # Add each confidence array to a dictionary under key <view>
    confs = {str(v): confs[i] for i, v in enumerate(views)}

    return confs


def get_class_counts(samples, unload=False):
    classes, counts = None, None
    if isinstance(samples, dict):
        for v in samples:
            s = samples[v].flatten()
            r = np.unique(s, return_counts=True)
            if counts is None:
                classes, counts = r
            else:
                counts[np.in1d(np.arange(0, len(classes)), r[0])] += r[1]
    else:
        try:
            samples = samples.flatten()
            classes, counts = np.unique(samples, return_counts=True)
        except AttributeError:
            for image in samples.images:
                r = np.unique(image.labels, return_counts=True)
                if counts is None:
                    classes, counts = r
                else:
                    counts[np.in1d(np.arange(0, len(classes)), r[0])] += r[1]
                if unload:
                    image.unload()

    return classes, counts


def get_class_weights(samples, as_array=False, return_counts=False, unload=False):
    classes, counts = get_class_counts(samples, unload)

    # Get total number of samples
    n_samples = np.sum(counts)

    # Calculate weights by class as inverse of their abundance
    weights = {cls: n_samples/(3*c*len(classes)) for cls, c in zip(classes, counts)}

    if as_array:
        weights = np.array([weights[w] for w in sorted(weights)])
    else:
        counts = {cls: c for cls, c in zip(classes, counts)}

    if not return_counts:
        return weights
    else:
        return weights, counts


def get_sample_weights_vectorizor(class_weights_dict):
    return np.vectorize(class_weights_dict.get)


def get_sample_weights(samples):
    # Map class weights to samples
    class_weights_dict = get_class_weights(samples, as_array=False)
    return get_sample_weights_vectorizor(class_weights_dict)(samples)


def random_split(X, y, fraction):
    # Take random split of validation data
    n_val = int(X.shape[0] * fraction)
    val_ind = np.random.choice(np.arange(X.shape[0]), size=n_val)
    X_val, y_val = X[val_ind], y[val_ind]

    # Get inverse for training set
    X, y = np.delete(X, val_ind, axis=0), np.delete(y, val_ind, axis=0)

    return X, y, X_val, y_val


def create_folders(folders):
    if isinstance(folders, str):
        if not os.path.exists(folders):
            os.mkdir(folders)
    else:
        folders = list(folders)
        for f in folders:
            if f is None:
                continue
            if not os.path.exists(f):
                os.mkdir(f)


@contextlib.contextmanager
def print_options_context(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class DummyContext(object):
    def __enter__(self): return self

    def __exit__(*x): pass


def pred_to_class(tensor, img_dims=3, threshold=0.5, has_batch_dim=False):
    tensor_dim = img_dims + int(has_batch_dim)
    dims = len(tensor.shape)
    if dims == tensor_dim:
        # Check if already integer targets
        if np.issubdtype(tensor.dtype, np.integer):
            return tensor
        else:
            return tensor >= threshold
    elif tensor.shape[-1] == 1:
        if np.issubdtype(tensor.dtype, np.integer):
            # Squeeze last axis
            return np.squeeze(tensor)
        else:
            return tensor >= threshold
    else:
        # Convert predicted probabilities to predicted class
        return tensor.argmax(-1).astype(np.uint8)


def highlighted(string):
    length = len(string) if "\n" not in string else max([len(s) for s in string.split("\n")])
    border = "-" * length
    return "%s\n%s\n%s" % (border, string, border)


def await_PIDs(PIDs, timeout=120):
    if isinstance(PIDs, str):
        for pid in PIDs.split(","):
            wait_for(int(pid), timeout=timeout)
    else:
        wait_for(PIDs, timeout=timeout)


def wait_for(PID, timeout=120):
    """
    Check for a running process with PID 'PID' and only return when the process
    is no longer running. Checks the process list every 'timeout' seconds.
    """
    if not PID:
        return
    if not isinstance(PID, int):
        try:
            PID = int(PID)
        except ValueError as e:
            raise ValueError("Cannot wait for PID '%s', must be an integer"
                             % PID) from e
    _wait_for(PID, timeout)


def _wait_for(PID, timeout=120):
    still_running = True
    import subprocess
    import time
    print("\n[*] Waiting for process PID=%i to terminate..." % PID)
    while still_running:
        ps = subprocess.Popen(('ps', '-p', "%i" % PID), stdout=subprocess.PIPE)
        try:
            output = subprocess.check_output(('grep', '%i' % PID), stdin=ps.stdout)
        except subprocess.CalledProcessError:
            output = False
        ps.wait()

        still_running = bool(output)
        if still_running:
            print("Process %i still running... (sleeping %i seconds)" % (PID, timeout))
            time.sleep(timeout)
