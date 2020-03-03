import re
import os
import numpy as np
import glob
import contextlib


def _get_system_wide_set_gpus():
    allowed_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if allowed_gpus:
        allowed_gpus = allowed_gpus.replace(" ", "").split(",")
    return allowed_gpus


def get_free_gpus(max_allowed_mem_usage=400):
    # Check if allowed GPUs are set in CUDA_VIS_DEV.
    allowed_gpus = _get_system_wide_set_gpus()
    if allowed_gpus:
        print("[OBS] Considering only system-wise allowed GPUs: {} (set in"
              " CUDA_VISIBLE_DEVICES env variable).".format(allowed_gpus))
        return allowed_gpus
    # Else, check GPUs on the system and assume all non-used (mem. use less
    # than max_allowed_mem_usage) is fair game.
    from subprocess import check_output
    try:
        # Get list of GPUs
        gpu_list = check_output(["nvidia-smi", "-L"], universal_newlines=True)
        gpu_ids = np.array(re.findall(r"GPU[ ]+(\d+)", gpu_list), dtype=np.int)

        # Query memory usage stats from nvidia-smi
        output = check_output(["nvidia-smi", "-q", "-d", "MEMORY"],
                              universal_newlines=True)

        # Fetch the memory usage of each GPU
        mem_usage = re.findall(r"FB Memory Usage.*?Used[ ]+:[ ]+(\d+)",
                               output, flags=re.DOTALL)
        assert len(gpu_ids) == len(mem_usage)

        # Return all GPU ids for which the memory usage is exactly 0
        free = list(map(lambda x: int(x) <= max_allowed_mem_usage, mem_usage))
        return list(gpu_ids[free])
    except FileNotFoundError as e:
        raise FileNotFoundError("[ERROR] nvidia-smi is not installed. "
                                "Consider setting the --num_GPUs=0 flag.") from e


def _get_free_gpu(free_GPUs, N=1):
    try:
        free_gpu = ",".join(map(str, free_GPUs[0:N]))
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
    if len(os.listdir(model_dir)) == 0:
        raise OSError("Model dir {} is empty.".format(model_dir))
    # look for models, order: val_dice, val_loss, dice, loss, model_weights
    patterns = [
        ("@epoch*val_dice*", np.argmax),
        ("@epoch*val_loss*", np.argmin),
        ("@epoch*dice*", np.argmax),
        ("@epoch*loss*", np.argmin)
    ]
    for pattern, select_func in patterns:
        models = glob.glob(os.path.join(model_dir, pattern))
        if models:
            scores = []
            for m in models:
                scores.append(float(re.findall(r"(\d+[.]\d+)", m)[0]))
            return os.path.abspath(models[select_func(np.array(scores))])
    m = os.path.abspath(os.path.join(model_dir, "model_weights.h5"))
    if not os.path.exists(m):
        raise OSError("Did not find any model files matching the patterns {} "
                      "and did not find a model_weights.h5 file."
                      "".format(patterns))
    return m


def get_last_model(model_dir):
    models = glob.glob(os.path.join(model_dir, "@epoch*"))
    epochs = []
    for m in models:
        epochs.append(int(re.findall(r"@epoch_(\d+)_", m)[0]))
    if epochs:
        last = np.argmax(epochs)
        return os.path.abspath(models[last]), int(epochs[int(last)])
    else:
        generic_path = os.path.join(model_dir, "model_weights.h5")
        if os.path.exists(generic_path):
            # Return epoch 0 as we dont know where else to start
            # This may be changed elsewhere in the code based on the
            # training data CSV file
            return generic_path, 0
        else:
            # Start from scratch, or handle as see fit at call point
            return None, None


def get_lr_at_epoch(epoch, log_dir):
    log_path = os.path.join(log_dir, "training.csv")
    if not os.path.exists(log_path):
        print("No training.csv file found at %s. Continuing with default "
              "learning rate found in parameter file." % log_dir)
        return None, None
    import pandas as pd
    df = pd.read_csv(log_path)
    possible_names = ("lr", "LR", "learning_rate", "LearningRate")
    try:
        in_df = [l in df.columns for l in possible_names].index(True)
    except ValueError:
        return None, None
    col_name = possible_names[in_df]
    return float(df[col_name][int(epoch)]), col_name


def clear_csv_after_epoch(epoch, csv_file):
    if os.path.exists(csv_file):
        import pandas as pd
        try:
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            # Remove the file
            os.remove(csv_file)
            return
        # Remove any trailing runs and remove after 'epoch'
        try:
            df = df[np.flatnonzero(df["epoch"] == 0)[-1]:]
        except IndexError:
            pass
        df = df[:epoch+1]
        # Save again
        with open(csv_file, "w") as out_f:
            out_f.write(df.to_csv(index=False))


def get_last_epoch(csv_file):
    epoch = 0
    if os.path.exists(csv_file):
        import pandas as pd
        df = pd.read_csv(csv_file)
        epoch = int(df["epoch"].to_numpy()[-1])
    return epoch


def set_bias_weights_on_all_outputs(model, data_queue, hparams, logger):
    # This will bias the softmax output layer to output class confidences
    # equal to the class frequency
    if hasattr(model, "out_layers"):
        raise NotImplementedError("Multi task models not yet supported in "
                                  "mpunet >= 0.2.6")
        # Multiple output layers, set bias for each
        layers = model.out_layers
        loaders = [t.image_pair_loader for t in data_queue]
    else:
        layers = [None]
        for layer in model.layers[::-1]:
            # Start from last layer and go up until one that has an activation
            # funtion is met (note: this skips layers like Reshape, Cropping
            # etc.)
            if hasattr(layer, 'activation'):
                layers = [layer]
                break
        data_queues = [data_queue]
    for layer, data_queue in zip(layers, data_queues):
        set_bias_weights(layer=layer,
                         data_queue=data_queue,
                         class_counts=hparams.get("class_counts"),
                         logger=logger)


def set_bias_weights(layer, data_queue, class_counts=None, logger=None):
    if layer.activation.__name__ != "softmax":
        raise ValueError("Setting output layer bias currently only supported "
                         "with softmax activation functions. Output layer has "
                         "'%s'" % layer.activation.__name__)
    from mpunet.logging.default_logger import ScreenLogger
    logger = logger or ScreenLogger()
    # Get original and set new weights
    weights = layer.get_weights()
    if len(weights) != 2:
        raise ValueError("Output layer does not have bias weights.")
    bias_shape = weights[-1].shape
    n_classes = weights[-1].size

    # Estimate counts if not specified
    if class_counts is None:
        class_counts = np.zeros(shape=[n_classes], dtype=np.int)
        if hasattr(data_queue, 'max_loaded'):
            # Limitation queue, count once for each image currently in queue
            n_images = data_queue.max_loaded
        else:
            n_images = len(data_queue.dataset)
        logger("OBS: Estimating class counts from {} images".format(n_images))
        for _ in range(n_images):
            with data_queue.get_random_image() as image:
                class_counts += np.bincount(image.labels.ravel(),
                                            minlength=n_classes)

    # Compute frequencies
    freq = np.asarray(class_counts/np.sum(class_counts))

    # Compute bias weights
    bias = np.log(freq * np.sum(np.exp(freq)))
    bias /= np.linalg.norm(bias)
    weights[-1] = bias.reshape(bias_shape)

    layer.set_weights(weights)
    logger("Setting bias weights on output layer to:\n%s" % bias)


def get_confidence_dict(path, views):
    # Calculate linearly distributed per class weights between the models
    conf_paths = [path + "/validation_confidence_%s.npz" % v for v in views]
    confs = np.column_stack([np.load(x)["arr_0"] for x in conf_paths])
    confs = (confs/np.sum(confs, axis=1).reshape((confs.shape[0], 1))).T

    # Add each confidence array to a dictionary under key <view>
    confs = {str(v): confs[i] for i, v in enumerate(views)}

    return confs


def random_split(X, y, fraction):
    # Take random split of validation data
    n_val = int(X.shape[0] * fraction)
    val_ind = np.random.choice(np.arange(X.shape[0]), size=n_val)
    X_val, y_val = X[val_ind], y[val_ind]

    # Get inverse for training set
    X, y = np.delete(X, val_ind, axis=0), np.delete(y, val_ind, axis=0)

    return X, y, X_val, y_val


def create_folders(folders, create_deep=False):
    def safe_make(path, make_func):
        try:
            make_func(path)
        except FileExistsError:
            # If running many jobs in parallel this may occur
            pass
    make_func = os.mkdir if not create_deep else os.makedirs
    if isinstance(folders, str):
        if not os.path.exists(folders):
            safe_make(folders, make_func)
    else:
        folders = list(folders)
        for f in folders:
            if f is None:
                continue
            if not os.path.exists(f):
                safe_make(f, make_func)


@contextlib.contextmanager
def print_options_context(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def arr_to_fixed_precision_string(arr, precision):
    f = np.format_float_positional
    s = map(lambda x: f(x, precision, pad_right=precision), arr)
    return "[{}]".format(" ".join(s))


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


def await_PIDs(PIDs, check_every=120):
    if isinstance(PIDs, str):
        for pid in PIDs.split(","):
            wait_for(int(pid), check_every=check_every)
    else:
        wait_for(PIDs, check_every=check_every)


def wait_for(PID, check_every=120):
    """
    Check for a running process with PID 'PID' and only return when the process
    is no longer running. Checks the process list every 'check_every' seconds.
    """
    if not PID:
        return
    if not isinstance(PID, int):
        try:
            PID = int(PID)
        except ValueError as e:
            raise ValueError("Cannot wait for PID '%s', must be an integer"
                             % PID) from e
    _wait_for(PID, check_every)


def _wait_for(PID, check_every=120):
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
            print("Process %i still running... (sleeping %i seconds)" % (PID, check_every))
            time.sleep(check_every)


def check_kwargs(kwargs, allowed, func=None):
    s = ("Function '{}': ".format(func.__name__)) if func is not None else ""
    for param in kwargs:
        if param not in allowed:
            raise RuntimeError("{}Unexpected parameter '{}' passed.".
                               format(s, param))


def ensure_list_or_tuple(obj):
    return [obj] if not isinstance(obj, (list, tuple)) else obj
