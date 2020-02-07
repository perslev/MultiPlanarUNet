"""
See
https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
"""

from tensorflow._api.v1.keras import backend as K
from PIL import Image as pil_image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def _get_figure(figsize, rows=1, cols=1):
    figsize = (figsize, figsize) if isinstance(figsize, int) else figsize
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.subplots_adjust(hspace=0.005, wspace=0.005)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    if not isinstance(axes[0], (list, np.ndarray)):
        axes = [axes]
    for row in axes:
        for ax in row:
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
    return fig, axes


def display_filter(filter, out_path=None, dpi=200, figsize=6, overwrite=False):
    fig, ax = _get_figure(figsize)
    ax[0].imshow(filter[0, :, :, 0], cmap="gray")
    fig.tight_layout()
    out_path = out_path or "filter.png"
    if not overwrite and os.path.exists(out_path):
        raise OSError("Out path {} already exists.".format(out_path))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def display_filter_grid(filters, out_path=None, dpi=300, fig_width=8,
                        overwrite=False, layer_names=None):
    filters = np.array(filters)
    if filters.ndim == 3:
        filters = np.expand_dims(filters, 0)

    # Compute figure height and get figure with subplots
    rows, cols = filters.shape[0:2]
    ax_size = fig_width / cols
    fig_height = ax_size * rows
    fig, axes = _get_figure((fig_width, fig_height), rows=rows, cols=cols)

    # Plot the filters
    for layer_ind, (row, row_filters) in enumerate(zip(np.asarray(axes),
                                                       filters)):
        vmin, vmax = np.min(row_filters), np.max(row_filters)
        for i, (ax, filter) in enumerate(zip(row, row_filters)):
            ax.imshow(filter, cmap="gray", vmin=vmin, vmax=vmax)
            if i == 0:
                name = layer_names[layer_ind] if layer_names else layer_ind
                if len(name) > 10:
                    name = name[:10] + "\n" + name[10:]
                ax.annotate(s="{}".format(name),
                            xy=(-0.2, 0.5),
                            xycoords=ax.transAxes,
                            ha="center", va="center",
                            rotation=90, size=ax_size * 7)
    # Save image
    out_path = out_path or "filters.png"
    if not overwrite and os.path.exists(out_path):
        raise OSError("Out path {} already exists.".format(out_path))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def deprocess_image(x):
    """utility function to convert a float array into a valid uint8 image.
    # Arguments
        x: A numpy-array representing the generated image.
    # Returns
        A processed numpy-array, which could be used in e.g. imshow.
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def process_image(x, former):
    """utility function to convert a valid uint8 image back into a float array.
       Reverses `deprocess_image`.
    # Arguments
        x: A numpy-array, which could be used in e.g. imshow.
        former: The former numpy-array.
                Need to determine the former mean and variance.
    # Returns
        A processed numpy-array representing the generated image.
    """
    if K.image_data_format() == 'channels_first':
        x = x.transpose((2, 0, 1))
    return (x / 255 - 0.5) * 4 * former.std() + former.mean()


def _run_optim(img, iterate, dim, step, steps):
    loss_value = 0.0
    for i in range(steps):
        s = "[*] Size {} - Step {}/{} - " \
            "loss {:.3f}".format(dim, i + 1, steps, loss_value)
        print(s, end="\r", flush=True)

        loss_value, grads_value = iterate([img])
        img += grads_value * step
        if loss_value <= K.epsilon():
            # some filters get stuck to 0, we can skip them
            break
    return img, loss_value


def visualize_filter(model,
                     layer_ind,
                     filter_index,
                     steps=20,
                     lr=1.0,
                     upscaling_steps=9,
                     upscaling_factor=1.2,
                     output_dim=(384, 384)):

    # Build loss function for the specified filter of the specified layer
    layer = model.layers[layer_ind]
    if not isinstance(layer, tf.keras.layers.Conv2D):
        raise ValueError("Currently supports only Conv2D layers, "
                         "got {}".format(type(layer)))
    loss = K.mean(K.abs(layer.output[..., filter_index]))
    input_image = model.inputs[0]
    grads = K.gradients(loss, input_image)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate_func = K.function([input_image], [loss, grads])

    intermediate_dim = tuple(
        int(x / (upscaling_factor ** upscaling_steps)) for x in output_dim
    )
    img = np.random.randn(1, intermediate_dim[0], intermediate_dim[1], 1) * 0.1

    for up in reversed(range(upscaling_steps)):
        # we run gradient ascent for e.g. 20 steps
        img, loss_value = _run_optim(img, iterate_func, intermediate_dim,
                                     lr, steps)

        # Calculate up-scaled dimensions
        intermediate_dim = tuple(
            int(x / (upscaling_factor ** up)) for x in output_dim
        )
        # Upscale
        temp_img = deprocess_image(img[0])
        temp_img = np.stack((temp_img.squeeze(),)*3, axis=-1)
        temp_img = pil_image.fromarray(temp_img)
        temp_img = np.array(temp_img.resize(intermediate_dim,
                                            pil_image.BICUBIC))[..., 0:1]
        img = np.array([process_image(temp_img, img[0])])

    # Run one final time on full-size image
    img, loss_value = _run_optim(img, iterate_func, intermediate_dim,
                                 lr, steps)
    print("")
    return img.squeeze(), loss_value


if __name__ == "__main__":
    from mpunet.bin.predict import get_model, load_hparams
    from mpunet.utils import await_and_set_free_gpu

    p = "/home/jovyan/work/projects/knee_projects/" \
        "iwoai_challenge/splits/split_0"
    # Get model
    await_and_set_free_gpu(1)
    hparams = load_hparams(p)
    model = get_model(p, 1, hparams["build"])[0]

    # SETTINGS
    layer_range = range(0, len(model.layers))
    filter_range = range(0, 200)
    n_highest = 9
    steps = 30
    lr = 0.5
    upscaling_steps = 19

    per_layer_filters = []
    layer_names = []
    for layer_ind in layer_range:
        layer = model.layers[layer_ind]
        print("\n[*] Layer {}".format(layer_ind))
        if not isinstance(layer, tf.keras.layers.Conv2D):
            print("--- skipping (not Conv2D)")
            continue
        filters, losses = [], []
        for filter_ind in filter_range:
            try:
                filter, loss = visualize_filter(model=model,
                                                layer_ind=layer_ind,
                                                steps=steps,
                                                lr=lr,
                                                upscaling_steps=upscaling_steps,
                                                filter_index=filter_ind)
            except ValueError as e:
                break
            filters.append(filter), losses.append(loss)

        # Save filters with highest losses
        inds = np.argsort(losses)[::-1][:n_highest]
        per_layer_filters.append([filters[i] for i in inds])
        layer_names.append(layer.name)

        display_filter_grid(per_layer_filters,
                            overwrite=True,
                            layer_names=layer_names,
                            fig_width=10,
                            dpi=1200)
