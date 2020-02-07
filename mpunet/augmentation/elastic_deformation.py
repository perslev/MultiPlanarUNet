import numpy as np
from mpunet.interpolation import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter


def elastic_transform_2d(image, labels, alpha, sigma, bg_val=0.0):
    """
    Elastic deformation of images as described in [Simard2003]_.
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.

    Modified from:
    https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a

    Modified to take 3 and 4 dimensional inputs
    Deforms both the image and corresponding label file
    image tri-linear interpolated
    Label volumes nearest neighbour interpolated
    """
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    shape = image.shape[:2]
    channels = image.shape[-1]
    dtype = image.dtype
    bg_val = bg_val if isinstance(bg_val, (list, tuple, np.ndarray)) \
        else [bg_val] * channels

    # Define coordinate system
    coords = np.arange(shape[0]), np.arange(shape[1])

    # Initialize interpolators
    im_intrps = []
    for i in range(channels):
        im_intrps.append(RegularGridInterpolator(coords, image[..., i],
                                                 method="linear",
                                                 bounds_error=False,
                                                 fill_value=bg_val[i],
                                                 dtype=np.float32))

    # Get random elastic deformations
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha

    # Define sample points
    x, y = np.mgrid[0:shape[0], 0:shape[1]]
    indices = np.reshape(x + dx, (-1, 1)), \
              np.reshape(y + dy, (-1, 1))

    # Interpolate all image channels
    image = np.empty(shape=image.shape, dtype=dtype)
    for i, intrp in enumerate(im_intrps):
        image[..., i] = intrp(indices).reshape(shape)

    # Interpolate labels
    if labels is not None:
        lab_intrp = RegularGridInterpolator(coords, labels,
                                            method="nearest",
                                            bounds_error=False,
                                            fill_value=0,
                                            dtype=np.uint8)

        labels = lab_intrp(indices).reshape(shape).astype(labels.dtype)

    # Interpolate and return in image shape
    return image, labels


def elastic_transform_3d(image, labels, alpha, sigma, bg_val=0.0):
    """
    Elastic deformation of images as described in [Simard2003]_.
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.

    Modified from:
    https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a

    Modified to take 3 and 4 dimensional inputs
    Deforms both the image and corresponding label file
    image tri-linear interpolated
    Label volumes nearest neighbour interpolated
    """
    if image.ndim == 3:
        image = np.expand_dims(image, axis=-1)
    shape = image.shape[:3]
    channels = image.shape[-1]
    dtype = image.dtype
    bg_val = bg_val if isinstance(bg_val, (list, tuple, np.ndarray)) \
        else [bg_val] * channels

    # Define coordinate system
    coords = np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])

    # Initialize interpolators
    im_intrps = []
    for i in range(channels):
        im_intrps.append(RegularGridInterpolator(coords, image[..., i],
                                                 method="linear",
                                                 bounds_error=False,
                                                 fill_value=bg_val[i],
                                                 dtype=np.float32))

    # Get random elastic deformations
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha

    # Define sample points
    x, y, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    indices = np.reshape(x + dx, (-1, 1)), \
              np.reshape(y + dy, (-1, 1)), \
              np.reshape(z + dz, (-1, 1))

    # Interpolate all image channels
    image = np.empty(shape=image.shape, dtype=dtype)
    for i, intrp in enumerate(im_intrps):
        image[..., i] = intrp(indices).reshape(shape)

    # Interpolate labels
    if labels is not None:
        lab_intrp = RegularGridInterpolator(coords, labels,
                                            method="nearest",
                                            bounds_error=False,
                                            fill_value=0,
                                            dtype=np.uint8)

        labels = lab_intrp(indices).reshape(shape).astype(labels.dtype)

    # Interpolate and return in image shape
    return image, labels
