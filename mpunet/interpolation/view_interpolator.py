import numpy as np
from mpunet.interpolation.regular_grid_interpolator import RegularGridInterpolator
from mpunet.logging import ScreenLogger
from mpunet.interpolation.linalg import points_to_mgrid, mgrid_to_points
from mpunet.interpolation.sample_grid import get_voxel_axes_real_space


def is_rot_mat(mat):
    """
    Validate that a square matrix is a rotation matrix
    """
    is_ortho = np.all(np.isclose(mat.dot(mat.T), mat.T.dot(mat)))
    is_unimodular = np.isclose(np.linalg.det(mat), 1)
    return is_ortho and is_unimodular


class ViewInterpolator(object):
    def __init__(self, image, labels, affine,
                 bg_value=0.0, bg_class=0, logger=None):

        # Ensure 4D
        if not image.ndim == 4:
            raise ValueError("Input img of dim %i must be dim 4."
                             "If image has only 1 channel, use "
                             "np.expand_dims(img, -1)." % image.ndim)

        # Set logger
        self.logger = logger if logger is not None else ScreenLogger()

        # Number of channels in the input image
        self.im_shape = image.shape
        self.n_channels = self.im_shape[-1]
        self.im_dtype = image.dtype

        # Cast bg-value to list of length n_channels
        if not isinstance(bg_value, (list, tuple, np.ndarray)):
            bg_value = [bg_value] * self.n_channels
        if not len(bg_value) == self.n_channels:
            raise ValueError("'bg_value' should be a list of length "
                             "'n_channels'. Got {} for n_channels={}".format(
                bg_value, self.n_channels))
        self.bg_value = bg_value

        # Store potential transformation to regular grid
        self.rot_mat = None

        # Define interpolators
        self.im_intrps, self.lab_intrp = self._init_interpolators(image,
                                                                  labels,
                                                                  bg_value,
                                                                  bg_class,
                                                                  affine)

    def apply_rotation(self, mgrid):
        if self.rot_mat is not None:
            shape = mgrid[0].shape
            rotated = self.rot_mat.dot(mgrid_to_points(mgrid).T).T
            return points_to_mgrid(rotated, shape)
        else:
            return mgrid

    def __call__(self, rgrid_mgrid):
        # Align grid if necessary
        rgrid_mgrid = self.apply_rotation(rgrid_mgrid)

        # Interpolate image and labels
        image = self.intrp_image(rgrid_mgrid, apply_rot=False)
        labels = self.intrp_labels(rgrid_mgrid, apply_rot=False)

        return image, labels

    def intrp_image(self, mgrid, apply_rot=True):
        if apply_rot:
            mgrid = self.apply_rotation(mgrid)

        if not isinstance(mgrid, tuple):
            # RegularGridInterpolator expects this tuple(xx, yy, zz) format
            mgrid = tuple(mgrid)

        # Interpolate image along all channels
        image = np.zeros(shape=mgrid[0].squeeze().shape + (self.n_channels,),
                         dtype=self.im_dtype)

        for i in range(self.n_channels):
            image[..., i] = self.im_intrps[i](mgrid).squeeze()

        return image

    def intrp_labels(self, mgrid, apply_rot=True):
        if apply_rot:
            mgrid = self.apply_rotation(mgrid)

        # Interpolate labels
        if self.lab_intrp:
            if not isinstance(mgrid, tuple):
                # RegularGridInterpolator expects tuple(xx, yy, zz) format
                mgrid = tuple(mgrid)

            return self._cast_labels(self.lab_intrp(mgrid).squeeze())
        else:
            return None

    def _init_interpolators(self, image, labels, bg_value, bg_class, affine):

        # Get voxel regular grid centered in real space
        g_all, basis, rot_mat = get_voxel_axes_real_space(image, affine,
                                                          return_basis=True)

        # Set rotation matrix
        self.rot_mat = rot_mat

        # Flip axes? Must be strictly increasing
        flip = np.sign(np.diagonal(basis)) == -1
        assert not np.any(flip)
        g_xx, g_yy, g_zz = g_all

        # Set interpolator for image, one for each channel
        im_intrps = []
        for i in range(self.n_channels):
            im_intrps.append(RegularGridInterpolator((g_xx, g_yy, g_zz),
                                                     image[..., i].squeeze(),
                                                     bounds_error=False,
                                                     fill_value=bg_value[i],
                                                     method="linear",
                                                     dtype=np.float32))

        try:
            # Set interpolator for labels
            lab_intrp = RegularGridInterpolator((g_xx, g_yy, g_zz), labels,
                                                bounds_error=False,
                                                fill_value=bg_class,
                                                method="nearest",
                                                dtype=np.uint8)
        except (AttributeError, TypeError, ValueError):
            lab_intrp = None

        return im_intrps, lab_intrp

    def _cast_labels(self, labels):
        # Cast labels float64 -> uint8/16
        type_info = np.iinfo(np.uint8)
        l = type_info.min
        u = type_info.max
        if np.all((labels >= l) & (labels <= u)):
            return labels.astype(np.uint8)
        else:
            return labels.astype(np.uint16)
