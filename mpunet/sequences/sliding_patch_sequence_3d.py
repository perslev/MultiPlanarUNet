from . import PatchSequence3D
from mpunet.interpolation.linalg import mgrid_to_points
import numpy as np


def standardize_strides(strides):
    if isinstance(strides, list):
        return tuple(strides)
    elif isinstance(strides, tuple):
        return strides
    else:
        return 3 * (int(strides),)


class SlidingPatchSequence3D(PatchSequence3D):
    def __init__(self, strides, no_log=False, *args, **kwargs):
        """
        strides: tuple (s1, s2, s3) of strides or integer s --> (s, s, s)
        """
        super().__init__(no_log=True, *args, **kwargs)

        # Stride attribute gives the number pixel distance between
        # patch samples in 3 dimensions
        self.strides = standardize_strides(strides)
        self.corners = self.get_patch_corners().astype(np.uint16)
        self.ind = np.arange(self.corners.shape[0])

        if not self.is_validation and not no_log:
            self.log()

    @property
    def n_samples(self):
        return self.corners.shape[0] * len(self.data)

    def get_patch_corners(self):
        xc = np.linspace(0, self.dim_r[0], self.strides[0]).astype(np.int)
        yc = np.linspace(0, self.dim_r[1], self.strides[1]).astype(np.int)
        zc = np.linspace(0, self.dim_r[2], self.strides[2]).astype(np.int)

        return mgrid_to_points(np.meshgrid(xc, yc, zc))

    def get_box_coords(self):
        return self.corners[np.random.choice(self.ind)]

    def get_base_patches(self, image_id):
        # Get sliding windows for X
        X = self.data[image_id][0]
        for xc, yc, zc in self.corners:
            patch = X[xc:xc + self.dim, yc:yc + self.dim, zc:zc + self.dim]
            yield patch, (xc, yc, zc)

    def log(self):
        self.logger("Sequence Generator: %s" % self.__class__.__name__)
        self.logger("Box dimensions:     %s" % self.dim)
        self.logger("Image dimensions:   %s" % self.im_dim)
        self.logger("Sample space dim:   %s" % self.dim_r)
        self.logger("Strides:            %s" % list(self.strides))
        self.logger("N boxes:            %s" % self.corners.shape[0])
        self.logger("Batch size:         %s" % self.batch_size)
        self.logger("N fg slices/batch:  %s" % self.n_fg_slices)
