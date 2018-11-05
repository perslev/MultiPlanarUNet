from MultiViewUNet.interpolation.sample_grid import get_real_image_size, get_pix_dim
from MultiViewUNet.utils import highlighted
from MultiViewUNet.logging import ScreenLogger
import nibabel as nib
import numpy as np


class Auditor(object):
    """
    Parses all .nii/.nii.gz images of a specified folder and proposes
    heuristically determined interpolation parameters for models working in
    isotropic scanner space coordinates.

    Suggested parameters are stored for both 2D and 3D models on this object
    The selected parameters can be written to a MultiVieUNet.train.hparams
    object which in turn may write them to the train_hparams.yaml file on disk.

    The heuristic covers 3 parameters:
        1) The sample dimension
           - The number of pixels to sample in each dim.
        2) The real space span
           - The number of mm along each axis that should define the sample
             space around the image (real space) center within which image may
             be sampled.
        3) Real box dimension (3D only)
           - The number of mm the sampled 3D box spans along all 3 axes

    In addition, the auditor stores the number of channels in the images and
    estimated the total memory needed to store all images in memory.

    See paper for a description of how the heuristic define those parameters

    ------
    NOTE: This class allows for fully autonomous use of the 2D MultiViewUnet
    and 3D UNet models when used with the MultiViewUNet.bin scrips.

    NOTE: The heuristic is not guaranteed to be optimal for all problems.
    """
    def __init__(self, nii_paths, logger=None, min_dim_2d=128, max_dim_2d=512,
                 dim_3d=64, span_percentile=75, res_percentile=25):
        """
        Args:
            nii_paths: Path to a folder storing a set of (typically training)
                       .nii/.nii.gz images to audit
            logger: A MultiViewUNet logger object
            min_dim_2d: Minimum pixel dimension to use
            max_dim_2d: Maximum pixel dimension to use (usually GPU limited)
            dim_3d: Pixel dimensionality of the 3D model
            span_percentile: The real space span param will be set close to the
                             'span_percentile' percentile computed across all
                             spans recorded across images and axes.
            res_percentile: The sampled resolution will be set close to the
                            'span_percentile' percentile computed across all
                            voxel resolutions recorded across images and axes.
        """
        self.nii_paths = nii_paths
        self.logger = logger or ScreenLogger()

        # Fetch basic information on the images
        self.info = self.audit()

        """ Set some attributes used for image sampling """
        assert np.all(np.array(self.info["n_channels"]) == self.info["n_channels"][0])
        self.n_channels = int(self.info["n_channels"][0])

        # 2D
        real_space_span = np.percentile(self.info["real_sizes"], span_percentile)
        sample_res = np.percentile(self.info["pixdims"], res_percentile)
        self.sample_dim_2D, self.real_space_span_2D = self.heurestic_sample_dim(real_space_span,
                                                                                sample_res,
                                                                                min_dim_2d, max_dim_2d)

        # 3D
        self.sample_dim_3D = dim_3d
        self.real_space_span_3D = real_space_span
        self.real_box_span = dim_3d * sample_res

        # Total memory (including channels)
        self.total_memory_bytes = sum(self.info["memory_bytes"])
        self.total_memory_gib = self.total_memory_bytes/np.power(1024, 3)

        # Set hparams pattern
        self.pattern_2d = {
            "real_space_span_2D": (["fit"], ["real_space_span"]),
            "sample_dim_2D": (["build", "fit"], ["dim", "sample_dim"]),
            "n_channels": (["build"], ["n_channels"])
        }
        self.pattern_3d = {
            "real_space_span_3D": (["fit"], ["real_space_span"]),
            "sample_dim_3D": (["build", "fit"], ["dim", "sample_dim"]),
            "real_box_span": (["fit"], ["real_box_dim"]),
            "n_channels": (["build"], ["n_channels"])
        }

        # Write to log
        self.log()

    def log(self):
        self.logger(highlighted("\nAudit for %i images" % len(self.nii_paths)))
        self.logger("Total memory GiB:  %.3f" % self.total_memory_gib)
        self.logger("\n2D:\n"
                    "Real space span:   %.3f\n"
                    "Sample dim:        %.3f" % (self.real_space_span_2D,
                                                 self.sample_dim_2D))
        self.logger("\n3D:\n"
                    "Sample dim:        %i\n"
                    "Real space span:   %.3f\n"
                    "Box span:          %.3f" % (self.sample_dim_3D,
                                                 self.real_space_span_3D,
                                                 self.real_box_span))

    def fill(self, hparams, model_type):
        if model_type.lower() == "2d":
            pattern = self.pattern_2d
        elif model_type.lower() == "3d":
            pattern = self.pattern_3d
        else:
            raise ValueError("Unknown model type: '%s'" % model_type)

        changes = False
        for attr in pattern:
            subdirs, names = pattern[attr]
            value = getattr(self, attr)

            for s, n in zip(subdirs, names):
                c = hparams.set_value(s, n, value, update_string_rep=True)
                if c:
                    changes = True
        if changes:
            hparams.save_current()

    def heurestic_sample_dim(self, real_space_span, res, _min, _max):
        valid = np.array([i for i in range(_min, _max+1) if (i*0.5**4).is_integer()])
        sample_dim = real_space_span / res
        nearest_valid = valid[np.abs(valid - sample_dim).argmin()]

        if nearest_valid < (sample_dim * 0.90):
            # Reduce real space span a bit to increase resolution
            pref = nearest_valid * res
            real_space_span = max(int(real_space_span * 0.70), pref)

        return nearest_valid, real_space_span

    def audit(self):
        shapes = []
        channels = []
        real_sizes = []
        pixdims = []
        memory = []

        for path in self.nii_paths:
            # Load the nii file without loading image data
            im = nib.load(path)

            # Get image voxel shape
            shape = im.shape
            shapes.append(shape[:3])

            try:
                c = shape[3]
            except IndexError:
                c = 1
            channels.append(c)

            # Get image real shape
            real_sizes.append(get_real_image_size(im))

            # Get pixel dims
            pixdims.append(get_pix_dim(im))

            # Calculate memory in bytes to store image
            memory.append(im.get_data_dtype().itemsize * np.prod(shape))

        info = {
            "shapes": shapes,
            "real_sizes": real_sizes,
            "pixdims": pixdims,
            "memory_bytes": memory,
            "n_channels": channels
        }
        return info
