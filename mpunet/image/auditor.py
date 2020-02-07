from mpunet.interpolation.sample_grid import get_real_image_size, get_pix_dim
from mpunet.utils import highlighted
from mpunet.logging import ScreenLogger
import nibabel as nib
import numpy as np


def _audit_classes(nii_lab_paths, logger):
    logger("Auditing number of target classes. This may take "
           "a while as data must be read from disk."
           "\n-- Note: avoid this by manually setting the "
           "n_classes attribute in train_hparams.yaml.")
    # Select up to 50 random images and find the unique classes
    lab_paths = np.random.choice(nii_lab_paths,
                                 min(50, len(nii_lab_paths)),
                                 replace=False)
    classes = []
    for l in lab_paths:
        classes.extend(np.unique(nib.load(l).get_data()))
    classes = np.unique(classes)
    n_classes = classes.shape[0]

    # Make sure the classes start from 0 and step continuously by 1
    c_min, c_max = np.min(classes), np.max(classes)
    if c_min != 0:
        raise ValueError("Invalid class audit - Class integers should"
                         " start from 0, found %i (classes found: %s)"
                         % (c_min, classes))
    if n_classes != max(classes) + 1:
        raise ValueError("Invalid class audit - Found %i classes, but"
                         " expected %i, as the largest class value"
                         " found was %i. Classes found: %s"
                         % (n_classes, c_max+1, c_max, classes))
    return n_classes


class Auditor(object):
    """
    Parses all .nii/.nii.gz images of a specified folder and proposes
    heuristically determined interpolation parameters for models working in
    isotropic scanner space coordinates.

    If label paths are specified, also audits the number of target classes
    for this segmentation task by sampling up to 50 images and noting the
    number of unique classes across them.

    Suggested parameters are stored for both 2D and 3D models on this object
    The selected parameters can be written to a mpunet YAMLHParams
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
    NOTE: This class allows for fully autonomous use of the 2D mpunet
    and 3D UNet models when used with the mpunet.bin scrips.

    NOTE: The heuristic is not guaranteed to be optimal for all problems.
    """
    def __init__(self, nii_paths, nii_lab_paths=None, logger=None,
                 min_dim_2d=128, max_dim_2d=512, dim_3d=64, span_percentile=75,
                 res_percentile=25, hparams=None):
        """
        Args:
            nii_paths: A list of paths pointing to typically training and val
                       .nii/.nii.gz images to audit
            nii_lab_paths: Optional paths pointing to .nii/.nii.gz label images
                           from which target class number is inferred
            logger: A mpunet logger object
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
        self.nii_lab_paths = nii_lab_paths
        self.logger = logger or ScreenLogger()

        # Fetch basic information on the images
        self.hparms = hparams
        self.info = self.audit()

        """ Set some attributes used for image sampling """
        assert np.all(np.array(self.info["n_channels"]) == self.info["n_channels"][0])
        self.n_channels = int(self.info["n_channels"][0])

        # Number of classes
        self.n_classes = self.info["n_classes"]

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
        # These patterns map a Auditor attribute to the sub-field and name
        # under this field in which the value should be stored in the
        # train_hparams.yaml file
        # Currently, these are specified broadly for 2D and 3D models
        # TODO: Integrate this with the logic of the mpunet.models
        # TODO: __init__.py file that already sets preprep functions for each
        # TODO: model type.
        self.patterns = {
            "2d": {
                "real_space_span_2D": (["fit"], ["real_space_span"]),
                "sample_dim_2D": (["build"], ["dim"]),
                "n_channels": (["build"], ["n_channels"]),
                "n_classes": (["build"], ["n_classes"])
            },
            "3d": {
                "real_space_span_3D": (["fit"], ["real_space_span"]),
                "sample_dim_3D": (["build"], ["dim"]),
                "real_box_span": (["fit"], ["real_box_dim"]),
                "n_channels": (["build"], ["n_channels"]),
                "n_classes": (["build"], ["n_classes"])
            },
            "multi_task_2d": {
                "real_space_span_2D": (["task_specifics"], ["real_space_span"]),
                "sample_dim_2D": (["task_specifics"], ["dim"]),
                "n_channels": (["task_specifics"], ["n_channels"]),
                "n_classes": (["task_specifics"], ["n_classes"])
            }
        }

        # Write to log
        self.log()

    def log(self):
        self.logger(highlighted("\nAudit for %i images" % len(self.nii_paths)))
        self.logger("Total memory GiB:  %.3f" % self.total_memory_gib)
        if self.n_classes is not None:
            self.logger("Number of classes: %i" % self.n_classes)
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
        """
        Add and write attributes stored in this Auditor object to the
        YAMLHParams object and train_hparams.yaml file according to the
        patterns self.pattern_2d and self.pattern_3d (see init)

        Only attributes not already manually specified by the user will be
        changed. See YAMLHParams.set_value().

        Args:
            hparams:     mpunet YAMLHParams object
            model_type:  A string representing the model type and thus which
                         pattern to apply. Must be either "2d", "3d" (upper case tolerated)
        """
        model_type = model_type.lower()
        pattern = self.patterns.get(model_type)
        if pattern is None:
            raise ValueError("Unknown model type: '%s'" % model_type)

        for attr in pattern:
            subdirs, names = pattern[attr]
            value = getattr(self, attr)

            for s, n in zip(subdirs, names):
                hparams.set_value(subdir=s, name=n, value=value)
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

        for im_path in self.nii_paths:
            # Load the nii file without loading image data
            im = nib.load(im_path)

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

        n_classes = None
        if self.nii_lab_paths is not None:
            # Labels exists, thus we need the n_classes attribute
            if self.hparms is not None:
                # Attempt to get it from a potentially specified hparams obj
                n_classes = self.hparms.get_from_anywhere("n_classes")
            if n_classes is None:
                # If still none, infer it
                n_classes = _audit_classes(self.nii_lab_paths,
                                           self.logger)

        info = {
            "shapes": shapes,
            "real_sizes": real_sizes,
            "pixdims": pixdims,
            "memory_bytes": memory,
            "n_channels": channels,
            "n_classes": n_classes,
        }
        return info
