"""
Mathias Perslev
MSc Bioinformatics

University of Copenhagen
November 2017
"""

import numpy as np
import nibabel as nib
from contextlib import contextmanager

from pathlib import Path
from mpunet.preprocessing import get_scaler
from mpunet.logging import ScreenLogger
from mpunet.interpolation.sample_grid import get_real_image_size, get_pix_dim
from mpunet.interpolation.view_interpolator import ViewInterpolator
from mpunet.utils import ensure_list_or_tuple

# Errors
from mpunet.errors.image_errors import ReadOnlyAttributeError

# w2 negative threshold is too strict for this data set
nib.Nifti1Header.quaternion_threshold = -1e-6


class ImagePair(object):
    """
    ImagePair
    ---
    Represents one data point of 1 .nii image file and corresponding labels
    """
    def __init__(self, img_path, labels_path=None, sample_weight=1.0,
                 bg_class=0, logger=None, im_dtype=np.float32,
                 lab_dtype=np.uint8):
        """
        Initializes the ImagePair object from two paths to .nii file images

        The following actions are taken immediately on initialization:
            - Image and label Nibabel objects are created and metadata is
              loaded, but image data is not yet loaded into memory
              Loading of data happens automatically at the first reference to
              the image or label attributes
            - In identifier name is established (filename minus extension)
            - Placeholder attributes are created that may be used in methods

        Args:
            img_path:      Path to a Nifti image file (.nii, .nii.gz)
            labels_path:   Path to a Nifti image file (.nii, .nii.gz)
                           Filename must be identical to the img_path filename!
                           Note: A labels_path can be omitted if no ground
                           truth label map exists
            sample_weight: A float value assigning an overall weight to the
                           image/label pair - used by some optimization schemas
            bg_class:       The background class integer value, usually 0
            logger:        A mpunet logger object writing to screen and
                           /or a logfile on disk
            im_dtype:      A numpy data type that the image will be cast to
            lab_dtype:     A numpy data type that the labels will be cast to
        """
        # Labels included?
        self.predict_mode = not labels_path

        # Set a weight for all slices fetches from this image
        # We set value trough property setter to validate input
        self._sample_weight = None
        self.sample_weight = sample_weight

        # Set logger or default print
        self.logger = logger if logger is not None else ScreenLogger()

        # Set image and label paths (absolute)
        self.image_path = self._validate_path(img_path)
        if not self.predict_mode:
            self.labels_path = self._validate_path(labels_path)

        # Validate that the image and label data match and get image identifier
        self.identifier = self._get_and_validate_id()

        # Set variables to store loaded image and label information
        self.image_obj = nib.load(self.image_path)
        self.labels_obj = None
        if not self.predict_mode:
            self.labels_obj = nib.load(self.labels_path)

        # Stores the data of the image and labels objects
        self._image = None
        self._labels = None
        self._scaler = None
        self._bg_value = None
        self._bg_class = int(bg_class)

        # ViewInterpolator object initialized with set_interpolator_object
        self._interpolator = None

        # Data types
        self.im_dtype = im_dtype
        self.lab_dtype = lab_dtype

    def __str__(self):
        return "ImagePair(id={}, shape={}, real_shape={}, loaded={})".format(
            self.identifier, self.shape, self.real_center, self.is_loaded
        )

    def __repr__(self):
        return self.__str__()

    def log_image(self, print_calling_method=False):
        """
        Log basic stats for this ImagePair.
        """
        self.logger("%s\n"
                    "--- loaded:     %s\n"
                    "--- shape:      %s\n"
                    "--- bg class    %i\n"
                    "--- bg value    %s\n"
                    '--- scaler      %s\n'
                    "--- real shape: %s\n"
                    "--- pixdim:     %s" % (
                        self.identifier,
                        self.is_loaded,
                        self.shape,
                        self._bg_class,
                        self._bg_value,
                        ensure_list_or_tuple(self._scaler)[0],
                        np.round(get_real_image_size(self), 3),
                        np.round(get_pix_dim(self), 3)
                    ), print_calling_method=print_calling_method)

    def _get_and_validate_id(self):
        """
        Validates if the image identifier and label identifier match.
        Returns the image identifier.
        """
        img_id = self.image_path.stem.split('.')[0]
        if not self.predict_mode:
            labels_id = self.labels_path.stem.split('.')[0]
            if img_id != labels_id:
                raise ValueError("Image identifier '%s' does not match labels identifier '%s'"
                                 % (img_id, labels_id))
        return img_id

    @property
    def affine(self):
        return self.image_obj.affine

    @affine.setter
    def affine(self, _):
        raise ReadOnlyAttributeError("Manually setting the affine attribute "
                                     "is not allowed. Initialize a new "
                                     "ImagePair object.")

    @property
    def header(self):
        return self.image_obj.header

    @header.setter
    def header(self, _):
        raise ReadOnlyAttributeError("Manually setting the header attribute "
                                     "is not allowed. Initialize a new "
                                     "ImagePair object.")

    @property
    def image(self):
        """
        Ensures image is loaded and then returns it
        Note that we load the Nibabel data with the caching='unchanged'. This
        means that the Nibabel Nifti1Image object does NOT maintain its own
        internal copy of the image. Un-assigning self._image will GC the array.
        """
        if self._image is None:
            self._image = self.image_obj.get_fdata(caching='unchanged',
                                                   dtype=self.im_dtype)
        if self._image.ndim == 3:
            self._image = np.expand_dims(self._image, -1)
        return self._image

    @image.setter
    def image(self, _):
        raise ReadOnlyAttributeError("Manually setting the image attribute "
                                     "is not allowed. Initialize a new "
                                     "ImagePair object.")

    @property
    def labels(self):
        """ Like self.image """
        if self._labels is None:
            try:
                self._labels = self.labels_obj.get_fdata(caching="unchanged").astype(self.lab_dtype)
            except AttributeError:
                return None
        return self._labels

    @labels.setter
    def labels(self, _):
        raise ReadOnlyAttributeError("Manually setting the labels "
                                     "attribute is not allowed. "
                                     "Initialize a new ImagePair object.")

    @staticmethod
    def _validate_path(path):
        path = Path(path)
        if path.exists() and path.suffix in (".nii", ".mat", ".gz"):
            return path
        else:
            raise FileNotFoundError("File '%s' not found or not a "
                                    ".nii or .mat file." % path)

    @property
    def estimated_memory(self):
        """
        Note this will overestimate the memory footprint, actual memory usage
        will never be above this estimation.
        """
        import pickle
        return len(pickle.dumps(self))

    @property
    def sample_weight(self):
        return self._sample_weight

    @sample_weight.setter
    def sample_weight(self, weight):
        try:
            weight = float(weight)
        except ValueError:
            raise ValueError("Sample weight must be a numeric type (got '%s')"
                             % type(weight))
        if weight <= 0 or weight > 1:
            raise ValueError("Sample weight must be greater than "
                             "0 and less than or equal to 1")
        self._sample_weight = weight

    @property
    def center(self):
        """
        Returns:
            The voxel-space center of the image
        """
        return (self.shape[:-1]-1)/2

    @property
    def real_center(self):
        """
        Returns:
            The scanner-space center of the image
        """
        return self.affine[:3, :3].dot(self.center) + self.affine[:3, -1]

    @property
    def shape(self):
        """
        Returns:
            The voxel shape of the image (always rank 4 with channels axis)
        """
        s = np.asarray(self.image_obj.shape)
        if len(s) == 3:
            s = np.append(s, 1)
        return s

    @property
    def real_shape(self):
        """
        Returns:
            The real (physical, scanner-space span) shape of the image
        """
        return get_real_image_size(self.image_obj)

    @property
    def n_channels(self):
        return self.shape[-1]

    @property
    def bg_class(self):
        return self._bg_class

    @bg_class.setter
    def bg_class(self, _):
        raise ReadOnlyAttributeError("Cannot set a new background class. "
                                     "Initialize a new ImagePair object.")

    @property
    def bg_value(self):
        if self._bg_value is None or isinstance(self._bg_value[0], str):
            self.set_bg_value(self._bg_value, compute_now=True)
        return self._bg_value

    @bg_value.setter
    def bg_value(self, _):
        raise ReadOnlyAttributeError("New background values must be set in the"
                                     " self.set_bg_value method.")

    def set_bg_value(self, bg_value, compute_now=False):
        """
        Set a new background value for this ImagePair.

        Args:
            bg_value:    A value defining the space outside of the image region
                         bg_value may be a number of string of the format
                         '[0-100]'pct specifying a percentile value to compute
                         across the image and use for bg_value.
            compute_now: If a percentile string was passed, compute the
                         percentile now (True) or lazily when accessed via
                         self.bg_value at a later time (False).
        """
        bg_value = self.standardize_bg_val(bg_value)
        if compute_now and isinstance(bg_value[0], str):
            bg_value = self._bg_pct_string_to_value(bg_value)
        self._bg_value = bg_value

    @property
    def scaler(self):
        """ Return the currently set scaler object """
        if isinstance(self._scaler, tuple):
            self.set_scaler(*self._scaler, compute_now=True)
        return self._scaler

    @scaler.setter
    def scaler(self, _):
        raise ReadOnlyAttributeError("New scalers must be set with the "
                                     "self.set_scaler method.")

    def set_scaler(self, scaler, ignore_less_eq=None, compute_now=False):
        """
        Sets a scaler on the ImagePair fit to the stored image
        See mpunet.preprocessing.scaling

        Args:
            scaler:         A string naming a sklearn scaler type
            ignore_less_eq: A float or list of floats. Only consider values
                            above 'ignore_less_eq' in a channel when scaling.
            compute_now:    Initialize (and load data if not already) the
                            scaler now. Otherwise, the scaler will be
                            initialized at access time.
        """
        if compute_now:
            scaler = get_scaler(scaler=scaler,
                                ignore_less_eq=ignore_less_eq).fit(self.image)
            self._scaler = scaler
        else:
            self._scaler = (scaler, ignore_less_eq)

    def apply_scaler(self):
        """
        Apply the stored scaler (channel-wise) to the stored image
        Note: in-place opperation
        """
        self._image = self.scaler.transform(self.image)

    @property
    def interpolator(self):
        """
        Return a interpolator for this object.
        If not already set, will initialize an interpolator and store it
        """
        if not self._interpolator:
            self.set_interpolator_with_current()
        return self._interpolator

    @interpolator.setter
    def interpolator(self, _):
        raise ReadOnlyAttributeError("Cannot set the interpolator property. "
                                     "Call self.set_interpolator_with_current"
                                     " to set a new interpolator object.")

    @property
    def is_loaded(self):
        return self._image is not None

    def load(self):
        """
        Forces a load on any set attributes in
            (self.image, self.labels,
            self.interpolator, self.bg_value, self.scaler)
        """
        # OBS: Invoking the getter method on these properties invokes a load
        # No need to store the results!
        _ = [_ for _ in (self.image, self.labels, self.bg_value,
                         self.scaler, self.interpolator)]

    @contextmanager
    def loaded_in_context(self):
        """
        Context manager which keeps this ImagePair loaded in the context
        and unloads it at exit.
        """
        try:
            yield self.load()
        finally:
            self.unload()

    def unload(self, unload_scaler=False):
        """
        Unloads the ImagePair by un-assigning the image and labels attributes
        Also clears the currently set interpolator object, as this references
        the image and label arrays and thus prevents GC.

        Args:
            unload_scaler: boolean indicating whether or not to also clear the
                           currently set scaler object. Not clearing this may
                           be useful if the ImagePair is loaded/unloaded often
                           (in a data queue for instance), as the scaler
                           parameters may take time to estimate, but will be
                           the same across data loads/un-loads of the same
                           image.
        """
        self._image = None
        self._labels = None
        self._interpolator = None
        if unload_scaler:
            self._scaler = None

    def get_interpolator_with_current(self):
        """
        Initialize and return a ViewInterpolator object with references to this
        ImagePair's image and labels arrays. The interpolator performs linear
        and nearest interpolation of the two respectively on an arbitrary grid
        of positions in the real-scanner space (defined by the image affine).

        OBS: Does not perform interpolation in voxel space

        Returns:
            A ViewInterpolator object for the ImagePair image and labels arrays
        """
        if not self.bg_value:
            raise RuntimeError("Cannot get interpolator without a set "
                               "background value. Call self.set_bg_value "
                               "first.")
        # Standardize the bg_value, handles None and False differently from 0
        # or 0.0 - for None and False the 1st percentile of self.image is used
        if not self.predict_mode:
            labels = self.labels
        else:
            labels = None
        return ViewInterpolator(self.image, labels,
                                bg_value=self.bg_value,
                                bg_class=self.bg_class,
                                affine=self.affine)

    def set_interpolator_with_current(self):
        """
        Set an interpolator object in the ImagePair
        See get_interpolator_with_current
        """
        self._interpolator = self.get_interpolator_with_current()

    def standardize_bg_val(self, bg_value):
        """
        Standardize the bg_value, handles None and False differently from 0
        or 0.0 - for None and False the 1st percentile of self.image is used

        Args:
            bg_value: The non-standardized background value. Should be an int,
                      float, None or False

        Returns:
            A list of float(s)/int(s) image background value pr. channel
        """
        if not isinstance(bg_value, (list, tuple, np.ndarray)):
            bg_value = [bg_value]
        standardized_bg_values = []
        for value in bg_value:
            value = value if (bg_value is not None and bg_value is not False) else "1pct"
            standardized_bg_values.append(value)
        if len(standardized_bg_values) == 1 and self.n_channels != 1:
            standardized_bg_values *= self.n_channels
        return standardized_bg_values

    def _bg_pct_string_to_value(self, bg_pct_str):
        """
        TODO

        Args:
            bg_pct_str: TODO
        """
        bg_value = []
        for i, bg_str in enumerate(bg_pct_str):
            # assuming '<number>pct' format
            bg_pct = int(bg_str.lower().replace(" ", "").split("pct")[0])
            bg_value.append(np.percentile(self.image[..., i], bg_pct))
        self.logger.warn("Image %s: Using %s percentile BG value of %s" % (
            self.identifier, bg_pct_str, bg_value
        ),  no_print=True)
        return bg_value
