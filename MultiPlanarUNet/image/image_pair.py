"""
Mathias Perslev
MSc Bioinformatics

University of Copenhagen
November 2017
"""

import os
import numpy as np
import nibabel as nib

from MultiPlanarUNet.preprocessing import get_scaler
from MultiPlanarUNet.logging import ScreenLogger
from MultiPlanarUNet.interpolation.sample_grid import get_real_image_size, get_pix_dim
from MultiPlanarUNet.interpolation.view_interpolator import ViewInterpolator

# Errors
from MultiPlanarUNet.errors.image_errors import (NoLabelFileError,
                                                 ReadOnlyAttributeError)

# w2 negative threshold is too strict for this data set
nib.Nifti1Header.quaternion_threshold = -1e-6


class ImagePair(object):
    """
    ImagePair
    ---
    Represents one data point of 1 .nii image file and corresponding labels
    """
    def __init__(self, img_path, labels_path=None, sample_weight=1.0,
                 logger=None, im_dtype=np.float32, lab_dtype=np.uint8):
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
            logger:        A MultiPlanarUNet logger object writing to screen and
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
        self.id = self._get_and_validate_id()

        # Set variables to store loaded image and label information
        self.image_obj = nib.load(self.image_path)
        self.labels_obj = None
        if not self.predict_mode:
            self.labels_obj = nib.load(self.labels_path)

        # Stores the data of the image and labels objects
        self._image = None
        self._labels = None
        self.scaler = None

        # ViewInterpolator object initialized with set_interpolator_object
        self.interpolator = None

        # May be set by various functions to keep track of state of this image
        self.load_state = None

        # Data types
        self.im_dtype = im_dtype
        self.lab_dtype = lab_dtype

    def __str__(self):
        return "<ImagePair object, identifier: %s>" % self.id

    def __repr__(self):
        return self.__str__()

    def log_image(self, print_calling_method=False):
        """
        Log basic stats for this ImagePair.
        """
        self.logger("%s\n"
                    "--- shape:      %s\n"
                    "--- real shape: %s\n"
                    "--- pixdim:     %s" % (
                        self.id, self.shape,
                        np.round(get_real_image_size(self), 3),
                        np.round(get_pix_dim(self), 3)
                    ), print_calling_method=print_calling_method)

    @property
    def affine(self):
        return self.image_obj.affine

    @affine.setter
    def affine(self, affine):
        raise ReadOnlyAttributeError("Manually setting the affine attribute "
                                     "is not allowed. Initialize a new "
                                     "ImagePair object.")

    @property
    def header(self):
        return self.image_obj.header

    @header.setter
    def header(self, header):
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
    def image(self, image):
        raise ReadOnlyAttributeError("Manually setting the image attribute "
                                     "is not allowed. Initialize a new "
                                     "ImagePair object.")

    @property
    def labels(self):
        """ Like self.image """
        if self._labels is None:
            try:
                self._labels = self.labels_obj.get_fdata(caching="unchanged").astype(self.lab_dtype)
            except AttributeError as e:
                raise NoLabelFileError("No label file attached to "
                                       "this ImagePair object.") from e
        return self._labels

    @labels.setter
    def labels(self, labels):
        raise ReadOnlyAttributeError("Manually setting the labels "
                                     "attribute is not allowed. "
                                     "Initialize a new ImagePair object.")

    @staticmethod
    def _validate_path(path):
        if os.path.exists(path) and path.split(".")[-1] in ("nii", "mat", "gz"):
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

    def prepare_for_iso_live(self, bg_value, bg_class, scaler):
        """
        Utility method preparing the ImagePair for usage in the iso_live
        interpolation mode (see MultiPlanarUNet.image.ImagePairLoader class).

        Performs the following operations:
            1) Loads the image and labels if not already loaded (transparent)
            2) Define proper background value
            3) Setting multi-channel scaler
            4) Setting interpolator object

        Args:
            bg_value: A value defining the space outside of the image region.
                      bg_value may be a number of string of the format
                      '[0-100]'pct specifying a percentile value to compute
                      across the image and use for bg_value
            bg_class: An interger defining the background class to assign to
                      pixels getting the 'bg_value' value.
            scaler:   String indicating which sklearn scaler class to use for
                      preprocessing of the image.
        """
        if isinstance(bg_value, str):
            # assuming '<number>pct' format
            bg_pct = int(bg_value.lower().replace(" ", "").split("pct")[0])
            bg_value = [np.percentile(self.image[..., i], bg_pct) for i in range(self.n_channels)]

            self.logger("OBS: Using %i percentile BG value of %s" % (
                bg_pct, bg_value
            ))

        # Apply scaling
        if self.scaler is None:
            self.set_scaler(scaler, ignore_less_eq=bg_value)

        # Set interpolator object
        self.set_interpolator_with_current(bg_value=bg_value,
                                           bg_class=bg_class)

    def unload(self, unload_scaler=False):
        """
        Unloads the ImagePair by un-assigning the image and labels attributes
        Also clears the currently set interpolator object, as this references
        the image and label arrays and thus might prevent GC.

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
        self.interpolator = None
        self.load_state = None
        if unload_scaler:
            self.scaler = None

    def _get_and_validate_id(self):
        """
        Validates if the image identifier and label identifier match.
        Returns the image identifier.
        """
        img_id = os.path.split(self.image_path)[-1].split(".")[0]
        if not self.predict_mode:
            labels_id = os.path.split(self.labels_path)[-1].split(".")[0]

            if img_id != labels_id:
                raise ValueError("Image id '%s' does not match labels id '%s'"
                                 % (img_id, labels_id))

        return img_id

    def get_interpolator_with_current(self, bg_value=None, bg_class=0):
        """
        Initialize and return a ViewInterpolator object with references to this
        ImagePair's image and labels arrays. The interpolator performs linear
        and nearest interpolation of the two respectively on an arbitrary grid
        of positions in the real-scanner space (defined by the image affine).

        OBS: Does not perform interpolation in voxel space

        Args:
            bg_value: A number value assigned to interpolated voxels outside
                      the image domain
            bg_class: An integer value assigned to the label map for voxels
                      outside the image volume

        Returns:
            A ViewInterpolator object for the ImagePair image and labels arrays
        """
        # Standardize the bg_value, handles None and False differently from 0
        # or 0.0 - for None and False the 1st percentile of self.image is used
        bg_value = self.standardize_bg_val(bg_value)
        if not self.predict_mode:
            labels = self.labels
        else:
            labels = None
        return ViewInterpolator(self.image, labels,
                                bg_value=bg_value,
                                bg_class=bg_class,
                                affine=self.affine)

    def set_interpolator_with_current(self, *args, **kwargs):
        """
        Set an interpolator object in the ImagePair
        See get_interpolator_with_current
        """
        self.interpolator = self.get_interpolator_with_current(*args, **kwargs)

    def set_scaler(self, scaler, ignore_less_eq=None):
        """
        Sets a scaler on the ImagePair fit to the stored image
        See MultiPlanarUNet.preprocessing.scaling
        """
        self.scaler = get_scaler(scaler=scaler,
                                 ignore_less_eq=ignore_less_eq).fit(self.image)

    def apply_scaler(self):
        """
        Apply the stored scaler (channel-wise) to the stored image
        Note: in-place opperation
        """
        self.image = self.scaler.transform(self.image)

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
        bg_value = bg_value if (bg_value is not None and bg_value is not False) \
                            else [np.percentile(self.image[..., i], 1) for i in range(self.n_channels)]
        return [bg_value] if not isinstance(bg_value,
                                            (list, tuple, np.ndarray)) else bg_value
