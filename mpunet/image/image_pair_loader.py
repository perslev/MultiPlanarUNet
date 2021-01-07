"""
Mathias Perslev
MSc Bioinformatics

University of Copenhagen
November 2017
"""

import glob
import os
import numpy as np

from pathlib import Path
from .image_pair import ImagePair
from mpunet.logging import ScreenLogger


class ImagePairLoader(object):
    """
    ImagePair data loader object
    Represents a collection of ImagePairs
    """
    def __init__(self,
                 base_dir="./",
                 img_subdir="images",
                 label_subdir="labels",
                 logger=None,
                 sample_weight=1.0,
                 bg_class=0,
                 predict_mode=False,
                 initialize_empty=False,
                 no_log=False,
                 identifier=None,
                 **kwargs):
        """
        Initializes the ImagePairLoader object from all .nii files in a folder
        or pair of folders if labels are also specified.

        If initialize_empty=False, the following actions are taken immediately
        on initialization:
            - All .nii/.nii.gz image files are found in base_dir/img_subdir
            - Unless predict_mode=True, finds all .nii/.nii.gz label files in
              base_dir/label_subdir
            - ImagePair objects are established for all images/image-label
              pairs. Not that since ImagePairs do not eagerly load data,
              the ImagePairLoader also does not immediately load data into mem

        If initialize_empty=True, the class is initialized but no images are
        loaded. Images can be manually added through the add_image and
        add_files methods.

        Args:
            base_dir:           A path to a directory storing the 'img_subdir'
                                and 'label_subdir' sub-folders
            img_subdir:         Name of sub-folder storing .nii images files
            label_subdir:       Name of sub-folder storing .nii labels files
            logger:             mpunet logger object
            sample_weight:      A float giving a global sample weight assigned
                                to all images loaded by the ImagePairLoader
            bg_class            Background class integer to pass to all
                                ImagePair objects. Usually int(0).
            predict_mode:       Boolean whether labels exist for the images.
                                If True, the labels are assumed stored in the
                                label_subdir with names identical to the images
            initialize_empty:   Boolean, if True do not load any images at init
                                This may be useful for manually assigning
                                individual image files to the object.
            no_log:             Boolean, whether to not log to screen/file
            identifier:         Optional name for the dataset
            **kwargs:           Other keywords arguments
        """
        self.logger = logger if logger is not None else ScreenLogger()

        # Set absolute paths to main folder, image folder and label folder
        self.data_dir = Path(base_dir).absolute()
        self.images_path = self.data_dir / img_subdir
        self.identifier = self.data_dir.name

        # Labels included?
        self.predict_mode = predict_mode or not label_subdir
        if not predict_mode:
            self.labels_path = self.data_dir / label_subdir
        else:
            self.labels_path = None

        # Load images unless initialize_empty is specified
        if not initialize_empty:
            # Get paths to all images
            self.image_paths = self.get_image_paths()

            if not predict_mode:
                # Get paths to labels if included
                self.label_paths = self.get_label_paths(img_subdir,
                                                        label_subdir)
            else:
                self.label_paths = None

            # Load all nii objects
            self.images = self.get_image_objects(sample_weight, bg_class)
        else:
            self.images = []

        if not initialize_empty and not self.image_paths:
            raise OSError("No image files found at %s." % self.images_path)
        if not initialize_empty and not predict_mode and not self.label_paths:
            raise OSError("No label files found at %s." % self.labels_path)

        self._id_to_image = self.get_id_to_images_dict()
        if not no_log:
            self._log()

    def __str__(self):
        return "ImagePairLoader(id={}, images={}, data_dir={})".format(
            self.identifier, len(self), self.data_dir
        )

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        return self.images[item]

    def __iter__(self):
        for im in self.images:
            yield im

    def __len__(self):
        return len(self.images)

    def _log(self):
        self.logger(str(self))
        self.logger("--- Image subdir: %s\n--- Label subdir: %s" % (self.images_path,
                                                                    self.labels_path))

    def load(self):
        """ Invokes the 'load' method on all ImagePairs """
        for image in self:
            image.load()

    def unload(self):
        """ Invokes the 'unload' method on all ImagePairs """
        for image in self:
            image.unload()

    @property
    def id_to_image(self):
        """
        Returns:
            A dictionary of image IDs pointing to image objects
        """
        return self._id_to_image

    def get_id_to_images_dict(self):
        return {image.identifier: image for image in self}

    @property
    def n_loaded(self):
        return sum([image.is_loaded for image in self.images])

    def get_by_id(self, image_id):
        """
        Get a specific ImagePair by its string identifier

        Args:
            image_id: String identifier of an ImagePair

        Returns:
            An ImagePair
        """
        return self.id_to_image[image_id]

    def get_random(self, N=1, unique=False):
        """
        Return N random images, with or without re-sampling

        Args:
            N:      Int, number of randomly sampled images to return
            unique: Bool, whether the sampled images should be all unique

        Returns:
            A list of ImagePair objects
        """
        returned = []
        while len(returned) < N:
            if self.queue:
                with self.queue.get() as image:
                    if unique and image in returned:
                        continue
                    else:
                        returned.append(image)
                        yield image
            else:
                image = self.images[np.random.randint(len(self))]
                if unique and image in returned:
                    continue
                else:
                    returned.append(image)
                    yield image

    def _get_paths_from_list_file(self, base_path, fname="LIST_OF_FILES.txt"):
        """
        Loads a set of paths pointing to .nii files in 'base_path'.
        This method is used in the rare cases that images are not directly
        stored in self.images_path or self.labels_path but those paths stores
        a file named 'fname' storing 1 absolute path per line pointing to the
        images to load.

        Args:
            base_path: A path to a folder
            fname:     The filename of the file at 'base_path' that stores the
                       paths to return

        Returns:
            A list of path strings
        """
        # Check if a file listing paths exists instead of actual files at the
        # image sub folder path
        list_file_path = base_path / fname
        images = []
        if os.path.exists(list_file_path):
            with open(list_file_path, "r") as in_f:
                for path in in_f:
                    path = path.strip()
                    if not path:
                        continue
                    images.append(path)
        else:
            raise OSError("File '%s' does not exist. Did you specify "
                          "the correct img_subdir?" % list_file_path)
        return images

    def get_image_paths(self):
        """
        Return a list of paths to all image files in the self.images_path folder

        Returns:
            A list of pathlib.Path
        """
        images = sorted(glob.glob(str(self.images_path / "*.nii*")))
        if not images:
            # Try to load from a file listing paths at the location
            # This is sometimes a format created by the cv_split.py script
            images = self._get_paths_from_list_file(self.images_path)
        return [Path(p) for p in images]

    def get_label_paths(self, img_subdir, label_subdir):
        """
        Return a list of paths to all label files in the self.labels_path folder
        The label paths are assumed to be identical to the image paths with the
        image subdir name replaced by the label subdir name.

        Args:
            img_subdir:   String, name of the image sub-folder
            label_subdir: String, name of the label sub-folder

        Returns:
            A list of pathlib.Path
        """
        if any([img_subdir not in str(p) for p in self.image_paths]):
            raise ValueError("Mismatch between image paths and specified "
                             "img_subdir. The subdir was not found in one or"
                             " more image paths - Do the paths in "
                             "LIST_OF_FILES.txt point to a subdir of name "
                             "'%s'?" % img_subdir)
        return [p.parent.parent / label_subdir / p.name for p in self.image_paths]

    def get_image_objects(self, sample_weight, bg_class):
        """
        Initialize all ImagePair objects from paths at self.image_paths and
        self.label_paths (if labels exist). Note that data is not loaded
        eagerly.

        Args:
            sample_weight: A float giving the weight to assign to the ImagePair
            bg_class:      Background (integer) class

        Returns:
            A list of initialized ImagePairs
        """
        image_objects = []
        if self.predict_mode:
            for img_path in self.image_paths:
                image = ImagePair(img_path,
                                  sample_weight=sample_weight,
                                  bg_class=bg_class,
                                  logger=self.logger)
                image_objects.append(image)
        else:
            for img_path, label_path in zip(self.image_paths, self.label_paths):
                image = ImagePair(img_path, label_path,
                                  sample_weight=sample_weight,
                                  bg_class=bg_class,
                                  logger=self.logger)
                image_objects.append(image)

        return image_objects

    def add_image(self, image_pair):
        """
        Add a single ImagePair object to the ImagePairLoader

        Args:
            image_pair: An ImagePair
        """
        self.images.append(image_pair)
        # Update ID dict
        self._id_to_image = self.get_id_to_images_dict()

    def add_images(self, image_pair_loader):
        """
        Add a set of ImagePair objects to the ImagePairLoader. Input can be
        either a different ImagePairLoader object or a list of ImagePairs.

        Args:
            image_pair_loader: ImagePairLoader or list of ImagePairs

        Returns:
            self
        """
        try:
            self.images += image_pair_loader.images
        except AttributeError:
            # Passed as list?
            self.images += list(image_pair_loader)
        # Update ID dict
        self._id_to_image = self.get_id_to_images_dict()
        return self

    def get_maximum_real_dim(self):
        """
        Returns the longest distance in mm covered by any axis across images
        of this ImagePairLoader.

        Returns:
            A float
        """
        from mpunet.interpolation.sample_grid import get_maximum_real_dim
        return np.max([get_maximum_real_dim(f.image_obj) for f in self])

    def set_scaler_and_bg_values(self, bg_value, scaler, compute_now=False):
        """
        Loads all images and prepares them for iso-live view interpolation
        training by performing the following operations on each:
            1) Loads the image and labels if not already loaded (transparent)
            2) Define proper background value
            3) Setting multi-channel scaler
            4) Setting interpolator object

        Args:
            bg_value:     See ImagePair.set_bg_value
            scaler:       See ImagePair.set_scaler
            compute_now:  TODO
        """
        # Run over volumes: scale, set interpolator, check for affine
        for image in self.id_to_image.values():
            image.set_bg_value(bg_value, compute_now=compute_now)
            image.set_scaler(scaler, compute_now=compute_now)
            image.log_image()
