from mpunet.sequences import BaseSequence
from mpunet.logging import ScreenLogger
from mpunet.interpolation.linalg import mgrid_to_points
import numpy as np


def center_expand(im, target_dim, bg_value, random=True):
    if im.ndim == 4:
        cim = np.empty(shape=(target_dim, target_dim, target_dim, im.shape[-1]),
                       dtype=im.dtype)
    else:
        cim = np.empty(shape=(target_dim, target_dim, target_dim),
                       dtype=im.dtype)
    cim.fill(bg_value)

    # Calculate size difference and starting positions in new volume
    diff = np.asarray(cim.shape[:3]) - im.shape[:3]
    if random and np.any(diff):
        start = [np.random.randint(0, max(d, 1), 1)[0] for d in diff]
    else:
        start = diff//2

    cim[start[0]:start[0] + im.shape[0],
        start[1]:start[1] + im.shape[1],
        start[2]:start[2] + im.shape[2]] = im

    return cim


class PatchSequence3D(BaseSequence):
    def __init__(self, image_pair_loader, dim, n_classes, batch_size, is_validation=False,
                 label_crop=None, fg_batch_fraction=0.33, logger=None, bg_val=0.,
                 no_log=False, **kwargs):
        super().__init__()

        # Set logger or default print
        self.logger = logger or ScreenLogger()
        self.image_pair_loader = image_pair_loader

        # Box dimension and image dim
        self.dim = dim

        # Various attributes
        self.n_classes = n_classes
        self.label_crop = label_crop
        self.is_validation = is_validation
        self.batch_size = batch_size
        self.bg_value = bg_val

        # How many foreground slices should be in each batch?
        self.fg_batch_fraction = fg_batch_fraction

        # Foreground label settings
        self.fg_classes = np.arange(1, self.n_classes)
        if self.fg_classes.shape[0] == 0:
            self.fg_classes = 1

        if not is_validation and not no_log:
            self.log()

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

    @property
    def n_samples(self):
        # In order to not load all the images, we manually specify the epoch
        # length in trainer.py
        return np.inf

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 0:
            raise ValueError("Invalid batch size of %i" % value)
        self._batch_size = value

    @property
    def n_fg_slices(self):
        if self.is_validation:
            return self.batch_size
        else:
            return int(np.ceil(self.batch_size * self.fg_batch_fraction))

    def get_N_random_patches_from(self, image, N):
        if N > 0:
            # Sample N patches from X
            X = image.image
            for i in range(N):
                xc, yc, zc = self.get_random_box_coords(X)
                patch = X[xc:xc + self.dim, yc:yc + self.dim, zc:zc + self.dim]
                yield image.scaler.transform(patch), (xc, yc, zc)
        else:
            return []

    def get_base_patches(self, image):
        X = image.image

        # Calculate positions
        sample_space = np.asarray([max(i, self.dim) for i in image.shape[:3]])
        d = (sample_space - self.dim)
        min_cov = [np.ceil(sample_space[i]/self.dim).astype(np.int) for i in range(3)]
        ds = [np.linspace(0, d[i], min_cov[i], dtype=np.int) for i in range(3)]

        # Get placement coordinate points
        placements = mgrid_to_points(np.meshgrid(*tuple(ds)))

        for p in placements:
            yield image.scaler.transform(X[p[0]:p[0]+self.dim,
                                         p[1]:p[1]+self.dim,
                                         p[2]:p[2]+self.dim]), p

    def get_patches_from(self, image, n_extra=0):
        for num, (p, coords) in enumerate(self.get_base_patches(image)):
            yield p, coords, "   Predicting on base patches (%i)" % (num+1)
        print("")
        for num, (p, coords) in enumerate(self.get_N_random_patches_from(image, n_extra)):
            yield p, coords, "   Predicting on extra patches (%i)" % (num+1)

    def validate_lab(self, lab, has_fg, cur_batch_size):
        valid = np.any(np.isin(self.fg_classes, lab))
        if valid:
            return valid, has_fg+1
        elif (self.n_fg_slices - has_fg) < (self.batch_size - cur_batch_size):
            # No FG, but there are still enough random slices left to fill the
            # minimum requirement
            return True, has_fg
        else:
            # No FG, but there is not enough random slices left to fill the
            # minimum requirement. Discard the slice and sample again.
            return False, has_fg

    def get_random_box_coords(self, im):
        dim = [max(0, s-self.dim) for s in im.shape[:3]]
        cords = np.round((dim * np.random.rand(3)).astype(np.uint16))
        return cords

    def get_box_coords(self, im):
        """
        Overwritten in SlidingPatchSequence3D to provide deterministic sampling
        """
        return self.get_random_box_coords(im)

    def __getitem__(self, idx, image_id=None):
        """
        Used by keras.fit_generator to fetch mini-batches during training
        """
        # If multiprocessing, set unique seed for this particular process
        self.seed()

        # Store how many slices has fg so far
        has_fg = 0

        # Interpolate on a random index for each sample image to generate batch
        batch_x, batch_y, batch_w = [], [], []

        with self.image_pair_loader.get_random() as image:
            while len(batch_x) < self.batch_size:

                # Fetch image, labels and weights
                X, y, w = image.image, image.labels, image.sample_weight

                # Sample a random box in the volume
                xc, yc, zc = self.get_box_coords(X)

                # Slice volume
                im = X[xc:xc+self.dim, yc:yc+self.dim, zc:zc+self.dim]
                lab = y[xc:xc+self.dim, yc:yc+self.dim, zc:zc+self.dim]

                # Make sure the box is of sufficient size
                im = center_expand(im, self.dim, self.bg_value, random=True)
                lab = center_expand(lab, self.dim, self.bg_value, random=True)

                # Validate label volume
                valid, has_fg = self.validate_lab(lab, has_fg, len(batch_y))
                if valid:
                    # Normalize image
                    im = image.scaler.transform(im)

                    batch_x.append(im)
                    batch_y.append(lab)
                    batch_w.append(w)

        # Reshape X (and one-hot encode y)
        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)
        batch_y = batch_y.reshape(batch_y.shape + (1,))

        return batch_x, batch_y, np.asarray(batch_w)

    def log(self):
        self.logger("Sequence Generator: %s" % self.__class__.__name__)
        self.logger("Box dimensions:     %s" % self.dim)
        self.logger("Batch size:         %s" % self.batch_size)
        self.logger("N fg slices/batch:  %s" % self.n_fg_slices)
