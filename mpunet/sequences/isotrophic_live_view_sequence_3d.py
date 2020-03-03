from mpunet.sequences.isotrophic_live_view_sequence import IsotrophicLiveViewSequence
from mpunet.interpolation.sample_grid import sample_box, sample_box_at
from mpunet.interpolation.linalg import mgrid_to_points
import numpy as np


class IsotrophicLiveViewSequence3D(IsotrophicLiveViewSequence):
    def __init__(self, image_pair_queue, real_box_dim, no_log=False, **kwargs):
        super().__init__(image_pair_queue, **kwargs)

        self.real_box_dim = real_box_dim
        self.batch_shape = (self.batch_size, self.sample_dim, self.sample_dim,
                            self.sample_dim, self.n_classes)

        if not no_log:
            self.log()

    def log(self):
        self.logger("Using sample dim:            %s" % self.sample_dim)
        self.logger("Using box real dim:          %s" % self.real_box_dim)
        self.logger("Using real space sample res: %s" % (self.real_box_dim/
                                                         self.sample_dim))
        self.logger("N fg slices:                 %s" % self.n_fg_slices)
        self.logger("Batch size:                  %s" % self.batch_size)
        self.logger("Force all FG:                %s" % self.force_all_fg)

    @staticmethod
    def _intrp_and_norm(image, grid, intrp_lab):
        # Interpolate
        im = image.interpolator.intrp_image(grid)

        # Normalize
        im = image.scaler.transform(im)

        lab = None
        if intrp_lab:
            lab = image.interpolator.intrp_labels(grid)

        return im, lab

    def get_base_patches_from(self, image, return_y=False):
        real_dims = image.real_shape

        # Calculate positions
        sample_space = np.asarray([max(i, self.real_box_dim) for i in real_dims])
        d = (sample_space - self.real_box_dim)
        min_cov = [np.ceil(sample_space[i]/self.real_box_dim).astype(np.int) for i in range(3)]
        ds = [np.linspace(0, d[i], min_cov[i]) - sample_space[i]/2 for i in range(3)]

        # Get placement coordinate points
        placements = mgrid_to_points(np.meshgrid(*tuple(ds)))

        for p in placements:
            grid, axes, inv_mat = sample_box_at(real_placement=p,
                                                sample_dim=self.sample_dim,
                                                real_box_dim=self.real_box_dim,
                                                noise_sd=0.0,
                                                test_mode=True)

            im, lab = self._intrp_and_norm(image, grid, return_y)

            if return_y:
                yield im, lab, grid, axes, inv_mat, len(placements)
            else:
                yield im, grid, axes, inv_mat, len(placements)

    def get_N_random_patches_from(self, image, N, return_y=False):
        if N > 0:
            # Sample N patches from X
            for i in range(N):
                # Get grid and interpolate
                grid, axes, inv_mat = sample_box(sample_dim=self.sample_dim,
                                                 real_box_dim=self.real_box_dim,
                                                 real_dims=image.real_shape,
                                                 noise_sd=self.noise_sd,
                                                 test_mode=True)

                im, lab = self._intrp_and_norm(image, grid, return_y)

                if return_y:
                    yield im, lab, grid, axes, inv_mat
                else:
                    yield im, grid, axes, inv_mat
        else:
            return []

    def _get_valid_box_from(self, image, max_tries, has_fg_vec, has_fg_count, cur_bs):
        """
        TODO
        """
        tries = 0
        while tries < max_tries:
            # Sample a batch from the image
            tries += 1

            # Get grid and interpolate
            mgrid = sample_box(sample_dim=self.sample_dim,
                               real_box_dim=self.real_box_dim,
                               real_dims=image.real_shape,
                               noise_sd=self.noise_sd)

            # Get interpolated labels
            lab = image.interpolator.intrp_labels(mgrid)
            valid_lab, fg_change = self.validate_lab(lab, has_fg_count, cur_bs)

            if self.force_all_fg and tries < max_tries:
                valid, has_fg_vec = self.validate_lab_vec(lab, has_fg_vec, cur_bs)
                if not valid:
                    continue

            if valid_lab or tries == max_tries:
                # Get interpolated image
                im = image.interpolator.intrp_image(mgrid)
                im_bg_val = image.interpolator.bg_value
                if tries == max_tries or self.is_valid_im(im, im_bg_val):
                    # Accept box, update foreground counter
                    has_fg_vec += fg_change
                    return im, lab, has_fg_count

    def __getitem__(self, idx):
        """
        Used by keras.fit_generator to fetch mini-batches during training
        """
        # If multiprocessing, set unique seed for this particular process
        self.seed()

        # Store how many slices has fg so far
        has_fg_count = 0
        has_fg_vec = np.zeros_like(self.fg_classes)

        # Interpolate on a random index for each sample image to generate batch
        batch_x, batch_y, batch_w = [], [], []

        # Get a random image
        max_tries = self.batch_size * 10

        scalers = []
        bg_values = []
        for _ in range(self.batch_size):
            with self.image_pair_queue.get_random_image() as image:
                im, lab, has_fg_count = self._get_valid_box_from(
                    image=image,
                    max_tries=max_tries,
                    has_fg_vec=has_fg_vec,
                    has_fg_count=has_fg_count,
                    cur_bs=len(batch_y)
                )

                # Save scaler to normalize image later (after potential aug)
                scalers.append(image.scaler)

                # Save bg value if needed in potential augmenters
                bg_values.append(image.interpolator.bg_value)

                # Add to batches
                batch_x.append(im)
                batch_y.append(lab)
                batch_w.append(image.sample_weight)

        # Normalize images
        batch_x = self.scale(batch_x, scalers)

        # Apply augmentation if specified
        batch_x, batch_y, batch_w = self.augment(batch_x, batch_y,
                                                 batch_w, bg_values)

        # Reshape, one-hot encode etc.
        batch_x, batch_y, batch_w = self.prepare_batches(batch_x,
                                                         batch_y,
                                                         batch_w)

        assert len(batch_x) == self.batch_size
        return batch_x, batch_y, batch_w
