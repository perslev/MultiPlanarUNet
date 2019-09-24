from MultiPlanarUNet.sequences.isotrophic_live_view_sequence import IsotrophicLiveViewSequence
from MultiPlanarUNet.interpolation.sample_grid import sample_plane_at, get_bounding_sphere_real_radius
import numpy as np


class IsotrophicLiveViewSequence2D(IsotrophicLiveViewSequence):
    def __init__(self, image_pair_loader, views, no_log=False, **kwargs):

        super().__init__(image_pair_loader, **kwargs)
        self.views = views
        self.batch_shape = (self.batch_size, self.sample_dim, self.sample_dim,
                            self.n_classes)

        if not no_log:
            self.log()

    def log(self):
        self.logger("\nIs validation:               %s" % self.is_validation)
        self.logger("Using real space span:       %s" % self.real_space_span)
        self.logger("Using sample dim:            %s" % self.sample_dim)
        self.logger("Using real space sample res: %s" % (self.real_space_span/
                                                         self.sample_dim))
        self.logger("N fg slices:                 %s" % self.n_fg_slices)
        self.logger("Batch size:                  %s" % self.batch_size)
        self.logger("Force all FG:                %s" % self.force_all_fg)
        self.logger("Noise SD:                    %s" % self.noise_sd)
        self.logger("Augmenters:                  %s" % self.list_of_augmenters)

    def __len__(self):
        n_samples = self.sample_dim * len(self.images) * len(self.views)
        return int(np.ceil(n_samples / self.batch_size))

    def get_view_from(self, image_id, view, n_planes):
        image = [m for m in self.images if m.id == image_id][0]

        sample_res = self.real_space_span/(self.sample_dim-1)
        if n_planes == "by_radius":
            # Get sample sphere radius
            bounds = get_bounding_sphere_real_radius(image)
            n_planes = int(2 * bounds / sample_res)
        else:
            extra = 0
            if n_planes == "same":
                n_planes = self.sample_dim
            elif isinstance(n_planes, str) and n_planes[:5] == "same+":
                extra = int(n_planes.split("+")[-1])
                n_planes = self.sample_dim + extra
            bounds = (self.real_space_span+(extra*sample_res))/2

        # Prepare sample plane arguments
        kwargs = {
            "norm_vector": view,
            "sample_dim": self.sample_dim,
            "real_space_span": self.real_space_span,
            "noise_sd": 0.,
            "test_mode": True
        }

        # Define offsets
        offsets = np.linspace(-bounds, bounds, n_planes)
        self.logger("Sampling %i planes from "
                    "offset %.3f to %.3f..." % (n_planes, offsets[0],
                                                offsets[-1]))

        # Prepare results arrays
        shape = (self.sample_dim, self.sample_dim, n_planes)
        Xs = np.empty(shape + (image.n_channels,), dtype=image.image.dtype)
        if not image.predict_mode:
            ys = np.empty(shape, dtype=image.labels.dtype)
        else:
            ys = None

        # Prepare thread pool
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=7)

        def _do(offset, ind):
            im, lab, real_axis, inv_basis = self.sample_at(offset,
                                                           image.interpolator,
                                                           image.scaler, kwargs)
            return im, lab, real_axis, inv_basis, ind

        # Perform interpolation
        inds = np.arange(offsets.shape[0])
        result = pool.map(_do, offsets, inds)

        i = 1
        for im, lab, real_axis, inv_basis, ind in result:
            print("   %i/%i" % (i, len(offsets+1)), end="\r", flush=True)
            i += 1

            # Add planes to volumes
            Xs[..., ind, :] = im
            if not image.predict_mode:
                ys[..., ind] = lab

        print('')
        return Xs, ys, (real_axis, real_axis, offsets), inv_basis

    def sample_at(self, offset, interpolator, scaler, kwargs):
        # Get plane mgrid
        grid, real_axis, inv_basis = sample_plane_at(offset_from_center=offset,
                                                     **kwargs)

        # Interpolate at grid points
        im, lab = interpolator(grid)

        # Normalize
        im = scaler.transform(im)

        return im, lab, real_axis, inv_basis

    def __getitem__(self, idx):
        """
        Used by keras.fit_generator to fetch mini-batches during training
        """
        # If multiprocessing, set unique seed for this particular process
        self.seed()

        # Store how many slices has fg so far
        has_fg = 0
        has_fg_vec = np.zeros_like(self.fg_classes)

        # Interpolate on a random index for each sample image to generate batch
        batch_x, batch_y, batch_w = [], [], []

        # Maximum number of sampling trails
        max_tries = self.batch_size * 15

        # Number of images to use in each batch. Number should be low enough
        # to not exhaust queue generator.
        N = 2 if self.image_pair_loader.queue else self.batch_size
        cuts = np.round(np.linspace(0, self.batch_size, N+1)[1:])

        scalers = []
        bg_values = []
        for i, image in enumerate(self.image_pair_loader.get_random(N=N)):
            tries = 0
            # Sample a batch from the image
            while len(batch_x) < cuts[i]:
                tries += 1

                # Randomly sample a slice from a random image and random view
                view = self.views[np.random.randint(0, len(self.views), 1)[0]]

                # Get sample sphere radius
                sphere_r_real = self.real_space_span // 2

                # Sample a position on the axis
                rd = np.random.uniform(-sphere_r_real, sphere_r_real, 1)[0]

                # Get grid and interpolate
                mgrid = sample_plane_at(view,
                                        sample_dim=self.sample_dim,
                                        real_space_span=self.real_space_span,
                                        offset_from_center=rd,
                                        noise_sd=self.noise_sd,
                                        test_mode=False)

                # Get interpolated labels
                lab = image.interpolator.intrp_labels(mgrid)

                if self.force_all_fg and tries < max_tries:
                    valid, has_fg_vec = self.validate_lab_vec(lab,
                                                              has_fg_vec,
                                                              len(batch_y))
                    if not valid:
                        tries += 1
                        continue

                valid_lab, fg_change = self.validate_lab(lab, has_fg, len(batch_y), debug=False)
                if valid_lab or tries > max_tries:
                    # Get interpolated image
                    im = image.interpolator.intrp_image(mgrid)
                    im_bg_val = image.interpolator.bg_value
                    if tries > max_tries or self.is_valid_im(im, im_bg_val):
                        # Update foreground counter
                        has_fg += fg_change

                        # Save scaler to normalize image later (after potential
                        # augmentation)
                        scalers.append(image.scaler)

                        # Save bg value if needed in potential augmenters
                        bg_values.append(im_bg_val)

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
