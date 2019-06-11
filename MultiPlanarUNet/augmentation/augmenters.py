from .elastic_deformation import elastic_transform_2d, elastic_transform_3d
import numpy as np


class Augmenter(object):
    """
    Not yet implemented
    """
    def __call__(self, batch_x, batch_y, bg_values, batch_w=None):
        raise NotImplemented


class Elastic(Augmenter):
    """
    2D and 3D random elastic deformations augmenter base class

    Applies either Elastic2D or Elastic3D to every element of a batch of images
    """
    def __init__(self, alpha, sigma, apply_prob,
                 transformer_func, aug_weight=0.33):
        """
        Args:
            alpha: A number of tuple/list of two numbers specifying a range
                   of alpha values to sample from in each augmentation call
                   The alpha value determines the strength of the deformation
            sigma: A number of tuple/list of two numbers specifying a range
                   of sigma values to sample from in each augmentation call
                   The sigma value determines the smoothness of the deformation
            apply_prob: Apply the transformation only with some probability
                        Otherwise, return the image untransformed
            transformer_func: The deformation function, either Elastic2D or
                              Elastic3D
            aug_weight: If a list of weights of len(batch_x) elements is passed
                        the aug_weight will replace the passed weight at index
                        i if image i in batch_x is transformed.
                        This allows for assigning a different weight to images
                        that were augmented versus real images.
        """
        # Initialize base
        super().__init__()

        if isinstance(alpha, (list, tuple)):
            if len(alpha) != 2:
                raise ValueError("Invalid list of alphas specified '%s'. "
                                 "Should be 2 numbers." % alpha)
            if alpha[1] <= alpha[0]:
                raise ValueError("alpha upper is smaller than sigma lower (%s)" % alpha)
        if isinstance(sigma, (list, tuple)):
            if len(sigma) != 2:
                raise ValueError("Invalid list of sigmas specified '%s'. "
                                 "Should be 2 numbers." % sigma)
            if sigma[1] <= sigma[0]:
                raise ValueError("Sigma upper is smaller than sigma lower (%s)" % sigma)
        if apply_prob > 1 or apply_prob < 0:
            raise ValueError("Apply probability is invalid with value %3.f" % apply_prob)

        self._alpha = alpha
        self._sigma = sigma
        self.apply_prob = apply_prob
        self.trans_func = transformer_func
        self.weight = aug_weight
        self.__name__ = "Elastic"

    @property
    def alpha(self):
        """
        Return a randomly sampled alpha value in the range [alpha[0], alpha[1]]
        or return the integer/float alpha if alpha is not a list
        """
        if isinstance(self._alpha, (list, tuple)):
            return np.random.uniform(self._alpha[0], self._alpha[1], 1)[0]
        else:
            return self._alpha

    @property
    def sigma(self):
        """
        Return a randomly sampled sigma value in the range [sigma[0], sigma[1]]
        or return the integer/float sigma if sigma is not a list
        """
        if isinstance(self._sigma, (list, tuple)):
            return np.random.uniform(self._sigma[0], self._sigma[1], 1)[0]
        else:
            return self._sigma

    def __call__(self, batch_x, batch_y, bg_values, batch_w=None):
        """
        Deform all images in a batch of images (using linear intrp) and
        corresponding labels (using nearest intrp)
        """
        # Only augment some of the images (determined by apply_prob)
        augment_mask = np.random.rand(len(batch_x)) <= self.apply_prob

        augmented_x, augmented_y = [], []
        for i, (augment, x, y, bg_vals) in enumerate(zip(augment_mask,
                                                         batch_x, batch_y,
                                                         bg_values)):
            if augment:
                x, y = self.trans_func(x, y, self.alpha, self.sigma, bg_vals)
                if batch_w is not None:
                    batch_w[i] = self.weight
            augmented_x.append(x)
            augmented_y.append(y)

        if batch_w is not None:
            return augmented_x, augmented_y, batch_w
        else:
            return augmented_x, augmented_y

    def __str__(self):
        return "%s(alpha=%s, sigma=%s, apply_prob=%.3f)" % (
            self.__name__, self._alpha, self._sigma, self.apply_prob
        )

    def __repr__(self):
        return str(self)


class Elastic2D(Elastic):
    """
    Applies random elastic deformations to 2D images.

    See docstring of Elastic (base class)
    """
    def __init__(self, alpha, sigma, apply_prob):
        """
        See docstring of Elastic (base class)
        """
        super().__init__(alpha, sigma, apply_prob,
                         transformer_func=elastic_transform_2d)
        self.__name__ = "Elastic2D"


class Elastic3D(Elastic):
    """
    Applies random elastic deformations to 3D images.

    See docstring of Elastic (base class)
    """
    def __init__(self, alpha, sigma, apply_prob):
        """
        See docstring of Elastic (base class)
        """
        super().__init__(alpha, sigma, apply_prob,
                         transformer_func=elastic_transform_3d)
        self.__name__ = "Elastic3D"

    def __str__(self):
        return "Elastic3D(alpha=%s, sigma=%s, apply_prob=%.3f)" % (
            self._alpha, self._sigma, self.apply_prob
        )
