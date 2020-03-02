from mpunet.logging.default_logger import ScreenLogger
from mpunet.image.queue import LazyQueue, BaseQueue


def get_sequence(data_queue,
                 is_validation,
                 logger=None,
                 augmenters=None,
                 **seq_kwargs):
    """
    Prepares a MultiPlanar.sequence object for generating batches of data from
    a set of images contained in a mpunet.image.queue typed object.

    These generator-like objects pull data from ImagePairs during
    training as needed. The sequences are specific to the model type (for
    instance 2D and 3D models have separate sequence classes) and may
    differ in the interpolation schema as well (voxel vs. iso scanner space
    coordinates for instance).

    Args:
        data_queue:    mpunet.image.queue data queue type object
        is_validation: Boolean, is this a validation sequence? (otherwise
                       training)
        logger:        TODO
        augmenters:    TODO
        **seq_kwargs:  Additional arguments passed to the Sequencer

    Raises:
        ValueError if intrp_style is not valid
    """
    logger = logger or ScreenLogger()
    # If data_queue in ImagePairDataset
    if not isinstance(data_queue, BaseQueue):
        # Wrap the passed ImagePairLoader in a LazyQueue
        data_queue = LazyQueue(data_queue)
    aug_list = []
    if not is_validation:
        # On the fly augmentation?
        list_of_aug_dicts = augmenters
        if list_of_aug_dicts:
            logger("Using on-the-fly augmenters:")
            from mpunet.augmentation import augmenters
            for aug in list_of_aug_dicts:
                aug_cls = augmenters.__dict__[aug["cls_name"]](**aug["kwargs"])
                aug_list.append(aug_cls)
                logger(aug_cls)

    if seq_kwargs['intrp_style'].lower() == "iso_live":
        # Isotrophic 2D plane sampling
        from mpunet.sequences import IsotrophicLiveViewSequence2D
        return IsotrophicLiveViewSequence2D(data_queue,
                                            is_validation=is_validation,
                                            list_of_augmenters=aug_list,
                                            logger=logger,
                                            **seq_kwargs)
    elif seq_kwargs['intrp_style'].lower() == "iso_live_3d":
        # Isotrophic 3D box sampling
        from mpunet.sequences import IsotrophicLiveViewSequence3D
        return IsotrophicLiveViewSequence3D(data_queue,
                                            is_validation=is_validation,
                                            list_of_augmenters=aug_list,
                                            logger=logger,
                                            **seq_kwargs)
    elif seq_kwargs['intrp_style'].lower() == "patches_3d":
        # Random selection of boxes
        from mpunet.sequences import PatchSequence3D
        return PatchSequence3D(data_queue,
                               is_validation=is_validation,
                               list_of_augmenters=aug_list, **seq_kwargs)
    elif seq_kwargs['intrp_style'].lower() == "sliding_patches_3d":
        # Sliding window selection of boxes
        from mpunet.sequences import SlidingPatchSequence3D
        return SlidingPatchSequence3D(data_queue,
                                      is_validation=is_validation,
                                      list_of_augmenters=aug_list,
                                      **seq_kwargs)
    else:
        raise ValueError("Invalid interpolator schema '%s' specified"
                         % seq_kwargs['intrp_style'])
