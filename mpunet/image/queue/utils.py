from mpunet.image.queue import (LoadingPool, LimitationQueue,
                                LazyQueue, EagerQueue)


def validate_queue_type(queue_type, dataset, max_loaded, logger):
    if queue_type is LimitationQueue and (max_loaded is None or
                                          len(dataset) <= max_loaded):
        logger.warn("Falling back to 'Eager' queue for dataset {} due "
                    "to 'max_loaded' value of {} which is either 'None' or "
                    "larger than or equal to the total number of images ({}) "
                    "in the dataset.".format(dataset, max_loaded, len(dataset)))
        return EagerQueue
    return queue_type


def get_data_queues(train_dataset,
                    val_dataset,
                    train_queue_type,
                    val_queue_type,
                    max_loaded,
                    num_access_before_reload,
                    logger):
    """
    TODO.

    Returns:

    """
    map_ = {'eager': EagerQueue,
            'lazy': LazyQueue,
            'limitation': LimitationQueue}
    train_queue = validate_queue_type(map_[train_queue_type.lower()],
                                      train_dataset, max_loaded, logger)
    if val_queue_type:
        val_queue = validate_queue_type(map_[val_queue_type.lower()],
                                        val_dataset, max_loaded, logger)
    else:
        val_queue = None

    # If validation queue, get a loader pool object, shared across datasets
    if train_queue is LimitationQueue or val_queue is LimitationQueue:
        loading_pool = LoadingPool(n_threads=3,
                                   max_queue_size=max_loaded or None,
                                   logger=logger)
    else:
        loading_pool = None

    train_queue = train_queue(
            dataset=train_dataset,
            max_loaded=max_loaded,
            num_access_before_reload=num_access_before_reload,  # TODO
            preload_now=True,
            await_preload=True,
            loading_pool=loading_pool,
            logger=logger
    )
    if val_dataset:
        val_queue = val_queue(
            dataset=val_dataset,
            max_loaded=max_loaded,
            num_access_before_reload=num_access_before_reload,  # TODO
            preload_now=True,
            await_preload=False,
            loading_pool=loading_pool,
            logger=logger
        )
    else:
        val_queue = None
    return train_queue, val_queue
