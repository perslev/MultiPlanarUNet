from mpunet.logging import ScreenLogger
import os


def init_model(build_hparams, logger=None):
    from mpunet import models
    logger = logger or ScreenLogger()

    # Build new model of the specified type
    cls_name = build_hparams["model_class_name"]
    logger("Creating new model of type '%s'" % cls_name)

    return models.__dict__[cls_name](logger=logger, **build_hparams)


def model_initializer(hparams, continue_training, project_dir,
                      initialize_from=None, logger=None):
    logger = logger or ScreenLogger()

    # Init model
    model = init_model(hparams["build"], logger)

    if continue_training:
        if initialize_from:
            raise ValueError("Failed to initialize model with both "
                             "continue_training and initialize_from set.")
        from mpunet.utils import get_last_model, get_lr_at_epoch, \
                                          clear_csv_after_epoch, get_last_epoch
        model_path, epoch = get_last_model(os.path.join(project_dir, "model"))
        if model_path:
            model.load_weights(model_path, by_name=True)
            model_name = os.path.split(model_path)[-1]
        else:
            model_name = "<No model found>"
        csv_path = os.path.join(project_dir, "logs", "training.csv")
        if epoch == 0:
            epoch = get_last_epoch(csv_path)
        else:
            if epoch is None:
                epoch = 0
            clear_csv_after_epoch(epoch, csv_path)
        hparams["fit"]["init_epoch"] = epoch+1

        # Get the LR at the continued epoch
        lr, name = get_lr_at_epoch(epoch, os.path.join(project_dir, "logs"))
        if lr:
            hparams["fit"]["optimizer_kwargs"][name] = lr

        logger("[NOTICE] Training continues from:\n"
               "Model: %s\n"
               "Epoch: %i\n"
               "LR:    %s" % (model_name, epoch, lr))
    else:
        hparams["fit"]["init_epoch"] = 0
        if initialize_from:
            model.load_weights(initialize_from, by_name=True)
            logger("[NOTICE] Initializing parameters from:\n"
                   "{}".format(initialize_from))
    return model
