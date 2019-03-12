from MultiPlanarUNet.logging import ScreenLogger
import os


def init_model(build_hparams, logger=None):
    from MultiPlanarUNet import models
    logger = logger or ScreenLogger()

    # Build new model of the specified type
    cls_name = build_hparams["model_class_name"]
    logger("Creating new model of type '%s'" % cls_name)

    return models.__dict__[cls_name](logger=logger, **build_hparams)


def model_initializer(hparams, continue_training, base_path, logger=None):
    logger = logger or ScreenLogger()

    # Init model
    model = init_model(hparams["build"], logger)

    if continue_training:
        from MultiPlanarUNet.utils import get_last_model, get_lr_at_epoch, \
                                          clear_csv_after_epoch, get_last_epoch
        model_path, epoch = get_last_model(os.path.join(base_path, "model"))
        model.load_weights(model_path, by_name=True)
        csv_path = os.path.join(base_path, "logs", "training.csv")
        if epoch == 0:
            epoch = get_last_epoch(csv_path)
        else:
            clear_csv_after_epoch(epoch, csv_path)
        hparams["fit"]["init_epoch"] = epoch+1

        # Get the LR at the continued epoch
        lr, name = get_lr_at_epoch(epoch, os.path.join(base_path, "logs"))
        if lr:
            hparams["fit"]["optimizer_kwargs"][name] = lr

        logger("[NOTICE] Training continues from:\n"
               "Model: %s\n"
               "Epoch: %i\n"
               "LR:    %s" % (os.path.split(model_path)[-1], epoch, lr))
    else:
        hparams["fit"]["init_epoch"] = 0
    return model
