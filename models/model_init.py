from MultiViewUNet.logging import ScreenLogger
import os


def init_model(build_hparams, logger):
    from MultiViewUNet import models

    # Build new model of the specified type
    cls_name = build_hparams["model_class_name"]
    logger("Creating new model of type '%s'" % cls_name)

    return models.__dict__[cls_name](logger=logger, **build_hparams)


def model_initializer(hparams, continue_training, base_path, logger=None):
    logger = logger or ScreenLogger()

    # Init model
    model = init_model(hparams["build"], logger)

    if continue_training:
        from MultiViewUNet.utils import get_last_model
        model_path, epoch = get_last_model(os.path.join(base_path, "model"))
        model.load_weights(model_path, by_name=True)
        hparams["fit"]["init_epoch"] = epoch+1

        logger("[NOTICE] Training continues from:\n"
               "Model: %s\n"
               "Epoch: %i\n" % (os.path.split(model_path)[-1], epoch))
    else:
        hparams["fit"]["init_epoch"] = 0

    return model
