from tensorflow.keras.callbacks import ModelCheckpoint
import warnings
import os


class ModelCheckPointClean(ModelCheckpoint):
    """
    Fixes bug in ModelCheckPoint that will fail to overwrite model/weight files
    of which the name changes as fitting progresses (epoch in name etc.)

    Overwrites the on_epoch_end method
    """
    def __init__(self, org_model=None, *args, **kwargs):
        ModelCheckpoint.__init__(self, *args, **kwargs)
        self.org_model = org_model
        self.last_file = None

    @property
    def __model(self):
        if self.org_model:
            return self.org_model
        else:
            return self.model

    def on_epoch_end(self, epoch, logs=None):
        print()
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)

            # Make root folder if not existing
            folder = os.path.split(os.path.abspath(filepath))[0]
            if not os.path.exists(folder):
                os.mkdir(folder)

            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.last_file:
                            # Make sure we remove the file even with changing
                            # filename over fitting
                            os.remove(self.last_file)
                        self.last_file = filepath
                        if self.save_weights_only:
                            self.__model.save_weights(filepath, overwrite=True)
                        else:
                            self.__model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.__model.save_weights(filepath, overwrite=True)
                else:
                    self.__model.save(filepath, overwrite=True)
