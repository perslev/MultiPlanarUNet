import os

from tensorflow.keras.callbacks import Callback
from MultiPlanarUNet.logging.default_logger import ScreenLogger
from MultiPlanarUNet.utils.plotting import plot_all_training_curves


class LearningCurve(Callback):
    """
    On epoch end this callback looks for all csv files matching the 'csv_regex'
    regex within the dir 'out_dir' and attempts to create a learning curve for
    each file that will be saved to 'out_dir'.

    Note: Failure to plot a learning curve based on a given csv file will
          is handled in the plot_all_training_curves function and will not
          cause the LearningCurve callback to raise an exception.
    """
    def __init__(self, log_dir="logs", out_dir="logs", fname="curve.png",
                 csv_regex="*training.csv", logger=None):
        """
        Args:
            log_dir: Relative path from the
            out_dir:
            fname:
            csv_regex:
            logger:
        """
        super().__init__()
        out_dir = os.path.abspath(out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.csv_regex = os.path.join(os.path.abspath(log_dir), csv_regex)
        self.save_path = os.path.join(out_dir, fname)
        self.logger = logger or ScreenLogger()

    def on_epoch_end(self, epoch, logs={}):
        plot_all_training_curves(self.csv_regex,
                                 self.save_path,
                                 logy=True,
                                 raise_error=False,
                                 logger=self.logger)
