History
-------

0.1.0 (2018-12-17)
--------------------
* Project packaged for PIP

0.1.1 (2019-01-08)
--------------------
* Added multi-task learning functionality.

0.1.2 (2019-02-18)
--------------------
* Many smaller fixes and performance improvements. Also fixes a critical error
  that in some cases would cause the validation callback to only consider a
  subset of the predicted batch when computing validation metrics, which could
  make validation metrics noisy especially for large batch sizes.

0.1.3 (2019-02-20)
--------------------
* One-hot encoded targets (set with sparse=False in the fit section of
  hyperparameter file) are no longer supported. Setting this value no longer
  has any effect and may not be allowed in future versions.
* The Validation callback has been changed significantly and now computes both
  loss and any metrics specified in the hyperparamter file as performed on the
  training set to facility a more easy comparison. Note that as is the case on
  the training set, these computations are averaged batch-wise metrics.
  The CB still computes the epoch-wise pr-class and average precision,
  recall and dice.
* Default parameter files no longer have pre-specified metrics. Metrics (such
  as categorical accuracy, fg_precision, etc.) must be manually specified.

0.1.4 (2019-03-02)
------------------
* Minor changes over 0.1.3, including ability to set a pre-specified set of
  GPUs to cycle in mp cv_experiment

0.2.0 (2019-02-27)
------------------
* MultiChannelScaler now ignores values equal to or smaller than the 'bg_value'
  for each channel separately.
  This value is either set manually by the user and must be a list of values
  equal to the number of channels or a single value (that will be applied to
  all channels). If bg_value='1pct' is specified (default for most models), or
  any other percentage following this specification ('2pct' for 2 percent etc),
  the 1st percentile will be computed for each channel individually and used
  to define the background value for that channel.
* ViewInterpolator similarly now accepts a channel-wise background value
  specification, so that bg_value=[0, 0.1, 1] will cause out-of-bounds
  interpolation to generate a pixel of value [0, 0.1, 1] for a 3-channel image.
  Before, all channels would share a single, global background value (this
  effect is still obtained if bg_value is set to a single integer or float).
* Note that these changes may affect performance negatively if using the v 0.2
  software on projects with models trained with version <0.2.0. Users will be
  warned if trying to do so.
* v0.2.0 now checks which MultiPlanarUNet version was used to create/run code
  in a give project. Using a new version of the software on an older project
  folder is no longer allowed. This behaviour may however be overwritten
  manually setting the __VERSION__ variable to the current software version in
  the hyperparamter file of the project (not recommended, instead, downgrade
  to a previous version by running 'git checkout v<VERSION>' inside the
  MultiPlanarUNet code folder).

0.2.1 (2019-04-16)
------------------
* Various smaller changes and bug-fixes across the code base. Thread pools are now
  generally limited to a maximum of 7 threads; The cv_experiment script now correctly
  handles using the 'mp' script entry point in the 'script' file (before full paths 
  to the given script had to be passed to the python interpreter)

0.2.2 (2019-06-17)
------------------
* Process has started to re-factor/re-write scripts in the bin module to make
  them clearer, remove deprecated command-line arguments etc.
* Evaluation results as stored in .csv files are now always saved and loaded
  with an index column as the first column of the file.

0.2.3 (2019-11-19)
------------------
* Simplified the functionality of the Validation callback so that it now only
  computes the F1/Dice, precision and recall scores. I.e. the callback no longer
  computes validation metrics. This choice was made to increase stability between
  TensorFlow versions. The callback should work for most versions of TF now, incl.
  TF 2.0. Future versions of MultiPlanarUNet will re-introduce validation metrics
  in a TF 2.0 only setting.
* Various smaller changes across the code

0.2.4 (2020-02-07)
------------------
* Package was updated to comply with the TensorFlow >= 2.0 API.
* Package was renamed from 'MultiPlanarUNet' to 'mpunet'. This affects imports as well 
  as installs from PyPi (i.e. 'pip install mpunet' now), but not the GitHub repo.
* Now requires the 'psutil' and 'tensorflow-addons' packages.
* Implements a temporary fix to the issue raised at https://github.com/perslev/MultiPlanarUNet/issues/8
* Fixed a number of smaller bugs

0.2.5 (2020-02-21)
------------------
* Implements a fix to high memory usage reported during training on some systems
* Now uses tf.distribution for multi-GPU training and prediction
* Custom loss functions should now be wrapped by tf.python.keras.losses.LossFunctionWrapper. I.e. any loss function must be 
  a class which accepts a tf.keras.losses.Reduction parameter and potentially other parameters and returns the compiled loss function.
    * Consequently, when setting a loss function for MultiPlanarUNet training in train_hparams.yaml one must specify the factory 
      class verysion of the loss. E.g. for 'sparse_categorical_crossentropy' one must now specify 'SparseCategoricalCrossentropy' instead.
      The same naming convention applies to all custom losses.
    * Arbitrary Parameters may now be passed to a loss function in the 'loss_kwargs' entry in train_hparams.yaml
* Some (deprecated) custom loss functions have been removed.

0.2.6 (2020-07-28)
------------------
* Implemented ability to load training images from a queue of a given max size during training to reduce memory consumption (--max_images flag).
* Updated to work with TensorFlow 2.2

0.2.7 (2020-10-13)
------------------
* Minor changes to LearningCurve callback and plot_training_curves function to no longer plot training time in default learning curves.

0.2.9 (2021-01-07)
------------------
* Improved Windows compatability

0.2.10 (2021-05-03)
-------------------
* Reduced maximum time spent looking for valid batches which may speed up training on sparsely labelled images at the cost of using samples with few labels on average. Minor changes to logging.

0.2.11 (2021-05-03)
-------------------
* Fixed logging file path bug

0.2.12 (2021-05-26)
-------------------
* Removed unnecessary prediction code from the predict.py script which slowed down execution.
