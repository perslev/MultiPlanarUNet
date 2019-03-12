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
