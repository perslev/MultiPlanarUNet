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
-----------------
* Minor changes over 0.1.3, including ability to set a pre-specified set of GPUs to cycle in mp cv_experiment
