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
