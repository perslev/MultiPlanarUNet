
class NotSparseError(AttributeError): pass


def raise_non_sparse_metric_or_loss_error():
    sparse_err = "mpunet 0.1.3 or higher requires integer targets" \
                 " as opposed to one-hot encoded targets. All metrics and " \
                 "loss functions should be named 'sparse_[org_name]' to " \
                 "reflect this change in accordance with the naming convention" \
                 " of TensorFlow.keras."
    raise NotSparseError(sparse_err)
