from functools import wraps
from inspect import signature


def accepts(*types):
    def decorator(function):
        _types = types
        f_sig = signature(function)
        if "self" in f_sig.parameters:
            _types = (object,) + _types
        assert len(_types) == len(f_sig.parameters)

        @wraps(function)
        def wrapper(*args, **kwargs):
            for i, (o, t) in enumerate(zip(args, _types)):
                valid = False
                if isinstance(t, (tuple, list)):
                    for tt in t:
                        if isinstance(o, tt):
                            valid = True
                else:
                    if isinstance(o, t):
                        valid = True
                if not valid:
                    raise ValueError("Invalid input passed to '%s'. "
                                     "Expected type(s) %s (got %s)" % (function.__name__,
                                                                       t,
                                                                       type(o).__name__))
            return function(*args, **kwargs)
        return wrapper
    return decorator
