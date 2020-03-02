

class ScreenLogger(object):
    """
    Minimal wrapper class around the built-in print function replicating some
    functionality of the mpunet Logger class so that this class can be
    used in a similar fashion while only printing to screen and not a log file
    """
    def __init__(self, print_to_screen=True):
        self.print_to_screen = print_to_screen

    def __call__(self,
                 *args,
                 print_to_screen=None,
                 print_calling_method=None,
                 no_print=None,
                 **kwargs):
        if self.print_to_screen and not no_print:
            print(*args, **kwargs)

    def warn(self, *args, **kwargs):
        self.__call__("[WARNING]", *args, **kwargs)

    def __enter__(self):
        return

    def __exit__(self, *args):
        return
