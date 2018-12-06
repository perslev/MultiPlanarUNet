import os
import inspect
from MultiPlanarUNet.utils.decorators import accepts


class Logger(object):
    def __init__(self, base_path, print_to_screen=True, active_file=None,
                 overwrite_existing=False, print_calling_method=True):
        self.base_path = os.path.abspath(base_path)
        self.path = os.path.join(self.base_path, "logs")
        self.overwrite_existing = overwrite_existing

        # Get built in print function
        # (if overwritten globally, Logger still maintains a reference to the
        # true print function)
        self.print_f = __builtins__["print"]

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        # Print options
        self.separator = "-" * 80
        self.print_to_screen = print_to_screen
        self.print_calling_method = print_calling_method

        # Set paths to log files
        self.log_files = {}
        self.currently_logging = {}
        self.active_log_file = active_file or "log"

    def __repr__(self):
        return "<MultiPlanarUNet.logging.Logger object>"

    def __str__(self):
        return "Logger(base_path=%s, print_to_screen=%s, " \
               "overwrite_existing=%s)" % (self.base_path,
                                           self.print_to_screen,
                                           self.overwrite_existing)

    def new_log_file(self, filename):
        file_path = os.path.join(self.path, "%s.txt" % filename)

        if os.path.exists(file_path):
            if self.overwrite_existing:
                os.remove(file_path)
            else:
                raise OSError("Logging path: %s already exists. "
                              "Initialize Logger(overwrite_existing=True) "
                              "to overwrite." % file_path)

        self.log_files[filename] = file_path
        self.currently_logging[filename] = None
        self.active_log_file = filename

        # Add reference to model folder in log
        ref = "Log for model in: %s" % self.base_path
        self._add_to_log(ref, no_print=True)

    @property
    def print_to_screen(self):
        return self._print_to_screen

    @print_to_screen.setter
    @accepts(bool)
    def print_to_screen(self, value):
        self._print_to_screen = value

    @property
    def print_calling_method(self):
        return self._print_calling_method

    @print_calling_method.setter
    @accepts(bool)
    def print_calling_method(self, value):
        self._print_calling_method = value

    @property
    def log(self):
        with open(self.log_files[self.active_log_file], "r") as log_f:
            return log_f.read()

    @property
    def active_log_file(self):
        return self._active_log_file

    @active_log_file.setter
    @accepts(str)
    def active_log_file(self, file_name):
        if file_name not in self.log_files:
            self.new_log_file(file_name)
        self._active_log_file = file_name

    def _add_to_log(self, *args, no_print=False, **kwargs):
        if self.print_to_screen and not no_print:
            self.print_f(*args, **kwargs)

        with open(self.log_files[self.active_log_file], "a") as log_file:
            self.print_f(*args, file=log_file, **kwargs)

    def _log(self, caller, print_calling_owerwrite=None, *args, **kwargs):
        if caller != self.currently_logging[self.active_log_file]:
            self.currently_logging[self.active_log_file] = caller
            if print_calling_owerwrite is not None:
                print_calling = print_calling_owerwrite
            else:
                print_calling = self.print_calling_method
            if print_calling:
                self._add_to_log("%s\n>>> Logged by: %s" % (self.separator,
                                                            self.currently_logging[self.active_log_file]))
        self._add_to_log(*args, **kwargs)

    def __call__(self, *args, print_calling_method=None, **kwargs):
        caller = inspect.stack()[1]
        caller = "'%s' in '%s'" % (caller[3], caller[1].rpartition("/")[2])
        self._log(caller, print_calling_method, *args, **kwargs)

    def __enter__(self):
        """
        Context manager
        Sets logger as global print function within context
        """
        __builtins__["print"] = self
        return self

    def __exit__(self, *args):
        """
        Revert to default print function in global scope
        """
        __builtins__["print"] = self.print_f
        return self
