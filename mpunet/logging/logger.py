import os
import inspect
from mpunet.utils.decorators import accepts
from mpunet.utils import create_folders
from contextlib import contextmanager
from threading import Lock


class Logger(object):
    def __init__(self, base_path, print_to_screen=True, active_file="log.txt",
                 overwrite_existing=False, append_existing=False,
                 print_calling_method=True, no_sub_folder=False,
                 log_prefix="", warnings_file="warnings"):
        self.base_path = os.path.abspath(base_path)
        if not no_sub_folder:
            self.path = os.path.join(self.base_path, "logs")
        else:
            self.path = self.base_path
        create_folders([self.path])
        if overwrite_existing and append_existing:
            raise ValueError("Cannot set both 'overwrite_existing' and "
                             "'append_existing' to True.")
        self.overwrite_existing = overwrite_existing
        self.append_existing = append_existing
        self._enabled = True

        # Get built in print function
        # (if overwritten globally, Logger still maintains a reference to the
        # true print function)
        self.print_f = __builtins__["print"]

        # Print options
        self.separator = "-" * 80
        self.print_to_screen = print_to_screen
        self.print_calling_method = print_calling_method

        # Set paths to log files
        self.log_files = {}
        self.currently_logging = {}
        self.prefix = "" if log_prefix is None else str(log_prefix)
        self.active_log_file = active_file
        self.warnings_file = warnings_file

        # For using the logger from multiple threads
        self.lock = Lock()

    def __repr__(self):
        return "<mpunet.logging.Logger object>"

    def __str__(self):
        return "Logger(base_path=%s, print_to_screen=%s, " \
               "overwrite_existing=%s, append_existing=%s)" \
               % (self.base_path, self.print_to_screen,
                  self.overwrite_existing, self.append_existing)

    def new_log_file(self, filename):
        file_path = os.path.join(self.path, filename)

        if os.path.exists(file_path):
            if self.overwrite_existing:
                os.remove(file_path)
            elif not self.append_existing:
                raise OSError("Logging path: %s already exists. "
                              "Initialize Logger with overwrite_existing=True "
                              "or append_existing=True to overwrite or continue"
                              " writing to the existing file." % file_path)

        self.log_files[filename] = file_path
        self.currently_logging[filename] = None

        # Add reference to model folder in log
        ref = "Log for model in: %s" % self.base_path
        self._add_to_log(ref, no_print=True)

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = bool(value)

    @contextmanager
    def disabled_in_context(self):
        self.enabled = False
        yield self
        self.enabled = True

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
        if self.prefix:
            file_name = file_name.replace(self.prefix + "_", "")
            file_name = self.prefix.rstrip("_") + "_" + file_name
        file_name, ext = os.path.splitext(file_name)
        file_name = "%s%s" % (file_name, ext or ".txt")
        self._active_log_file = file_name
        if file_name not in self.log_files:
            self.new_log_file(file_name)

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

    def __call__(self, *args,
                 print_calling_method=None,
                 out_file=None,
                 **kwargs):
        if not self.enabled:
            return None
        with self.lock:
            cur_file = self.active_log_file
            self.active_log_file = out_file or cur_file
            caller = inspect.stack()[1]
            caller = "'%s' in '%s'" % (caller[3], caller[1].rpartition("/")[2])
            self._log(caller, print_calling_method, *args, **kwargs)
            self.active_log_file = cur_file

    def warn(self, *args, **kwargs):
        self.__call__("[WARNING]", *args,
                      print_calling_method=False,
                      out_file=self.warnings_file,
                      **kwargs)
