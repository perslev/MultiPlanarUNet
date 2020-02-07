import subprocess
import os
import pkgutil


class VersionController(object):
    def __init__(self, logger=None, package='mpunet'):
        from mpunet.logging.default_logger import ScreenLogger
        self.package_name = package
        self.package_loader = pkgutil.get_loader(package)
        self.logger = logger or ScreenLogger()
        self.git_path = os.path.split(os.path.split(self.package_loader.path)[0])[0]
        self._mem_path = None

    def log_version(self, logger=None):
        logger = logger or self.logger
        logger("{} version: {} ({}, {})".format(self.package_name,
                                                self.version,
                                                self.branch,
                                                self.current_commit))

    def __enter__(self):
        self._mem_path = os.getcwd()
        os.chdir(self.git_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self._mem_path)
        self._mem_path = None

    def check_git(self):
        return bool(self.git_query("git status")) and \
               os.path.exists(self.git_path + "/.git")

    def git_query(self, string):
        with self:
            try:
                p = subprocess.Popen(string.split(),
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
                out, err = p.communicate()
            except FileNotFoundError:
                return None
            if not out:
                return None
            out = out.decode("utf-8").strip(" \n")
        return out

    @property
    def remote_url(self):
        return self.git_query("git config --get remote.origin.url")

    @property
    def version(self):
        module = self.package_loader.load_module()
        return module.__version__

    @property
    def current_commit(self):
        return self.git_query("git rev-parse --short HEAD")

    @property
    def latest_commit_in_branch(self, branch=None):
        branch = branch or self.branch
        url = self.remote_url
        commit = self.git_query("git ls-remote {} refs/heads/{}".format(
            url, branch
        ))
        if commit is None:
            raise OSError("Could not determine latest commit, did not find git.")
        return commit[:7]

    @property
    def branch(self):
        return self.git_query("git symbolic-ref --short HEAD")

    def set_commit(self, commit_id):
        self.git_query("git reset --hard {}".format(str(commit_id)[:7]))

    def set_branch(self, branch):
        self.git_query("git checkout {}".format(branch))

    def set_version(self, version):
        version = str(version).lower().strip(" v")
        self.set_branch("v{}".format(version))
        self.set_commit(self.latest_commit_in_branch)
