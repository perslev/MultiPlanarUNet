import subprocess
import path


class VersionController(object):
    def __init__(self):
        import MultiPlanarUNet
        code_path = MultiPlanarUNet.__path__
        assert len(code_path) == 1
        self.git_path = path.Path(code_path[0])

    def git_query(self, string):
        with self.git_path:
            p = subprocess.Popen(string.split(), stdout=subprocess.PIPE)
            out, _ = p.communicate()
            out = out.decode("utf-8").strip(" \n")
        return out

    @property
    def remote_url(self):
        return self.git_query("git config --get remote.origin.url")

    @property
    def version(self):
        from MultiPlanarUNet import __version__
        return __version__

    @property
    def current_commit(self):
        return self.git_query("git rev-parse --short HEAD")

    def get_latest_commit_in_branch(self, branch=None):
        branch = branch or self.branch
        url = self.remote_url
        return self.git_query("git ls-remote {} refs/heads/{}".format(
            url, branch
        ))[:7]

    @property
    def branch(self):
        return self.git_query("git symbolic-ref --short HEAD")
