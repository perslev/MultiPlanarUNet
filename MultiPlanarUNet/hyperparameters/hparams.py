from ruamel.yaml import YAML
import os
import re
from MultiPlanarUNet.logging import ScreenLogger
import numpy as np


def _cb_paths_to_abs_paths(callbacks, patterns, project_dir):
    for cb in callbacks:
        for arg_name in cb["kwargs"]:
            if any([bool(re.match(p, arg_name, flags=re.IGNORECASE))
                    for p in patterns]):
                path = cb["kwargs"][arg_name]
                if not os.path.isabs(path):
                    cb["kwargs"][arg_name] = os.path.abspath(os.path.join(project_dir, path))
    return callbacks


def _check_deprecated_params(hparams, logger):
    if hparams.get('fit') and hparams["fit"].get("sparse"):
        from MultiPlanarUNet.errors.deprecated_warnings import warn_sparse_param
        warn_sparse_param(logger)


def _check_version(hparams, logger):
    from MultiPlanarUNet.bin.version import VersionController
    vc = VersionController()
    if not vc.check_git():
        logger.warn("Path {} does not contain a Git repository, or Git is not"
                    " installed on this system. The software verison match "
                    "could not be varified against the hyperparameter file."
                    "".format(vc.git_path))
        return
    if "__VERSION__" not in hparams:
        e = "Could not infer the software version used to produce the " \
            "hyperparameter file of this project. Using a later " \
            "version of the MultiPlanarUNet software on this project " \
            "may produce unexpected results. If you wish to continue " \
            "using this software version on this project dir, " \
            "manually add the following line to the hyperparameter file:" \
            " \n\n__VERSION__: {}\n".format(vc.version)
        logger.warn(e)
        raise RuntimeWarning(e)
    hp_version = hparams["__VERSION__"]
    if isinstance(hp_version, str) and vc.version != hp_version:
        e = "Parameter file indicates that this project was created " \
            "under MultiPlanarUNet version {}, but the current " \
            "version is {}. If you wish to continue " \
            "using this software version on this project dir, " \
            "manually add the following line to the hyperparameter " \
            "file:\n\n__VERSION__: {}\n".format(hp_version, vc.version,
                                                vc.version)
        logger.warn(e)
        raise RuntimeWarning(e)


def _set_version(hparams, logger=None):
    from MultiPlanarUNet.bin.version import VersionController
    vc = VersionController()
    if logger:
        vc.log_version(logger)
    v, b, c = vc.version, vc.branch, vc.current_commit
    hparams.set_value(None, "__VERSION__", v)
    hparams.set_value(None, "__BRANCH__", b)
    hparams.set_value(None, "__COMMIT__", c)
    hparams.save_current()


class YAMLHParams(dict):
    def __init__(self, yaml_path, logger=None, no_log=False,
                 no_version_control=False, **kwargs):
        dict.__init__(self, **kwargs)

        # Set logger or default print
        self.logger = logger or ScreenLogger()

        # Set YAML path
        self.yaml_path = os.path.abspath(yaml_path)
        self.string_rep = ""
        self.project_path = os.path.split(self.yaml_path)[0]
        if not os.path.exists(self.yaml_path):
            raise OSError("YAML path '%s' does not exist" % self.yaml_path)
        else:
            with open(self.yaml_path, "r") as yaml_file:
                for line in yaml_file:
                    self.string_rep += line
            hparams = YAML(typ="safe").load(self.string_rep)

        # Set dict elements
        self.update({k: hparams[k] for k in hparams if k[:4] != "__CB"})

        if self.get('fit') and self["fit"].get("callbacks"):
            # Convert potential callback paths to absolute paths
            cb = _cb_paths_to_abs_paths(callbacks=self["fit"]["callbacks"],
                                        patterns=("log.?dir",
                                                  "file.?name",
                                                  "file.?path"),
                                        project_dir=self.project_path)
            self["fit"]["callbacks"] = cb

        # Log basic information here...
        self.no_log = no_log
        if not self.no_log:
            self.logger("YAML path:    %s" % self.yaml_path)

        # Version controlling
        _check_deprecated_params(self, self.logger)
        if not no_version_control:
            _check_version(self, self.logger)
            _set_version(self, self.logger if not no_log else None)

    @property
    def groups(self):
        groups_re = re.compile(r"\n^(?![ \n])(.*?:.*?\n)", re.MULTILINE)
        start, groups = 0, []
        for iter in re.finditer(groups_re, self.string_rep):
            end = iter.start(0)
            groups.append(self.string_rep[start:end])
            start = end
        groups.append(self.string_rep[start:])
        return groups

    def get_group(self, group_name):
        groups = [g.lstrip("\n").lstrip(" ") for g in self.groups]
        return groups[[g.split(":")[0] for g in groups].index(group_name)]

    def add_group(self, yaml_string):
        group_name = yaml_string.lstrip(" ").lstrip("\n").split(":")[0]

        # Set dict version in memory
        self[group_name] = YAML().load(yaml_string)

        # Add pure yaml string to string representation
        self.string_rep += "\n" + yaml_string

    def delete_group(self, group_name):
        self.string_rep = self.string_rep.replace(self.get_group(group_name), "")
        del self[group_name]

    def get_from_anywhere(self, key):
        found = []
        for group_str in self:
            group = self[group_str]
            try:
                f = key in group
            except TypeError:
                f = False
            if f:
                found.append((group, group[key]))
        if len(found) > 1:
            self.logger("[ERROR] Found key '%s' in multiple groups (%s)" %
                        (key, [g[0] for g in found]))
        elif len(found) == 0:
            return None
        else:
            return found[0][1]

    def log(self):
        for item in self:
            self.logger("%s\t\t%s" % (item, self[item]))

    def set_value_no_group(self, name, value, overwrite=False):
        if name in self:
            if not overwrite:
                self.logger("Item of name '{}' already set with value '{}'."
                            " Skipping.".format(name, value))
                return False
            # Remove existing
            del self[name]
            before, _, after = self.string_rep.partition(name)
            after = after.split("\n", 1)[1]
            self.string_rep = before + after
        self[name] = value
        self.string_rep = self.string_rep.rstrip("\n") + "\n{}: {}\n".format(name,
                                                                             value)
        return True

    def set_value(self, subdir, name, value, update_string_rep=True,
                  overwrite=False, err_on_missing_dir=True):
        if subdir is None:
            return self.set_value_no_group(name, value, overwrite=True)
        if subdir not in self:
            if err_on_missing_dir:
                raise AttributeError("Subdir '{}' does not exist.".format(subdir))
            else:
                if not self.no_log:
                    self.logger("Subdir {} does not exist. Skipping.".format(subdir))
                return False
        exists = name in self[subdir]
        cur_value = self[subdir].get(name)
        if not exists or (cur_value is None or cur_value is False) or overwrite:
            # str rep of value
            if isinstance(value, np.ndarray):
                str_value = np.array2string(value, separator=", ")
            else:
                str_value = str(value)

            if not self.no_log:
                self.logger("Setting value '%s' in subdir '%s' with "
                            "name '%s'" % (str_value, subdir, name))

            # Set the value in memory
            name_exists = name in self[subdir]
            self[subdir][name] = value

            # Update the string representation as well?
            if update_string_rep:
                # Get relevant group
                l = len(subdir)
                group = [g for g in self.groups if g.lstrip()[:l] == subdir][0]

                # Replace None or False with the new value within the group
                pattern = re.compile(r"%s:[ ]+(.*)[ ]?\n" % name)

                # Find and insert new value
                def rep_func(match):
                    """
                    Returns the full match with the
                    capture group replaced by str(value)
                    """
                    return match.group().replace(match.groups(1)[0], str_value)
                new_group = re.sub(pattern, rep_func, group)

                if not name_exists:
                    # Add the field if not existing already
                    assert new_group == group  # No changes should occur
                    new_group = new_group.strip("\n")
                    temp = new_group.split("\n")[1]
                    indent = len(temp) - len(temp.lstrip())
                    new_field = (" " * indent) + "%s: %s" % (name, str_value)
                    new_group = "\n%s\n%s\n" % (new_group, new_field)

                # Update string representation
                self.string_rep = self.string_rep.replace(group, new_group)
            return True
        else:
            self.logger("Attribute '%s' in subdir '%s' already set "
                        "with value '%s'" % (name, subdir, cur_value))
            return False

    def save_current(self, out_path=None):
        # Write to file
        out_path = os.path.abspath(out_path or self.yaml_path)
        if not self.no_log:
            self.logger("Saving current YAML configuration to file:\n", out_path)
        with open(out_path, "w") as out_f:
            out_f.write(self.string_rep)
