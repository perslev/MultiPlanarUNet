from ruamel.yaml import YAML
import os
import re
from mpunet.logging import ScreenLogger
import numpy as np


def _check_deprecated_params(hparams, logger):
    if hparams.get('fit') and hparams["fit"].get("sparse"):
        from mpunet.errors.deprecated_warnings import warn_sparse_param
        warn_sparse_param(logger)


def _check_version(hparams, logger, package="mpunet"):
    from mpunet.bin.version import VersionController
    vc = VersionController(package=package)
    if not vc.check_git():
        logger.warn("Path {} does not contain a Git repository, or Git is not"
                    " installed on this system.\n-- The software verison match "
                    "could not be varified against the hyperparameter file.\n"
                    "-- Software version will not be added to the "
                    "hyperparameter file.".format(vc.git_path))
        return False
    if "__VERSION__" not in hparams:
        e = "Could not infer the software version used to produce the " \
            "hyperparameter file of this project. Using a later " \
            "version of the {} software on this project " \
            "may produce unexpected results. If you wish to continue " \
            "using this software version on this project dir, " \
            "manually add the following line to the hyperparameter file:" \
            " \n\n__VERSION__: {}\n".format(package, vc.version)
        logger.warn(e)
        raise RuntimeWarning(e)
    hp_version = hparams["__VERSION__"]
    if isinstance(hp_version, str) and vc.version != hp_version:
        e = "Parameter file indicates that this project was created " \
            "under {} version {}, but the current " \
            "version is {}. If you wish to continue " \
            "using this software version on this project dir, " \
            "manually add the following line to the hyperparameter " \
            "file:\n\n__VERSION__: {}\n".format(package, hp_version,
                                                vc.version, vc.version)
        logger.warn(e)
        raise RuntimeWarning(e)
    return True


def _set_version(hparams, logger=None, package="mpunet"):
    from mpunet.bin.version import VersionController
    vc = VersionController(package=package)
    if logger:
        vc.log_version(logger)
    v, b, c = vc.version, vc.branch, vc.current_commit
    hparams.set_value(None, "__VERSION__", v, overwrite=True)
    hparams.set_value(None, "__BRANCH__", b, overwrite=True)
    hparams.set_value(None, "__COMMIT__", c, overwrite=True)
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
                self.string_rep = yaml_file.read()
            hparams = YAML(typ="safe").load(self.string_rep)

        # Set dict elements
        self.update({k: hparams[k] for k in hparams if k[:4] != "__CB"})

        # Log basic information here...
        self.no_log = no_log
        if not self.no_log:
            self.logger("YAML path:    %s" % self.yaml_path)

        # Version controlling
        _check_deprecated_params(self, self.logger)
        if not no_version_control:
            package = kwargs.get('package') or "mpunet"
            has_git = _check_version(self, self.logger, package)
            if has_git:
                _set_version(self, self.logger if not no_log else None, package)

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
        self[group_name] = YAML().load(yaml_string)[group_name]

        # Add pure yaml string to string representation
        self.string_rep += ("\n" + yaml_string.lstrip(" \n") + "\n")

    def delete_group(self, group_name):
        self.string_rep = self.string_rep.replace(self.get_group(group_name), "")
        del self[group_name]

    def get_from_anywhere(self, key, default=None):
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
            return default
        else:
            return found[0][1]

    def log(self):
        for item in self:
            self.logger("%s\t\t%s" % (item, self[item]))

    def _update_string_line_by_name(self, name, new_value, subdir=None):
        group = self.string_rep if not subdir else self.get_group(subdir)
        lines = group.split("\n")
        found = False
        for i, line in enumerate(lines):
            if found:
                break
            if line.lstrip().startswith(name):
                new_line = line.split(":")[0] + ": {}".format(new_value)
                lines[i] = new_line
                found = True
        if not found:
            raise AttributeError("No field has the name '{}'".format(name))
        new_group = "\n".join(lines)
        self.string_rep = self.string_rep.replace(group, new_group)

    def _set_value_no_subdir(self, name, value, str_value, overwrite,
                             add_if_missing):
        cur_value = self.get(name, None)
        if name in self:
            if cur_value is not None and not overwrite:
                return "Item of name '{}' already set with value '{}'." \
                       " Skipping. (overwrite=False)".format(name, value)
            # Update string representation
            self._update_string_line_by_name(name, str_value)
        elif not add_if_missing:
            raise AttributeError("Entry with name '{}' does not exist and "
                                 "add_if_missing was set to False."
                                 "".format(name))
        else:
            # Add to end of string representation
            self.string_rep = self.string_rep.rstrip("\n") + \
                              "\n\n{}: {}\n".format(name, str_value)
        # Set the value in memory
        self[name] = value

    def _set_value_in_existing_dir(self, subdir, name, value, str_value,
                                   overwrite, add_if_missing):
        cur_value = self[subdir].get(name, None)
        if name in self[subdir]:
            if cur_value is not None and not overwrite:
                return "Entry of name '{}' already set in subdir '{}' "\
                       "with value '{}'. Skipping "\
                       "(overwrite=False).".format(name, subdir, value)
            # Update string representation of new value
            self._update_string_line_by_name(name, str_value, subdir=subdir)
        elif not add_if_missing:
            raise AttributeError("Entry with name '{}' does not exist under "
                                 "subdir '{}' and add_if_missing was set to "
                                 "False.".format(name, subdir))
        else:
            # Add the line to the group
            group = self.get_group(subdir).rstrip(" \n")
            entry = "  {}: {}".format(name, str_value)
            new_group = "{}\n{}".format(group, entry)
            self.string_rep = self.string_rep.replace(group, new_group)
        # Update value in memory
        self[subdir][name] = value

    def _set_value_in_subdir(self, subdir, name, value, str_value, overwrite,
                             add_if_missing):
        if subdir not in self:
            if not add_if_missing:
                raise AttributeError("Subdir '{}' does not "
                                     "exist.".format(subdir))
            else:
                new_group = "{}:\n  {}: {}".format(subdir, name, str_value)
                self.add_group(new_group)
                return "Subdir '{}' does not exist, creating it now... "\
                       "(add_if_missing=True)".format(subdir)
        else:
            return self._set_value_in_existing_dir(
                subdir, name, value, str_value, overwrite, add_if_missing
            )

    def set_value(self, subdir, name, value, overwrite=False, add_if_missing=True):
        # Get propper str rep of value
        if isinstance(value, np.ndarray):
            str_value = np.array2string(value, separator=", ")
        else:
            str_value = str(value)

        if subdir is None:
            status = self._set_value_no_subdir(name, value, str_value,
                                               overwrite=overwrite,
                                               add_if_missing=add_if_missing)
        else:
            status = self._set_value_in_subdir(subdir, name, value, str_value,
                                               overwrite=overwrite,
                                               add_if_missing=add_if_missing)

        # Log what was actually done
        status = status or "Setting value '{}' (type {}) in subdir '{}' " \
                           "with name '{}'".format(str_value, type(value),
                                                   subdir, name)
        self.logger(status)

    def save_current(self, out_path=None):
        # Write to file
        out_path = os.path.abspath(out_path or self.yaml_path)
        if not self.no_log:
            self.logger("Saving current YAML configuration to file:\n", out_path)
        with open(out_path, "w") as out_f:
            out_f.write(self.string_rep)
