import argparse
import os

"""
Entry script redirecting all command line arguments to a specified script
from the mpunet.bin folder.

Usage:
mp [script] [script args...]
"""


def get_parser():
    from mpunet import bin, __version__
    import pkgutil
    mods = pkgutil.iter_modules(bin.__path__)

    ids = "Multi-Planar UNet ({})".format(__version__)
    sep = "-" * len(ids)
    usage = ("mp [script] [script args...]\n\n"
             "%s\n%s\n"
             "Available scripts:\n") % (ids, sep)

    choices = []
    file_name = os.path.split(os.path.abspath(__file__))[-1]
    for m in mods:
        if isinstance(m, tuple):
            name, ispkg = m[1], m[2]
        else:
            name, ispkg = m.name, m.ispkg
        if name == file_name[:-3] or ispkg:
            continue
        usage += "- " + name + "\n"
        choices.append(name)

    # Top level parser
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("script", help="Name of the mp script to run.",
                        choices=choices)
    parser.add_argument("args", help="Arguments passed to script",
                        nargs=argparse.REMAINDER)
    return parser


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parsed = get_parser().parse_args(args)
    script = parsed.script

    # Import the script
    import importlib
    mod = importlib.import_module("mpunet.bin." + script)

    # Call entry function with remaining arguments
    mod.entry_func(parsed.args)


if __name__ == "__main__":
    entry_func()
