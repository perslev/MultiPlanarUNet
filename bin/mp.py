import argparse
import os

"""
Entry script redirecting all command line arguments to a specified script
from the MultiViewUNet.bin folder.

Usage:
mp [script] [script args...]
"""


def get_parser():
    from MultiViewUNet import bin
    import pkgutil
    mods = pkgutil.iter_modules(bin.__path__)

    usage = "mp [script] [script args...]\n\n" + \
            "Multi Planar UNet\n-----------------\n" + \
            "Available scripts:\n"

    choices = []
    file_name = os.path.split(os.path.abspath(__file__))[-1]
    for m in mods:
        if m.name == file_name[:-3] or m.ispkg:
            continue
        usage += "- " + m.name + "\n"
        choices.append(m.name)

    # Top level parser
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("script", help="Name of the mp script to run.",
                        choices=choices)
    parser.add_argument("args", help="Arguments passed to script",
                        nargs=argparse.REMAINDER)
    return parser


if __name__ == "__main__":

    # Get the script to execute, parse only first input
    parsed = get_parser().parse_args()
    script = parsed.script

    # Import the script
    import importlib
    mod = importlib.import_module(script, package="MultiViewUNet.bin")

    # Call entry function with remaining arguments
    mod.entry_func(parsed.args)
