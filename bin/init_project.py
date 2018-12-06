from argparse import ArgumentParser
import os


def copy_yaml_and_set_data_dirs(in_path, out_path, data_dir):
    # Create YAML file
    with open(out_path, "w") as out_yaml:
        with open(in_path, "r") as in_yaml:
            for line in in_yaml:
                if "<<BASE_DIR_" in line:
                    _type = line.split("<<BASE_DIR_")[-1].split(">>")[0]
                    if data_dir:
                        line = line.replace("<<BASE_DIR_%s>>" % _type, data_dir + "/%s" % _type.lower())
                    else:
                        line = line.replace("<<BASE_DIR_%s>>" % _type, "Null")
                out_yaml.write(line)

def get_parser():
    parser = ArgumentParser(description='Create a new project folder')

    # Define groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required named arguments')
    optional = parser.add_argument_group('optional named arguments')

    required.add_argument('--name', type=str, required=True,
                        help='the name of the project folder')
    optional.add_argument('--root', type=str, default=os.path.abspath("./"),
                          help='a path to the root folder in '
                               'which the project will be initialized')
    optional.add_argument("--model", type=str, default="MultiPlanar",
                          help="Specify a model type parameter file "
                               "('MultiPlanar', '3D')")
    optional.add_argument("--data_dir", type=str, default=None,
                          help="Root data folder for the project")

    return parser


def entry_func(args=None):

    default_folder = os.path.split(os.path.abspath(__file__))[0] + "/defaults"
    if not os.path.exists(default_folder):
        raise OSError("Default path not found at %s" % default_folder)

    # Parse arguments
    parser = get_parser()
    args = vars(parser.parse_args(args))
    path = os.path.abspath(args["root"])
    name = args["name"]
    preset = args["model"]
    data_dir = os.path.abspath(args["data_dir"])

    # Validate project path and create folder
    if not os.path.exists(path):
        raise OSError("root path '%s' does not exist." % args["root"])
    else:
        folder_path = "%s/%s" % (path, name)
        if os.path.exists(folder_path):
            response = input("Folder at '%s' already exists. Overwrite? "
                             "Only parameter files and code will be replaced. (y/n) " % folder_path)
            if response.lower() == "n":
                raise OSError("Folder at '%s' already exists" % folder_path)
        else:
            os.makedirs("%s/%s" % (path, name))

    # Get yaml path
    yaml_path = ("train_hparams" + "_%s" % preset).rstrip("_")

    # Write file
    copy_yaml_and_set_data_dirs(in_path=default_folder + "/%s.yaml" % yaml_path,
                                out_path="%s/train_hparams.yaml" % folder_path,
                                data_dir=data_dir)


if __name__ == "__main__":
    entry_func()
