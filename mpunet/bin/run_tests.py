import argparse
from mpunet.tests import get_test_packages


def get_parser():
    parser = argparse.ArgumentParser(description="Run all or a set of tests.")
    parser.add_argument("--tests", type=str, default="all",
                        help="Select a package of tests to run. "
                             "Must be one of: (all, {}). "
                             "Defaults to "
                             "all.".format(", ".join(get_test_packages())))
    return parser


def entry_func(args=None):

    # Get parser
    parser = get_parser().parse_args(args)
    tests_to_run = parser.tests.lower()

    from mpunet.tests import run_tests
    try:
        run_tests(tests_to_run)
    except ValueError as e:
        print(str(e))


if __name__ == "__main__":
    entry_func()
