import unittest
import pkgutil
import importlib

_TESTS_PKG = "mpunet.tests"


def get_test_suite(test_class):
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    suite.addTests(loader.loadTestsFromTestCase(test_class))
    return suite


def _mod_name_to_cls_name(modname):
    """
    Converts the name of a module (such as test_image_pair) to the test suite
    class name (such as TestImagePair)
    """
    return "".join(map(lambda x: x.capitalize(), modname.split("_")))


def _run(test_pkg_name):
    runner = unittest.TextTestRunner()
    test_mod = importlib.import_module("{}.{}".format(_TESTS_PKG,
                                                      test_pkg_name))
    mods = pkgutil.iter_modules(test_mod.__path__)
    s = "[*] Running tests in package '{}' [*]".format(test_pkg_name)
    print("\n" + "-"*len(s) + "\n" + s + "\n" + "-"*len(s))
    suites = []
    for m in mods:
        print("Adding test module '{}'".format(m.name))
        mod = importlib.import_module("{}.{}.{}".format(_TESTS_PKG,
                                                        test_pkg_name, m.name))
        cls_name = _mod_name_to_cls_name(m.name)
        try:
            suite = get_test_suite(getattr(mod, cls_name))
        except AttributeError:
            print("-- Skipping (found no valid test "
                  "suite with name '{}')".format(cls_name))
        else:
            suites.append(suite)
    print("\nRunning...")
    runner.run(unittest.TestSuite(suites))


def run_tests(tests_to_run):
    tests_to_run = tests_to_run.lower()
    from mpunet.tests import get_test_packages
    test_pkgs = get_test_packages()
    if tests_to_run != "all" and tests_to_run not in test_pkgs:
        raise ValueError("Unknown set of tests '{}'. Must be 'all' or one "
                         "of {}.".format(tests_to_run, test_pkgs))
    to_run = test_pkgs if tests_to_run == "all" else [tests_to_run]

    for test_pkg in to_run:
        # Run each package of tests
        _run(test_pkg)


def run_all_tests():
    run_tests("all")


if __name__ == "__main__":
    run_all_tests()
