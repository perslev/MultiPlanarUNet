from .test_all import run_tests, run_all_tests


def get_test_packages():
    import pkgutil
    mods = list(filter(lambda x: x.ispkg, pkgutil.iter_modules(__path__)))
    return [m.name for m in mods]
