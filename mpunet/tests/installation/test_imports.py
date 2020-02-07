import unittest
import pkgutil


class TestImports(unittest.TestCase):

    needed_mods = ("mpunet", "matplotlib", "scipy", "numpy",
                   "nibabel", "pandas", "ruamel.yaml", "sklearn", "h5py",
                   "tensorflow", "psutil")

    def test_modules_exist(self):
        """ Check if needed modules exist (are visible to Python) """
        for mod in self.needed_mods:
            self.assertIsNot(pkgutil.find_loader(mod), None,
                             msg="Could not find loader for needed package "
                                 "'{}' - This could indicate that the package "
                                 "is not installed or is not visible to the "
                                 "current Python interpreter "
                                 "session.".format(mod))
