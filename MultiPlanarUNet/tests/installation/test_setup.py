import unittest
import os
import shutil


class TestSetup(unittest.TestCase):

    def test_mp_on_path(self):
        """ Tests if the mp script is reachable by command 'mp' """
        path = shutil.which("mp")
        self.assertIsNot(path, None)
        self.assertTrue(os.path.exists(path))
