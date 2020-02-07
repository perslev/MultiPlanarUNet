import unittest
import os
import numpy as np


class TestImagePairWithValidImage(unittest.TestCase):
    """
    Tests basic properties and error raising functionalities of an ImagePair
    object initialized with a (valid) nii.gz file stored on disk.
    """
    # Defines the properties of the (valid) image
    image_path = ".temp_valid_image.nii.gz"
    data = np.random.randn(12, 14, 16, 3).astype(np.float64)
    affine = np.diag([1, 0.5, 0.1, 1])

    @classmethod
    def setUpClass(cls):
        """ Save a temperature, valid image (nii.gz) file to disk """
        if os.path.exists(cls.image_path):
            raise OSError("Out path {} already exists".format(cls.image_path))
        import nibabel as nib
        nii = nib.Nifti1Image(cls.data, affine=cls.affine)
        nib.save(nii, cls.image_path)

    @classmethod
    def tearDownClass(cls):
        """ Remove the temporary nii.gz file """
        if os.path.exists(cls.image_path):
            os.remove(cls.image_path)

    def setUp(self):
        """ Load an ImagePair object from the .nii.gz file stored on disk """
        from mpunet.image.image_pair import ImagePair
        self.im = ImagePair(img_path=self.image_path)

    def test_stored_image_matches_disk_image(self):
        """ Tests if the stored image matches the Nifti image saved to disk """

        # Assert some fields are as expected
        self.assertTrue(self.im.predict_mode,
                        msg="ImagePair initialized with only an image should "
                            "have attribute predict_mode=True.")
        self.assertListEqual(list(self.im.shape), list(self.data.shape),
                             msg="The dimensionality of the loaded image "
                                 "object does not match the expected "
                                 "(got {}, expected {})".format(
                                 self.im.image.shape, self.data.shape))
        self.assertTrue(np.isclose(self.im.image, self.data).all(),
                        msg="The data of the loaded image is not (nearly) "
                            "identical to the data that was saved to disk.")
        self.assertTrue(self.im.image.dtype == np.dtype("float32"),
                        msg="The stored image has dtype {}, but ImagePair "
                            "should always store float32 "
                            "images".format(self.im.image.dtype))
        self.assertTrue(np.isclose(self.im.affine, self.affine).all(),
                        msg="The stored affine matrix is not (nearly) "
                            "identical to the one saved to disk.")

    def test_error_raising(self):
        """ Asserts that proper errors are raised with illegal usage """

        # Check errors are raised with illegal actions
        from mpunet.errors.image_errors import (NoLabelFileError,
                                                ReadOnlyAttributeError)
        with self.assertRaises(NoLabelFileError,
                               msg="Referencing the labels attribute of an "
                                   "ImagePair that has no labels should raise "
                                   "a {}".format(NoLabelFileError.__name__)):
            _ = self.im.labels
        with self.assertRaises(ReadOnlyAttributeError,
                               msg="Trying to set the 'image' attribute on an "
                                   "ImagePair object should raise a "
                                   "{}".format(ReadOnlyAttributeError.__name__)):
            self.im.image = [1, 2, 3]
        with self.assertRaises(ReadOnlyAttributeError,
                               msg="Trying to set the 'labels' attribute on an"
                                   " ImagePair object should raise a "
                                   "{}".format(ReadOnlyAttributeError.__name__)):
            self.im.labels = [1, 2, 3]
        with self.assertRaises(ReadOnlyAttributeError,
                               msg="Trying to set the 'affine' attribute on an"
                                   " ImagePair object should raise a "
                                   "{}".format(ReadOnlyAttributeError.__name__)):
            self.im.affine = [1, 2, 3]

    def test_shape_values(self):
        """ Tests that the ImagePair stores correct image coordinates """
        
        # Check voxel coordinates center
        self.assertListEqual(list(self.im.center), [5.5, 6.5, 7.5],
                             msg="Error in calculation of image voxel center "
                                 "(zero-indexed) coordinates")

        # Check scanner space center
        expected = np.array([5.5, 3.25, 0.75])
        got = np.array(self.im.real_center)
        self.assertTrue(np.isclose(got, expected).all(),
                        msg="Error in calculation of image scanner "
                            "space center coordinates. Expected "
                            "{}, got {}".format(expected, got))

        # Check real shape
        expected = np.array([12, 7, 1.6])
        got = np.array(self.im.real_shape)
        self.assertTrue(np.isclose(got, expected).all(),
                        msg="Error in calculation of image real shape "
                            "(mm in scanner space). Expected {}, "
                            "got {}".format(expected, got))
