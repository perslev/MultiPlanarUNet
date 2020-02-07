import nibabel as nib
import numpy as np
import numpy.linalg as npl
import random
from mpunet.interpolation.linalg import mgrid_to_points, points_to_mgrid
from mpunet.interpolation.linalg import get_rotation_matrix


def get_pix_dim(nii_image):
    return nii_image.header["pixdim"][1:4]


def get_real_image_size(nii_image):
    pix_dims = get_pix_dim(nii_image)
    shape = np.asarray(nii_image.shape)[:3]
    return shape * pix_dims


def get_maximum_real_dim(nii_image):
    return np.max(get_real_image_size(nii_image))


def get_bounding_sphere_radius(nii_image):
    return np.linalg.norm(nii_image.center)


def get_bounding_sphere_real_radius(nii_image):
    real_dim = get_real_image_size(nii_image)
    return np.linalg.norm(real_dim / 2)


def get_maximum_real_dim_in_folder(folder):
    import os
    sizes = []
    for f in os.listdir(folder):
        if os.path.splitext(f)[-1] not in (".nii", ".gz"):
            continue
        else:
            f = os.path.join(folder, f)
        im = nib.load(f)
        sizes.append(get_maximum_real_dim(im))
    return np.max(sizes)


def get_voxel_grid(images, as_points=False):
    shape = images.shape[:3]
    grid = np.mgrid[0:shape[0]:1,
                    0:shape[1]:1,
                    0:shape[2]:1]

    if as_points:
        return mgrid_to_points(grid)
    else:
        return grid


def get_angle(v1, v2):
    v1_u = v1 / npl.norm(v1)
    v2_u = v2 / npl.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def get_voxel_axes_real_space(image, affine, return_basis=False):
    g_xx, g_yy, g_zz = get_voxel_axes(image)

    # Extract orthonormal basis matrix
    assert affine.shape == (4, 4)
    basis = affine[:-1, :-1]

    # Make sure basis is aligned with regular grid
    # Otherwise, rotate the grid and store transformation for later use
    pixdims = np.linalg.norm(basis, axis=0)
    transform = np.diag(pixdims)
    # sign = np.sign([np.dot(transform[:, i], basis[:, i]) for i in range(3)])
    # transform = np.diag(sign).dot(transform)

    if np.any(~np.isclose(transform, basis)):
        rot_mat = transform.dot(np.linalg.inv(basis))
    else:
        rot_mat = None

    # Get grid in real space
    g_xx = g_xx * transform[0, 0]
    g_yy = g_yy * transform[1, 1]
    g_zz = g_zz * transform[2, 2]

    if return_basis:
        return (g_xx, g_yy, g_zz), transform, rot_mat
    else:
        return g_xx, g_yy, g_zz


def get_voxel_axes(image):
    x, y, z, _ = image.shape
    g_xx = np.arange(x, dtype=np.float32) - (x - 1) / 2
    g_yy = np.arange(y, dtype=np.float32) - (y - 1) / 2
    g_zz = np.arange(z, dtype=np.float32) - (z - 1) / 2
    return g_xx, g_yy, g_zz


def get_voxel_grid_real_space(images, append_ones=False):
    # Get shape excluding channels
    shape = images.shape[:-1]

    # Get affine transforming voxel positions to real space positions
    vox_to_real_affine = images.affine[:-1, :-1]

    # Transform axes from voxel space to real space
    grid_vox_space = np.mgrid[0:shape[0]:1,
                              0:shape[1]:1,
                              0:shape[2]:1]

    # Move grid to real space
    grid_points_real_space = vox_to_real_affine.dot(
        mgrid_to_points(grid_vox_space).T).T

    # Center
    centered_grid_points_real_space = grid_points_real_space - \
                                      np.mean(grid_points_real_space, axis=0)

    # Append column of ones?
    if append_ones:
        centered_grid_points_real_space = np.column_stack(
            (grid_points_real_space,
             np.ones(len(grid_points_real_space))))

    # Return real space grid as mgrid
    points = points_to_mgrid(centered_grid_points_real_space, shape)

    return points


def get_random_views(N, dim=3, norm=np.random.normal, pos_z=True, weights=None):
    """
    http://en.wikipedia.org/wiki/N-sphere#Generating_random_points
    """
    normal_deviates = norm(size=(N, dim))
    radius = np.linalg.norm(normal_deviates, axis=1)[:, np.newaxis]
    views = normal_deviates / radius
    if pos_z:
        views[:, -1] = np.abs(views[:, -1])

    if weights is not None:
        v_weghted = views * weights
        views = v_weghted / np.linalg.norm(v_weghted, axis=1)[:, np.newaxis]

    return views


def sample_random_views_with_angle_restriction(views, min_angle_deg,
                                               auditor=None, logger=None):
    from itertools import combinations
    from mpunet.logging import ScreenLogger
    logger = logger or ScreenLogger()
    logger("Generating %i random views..." % views)

    # Weight by median sample resolution along each axis
    if auditor is not None:
        res = np.median(auditor.info["pixdims"], axis=0)
        logger("[OBS] Weighting random views by median res: %s" % res)
    else:
        res = None

    N = views
    found = False
    tries = 0
    while not found:
        tries += 1
        views = get_random_views(N, dim=3, pos_z=True, weights=res)
        angles = [get_angle(v1, v2) for v1, v2 in combinations(views, 2)]
        found = np.all(np.asarray(angles) > min_angle_deg)
        min_angle_deg -= 1
    return views


def sample_plane(norm_vector, sample_dim, real_space_span,
                 real_space_sample_sphere_radius, noise_sd=0.,
                 return_real_space_grid=False):
    # Sample a random displacement
    # Get random displacement within sample sphere
    rd = np.random.randint(-real_space_sample_sphere_radius,
                           real_space_sample_sphere_radius, 1)[0]

    return sample_plane_at(norm_vector=norm_vector,
                           sample_dim=sample_dim,
                           real_space_span=real_space_span,
                           offset_from_center=rd,
                           noise_sd=noise_sd,
                           test_mode=return_real_space_grid)


def sample_plane_at(norm_vector, sample_dim, real_space_span,
                    offset_from_center, noise_sd, test_mode=False):
    # Prepare normal vector to the plane
    n_hat = np.array(norm_vector, np.float32)
    n_hat /= np.linalg.norm(n_hat)

    # Add noise?
    if type(noise_sd) is not np.ndarray:
        noise_sd = np.random.normal(scale=noise_sd, size=3)

    n_hat += noise_sd
    n_hat /= np.linalg.norm(n_hat)

    if np.all(n_hat[:-1] < 0.2):
        # Vector pointing primarily up, noise will have large effect on image
        # orientation. We force the first two components to go into the
        # positive direction to control variability of sampling
        n_hat[:-1] = np.abs(n_hat[:-1])
    if np.all(np.isclose(n_hat[:-1], 0)):
        u = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
    else:
        # Find vector in same vertical plane as nhat
        nhat_vs = n_hat.copy()
        nhat_vs[-1] = nhat_vs[-1] + 1
        nhat_vs /= np.linalg.norm(nhat_vs)

        # Get two orthogonal vectors in plane, u pointing down in z-direction
        u = get_rotation_matrix(np.cross(n_hat, nhat_vs), -90).dot(n_hat)
        v = np.cross(n_hat, u)

    # Define basis matrix + displacement to center (affine transformation)
    basis = np.column_stack((u, v, n_hat))

    # Define regular grid (centered at origin)
    hd = real_space_span // 2
    g = np.linspace(-hd, hd, sample_dim)

    j = complex(sample_dim)
    grid = np.mgrid[-hd:hd:j,
                    -hd:hd:j,
                    offset_from_center:offset_from_center:1j]

    # Calculate voxel coordinates on the real space grid
    points = mgrid_to_points(grid)

    real_points = basis.dot(points.T).T
    real_grid = points_to_mgrid(real_points, grid.shape[1:])

    if test_mode:
        return real_grid, g, np.linalg.inv(basis)
    else:
        return real_grid


def sample_box(sample_dim, real_box_dim, real_dims, noise_sd=0., test_mode=False):

    # Set sample space equal to real_dims or expanded to 1.1x sample box dim
    # 1.1x to give a little room around the image for sampling
    sample_space = np.asarray([max(i, real_box_dim*1.1) for i in real_dims])

    # Sample a random displacement
    # Get random displacement within sample space, center on origin
    d = (sample_space - real_box_dim)
    placement = np.array([random.uniform(0, d[i]) for i in range(3)]) - sample_space/2

    return sample_box_at(real_placement=placement,
                         sample_dim=sample_dim,
                         real_box_dim=real_box_dim,
                         noise_sd=noise_sd,
                         test_mode=test_mode)


def sample_box_at(real_placement, sample_dim, real_box_dim,
                  noise_sd, test_mode):

    j = complex(sample_dim)
    a, b, c = real_placement
    grid = np.mgrid[a:a + real_box_dim:j,
                    b:b + real_box_dim:j,
                    c:c + real_box_dim:j]

    rot_mat = np.eye(3)
    rot_grid = grid
    if noise_sd:
        # Get random rotation vector
        rot_axis = get_random_views(N=1, dim=3, pos_z=True)

        rot_angle = False
        while not rot_angle:
            angle = np.abs(np.random.normal(scale=noise_sd, size=1)[0])
            if angle < 2*np.pi:
                rot_angle = angle

        rot_mat = get_rotation_matrix(rot_axis, angle_rad=rot_angle)

        # Center --> apply rotation --> revert centering --> mgrid
        points = mgrid_to_points(grid)
        center = np.mean(points, axis=0)
        points -= center
        points = rot_mat.dot(points.T).T + center
        rot_grid = points_to_mgrid(points, grid.shape[1:])

    if test_mode:
        axes = (np.linspace(a, a+real_box_dim, sample_dim),
                np.linspace(b, b+real_box_dim, sample_dim),
                np.linspace(c, c+real_box_dim, sample_dim))
        return rot_grid, axes, np.linalg.inv(rot_mat)
    else:
        return rot_grid
