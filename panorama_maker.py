import os
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
from imageio import imwrite
import shutil

# consts for harris_corner_detector
KERNEL_SIZE = 3
K = 0.04

# Patch radius for sample_descriptor
DESC_RAD = 3

# consts for spread_out_corners
N = 8
M = 8
RADIUS = 3


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing a grayscale image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    conv = np.array([[1, 0, -1]])
    dx = signal.convolve2d(im, conv, mode="same", boundary="symm")
    dy = signal.convolve2d(im, conv.T, mode="same", boundary="symm")

    dx_sq = utils.blur_spatial(dx ** 2, KERNEL_SIZE)
    dy_sq = utils.blur_spatial(dy ** 2, KERNEL_SIZE)
    dx_dy = utils.blur_spatial(dx * dy, KERNEL_SIZE)

    R = ((dx_sq * dy_sq) - dx_dy ** 2) - K * ((dx_sq + dy_sq) ** 2)

    non_max_suppression_map = non_maximum_suppression(R)

    # transpose to get xy coordinates instead of yx as default
    xy_corners = np.argwhere(non_max_suppression_map.T)

    return xy_corners


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].The per−descriptor dimensions
             KxK are related to the desc rad argument as follows K = 1+2∗desc rad.
    """
    desc_vec = []
    for p in pos:
        d = get_patch(im, p, desc_rad)
        d_norm = normalize_descriptor(d)
        desc_vec.append(d_norm)

    return np.array(desc_vec)


def get_patch(im, p, r):
    """
    :param im: A 2D array representing an image.
    :param p: point in the image to take patch around.
    :param r: the radius around p to take patch.
    :return: the patch in im around p in radius r.
    """
    y, x = p
    patch_indexes = np.mgrid[x - r: x + r + 1, y - r: y + r + 1]
    patch = ndimage.map_coordinates(im, patch_indexes, order=1, prefilter=False)
    return patch


def normalize_descriptor(d):
    """
    :param d: descriptor
    :return: normalize descriptor
    """
    mu = np.mean(d)
    diff = d - mu
    diff_norm = np.linalg.norm(diff)
    if diff_norm == 0:
        return diff
    normalize_d = diff / diff_norm
    return normalize_d


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    corner_points = spread_out_corners(pyr[0], M, N, RADIUS)
    feature_descriptor = sample_descriptor(pyr[2], corner_points / 4, DESC_RAD)
    return [corner_points, feature_descriptor]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    scores = _all_desc_scores(desc1, desc2)

    # calc the second max in the rows an cols
    row_max = np.partition(scores, kth=-2, axis=1)
    col_max = np.partition(scores, kth=-2, axis=0)
    row_second_maxes = row_max[:, -2].reshape((scores.shape[0], 1))
    col_second_maxes = col_max[-2, :].reshape((1, scores.shape[1]))

    desc1_match, desc2_match = np.where(
        (scores > min_score) & (scores >= row_second_maxes) & (scores >= col_second_maxes))

    return [desc1_match.astype(dtype=int), desc2_match.astype(dtype=int)]


def _all_desc_scores(desc1, desc2):
    """
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :return: a matrix scores with shape (N1,N2) where scores[i,j] is the match score of desc1[i],desc2[j]
    """
    N1 = len(desc1)
    N2 = len(desc2)
    K = desc1.shape[1]
    scores = np.dot(desc1.reshape((N1, K * K)), desc2.reshape((N2, K * K)).T)

    return scores


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12
    """
    inhomogenous_pos1 = np.hstack((pos1, np.ones((len(pos1), 1)))).astype(dtype=int)

    inhomogenous_pos2 = np.dot(H12, inhomogenous_pos1.T)
    pos2 = np.divide(inhomogenous_pos2, inhomogenous_pos2[2, :])[:2, :].T

    return pos2


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    possible_indexes = np.arange(points1.shape[0])
    n_points = 1 if translation_only else 2
    best_inliers = np.array([])
    for i in range(num_iter):
        J = np.random.choice(possible_indexes, n_points)

        H12 = estimate_rigid_transform(points1[J], points2[J], translation_only)
        points12 = apply_homography(points1, H12)

        E = np.square(np.linalg.norm(points12 - points2, axis=1))

        curr_inliers = np.where(E < inlier_tol)[0]
        if len(curr_inliers) > len(best_inliers):
            best_inliers = curr_inliers

    H12 = estimate_rigid_transform(points1[best_inliers], points2[best_inliers], translation_only)
    return [H12 / H12[2, 2], best_inliers]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Display matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    concat_im = np.hstack((im1, im2))
    plt.imshow(concat_im, cmap='gray')
    im2_x_shift = im1.shape[1]

    # plot outliers
    outliers = np.setdiff1d(np.arange(len(points1)), inliers)
    outliers_x = np.stack((points1[outliers][:, 0], points2[outliers][:, 0] + im2_x_shift))
    outliers_y = np.stack((points1[outliers][:, 1], points2[outliers][:, 1]))
    plt.plot(outliers_x, outliers_y, mfc='r', mec='r', c='b', linewidth=.4, markersize=1, marker='o')

    # plot inliers
    inliers_x = np.stack((points1[inliers][:, 0], points2[inliers][:, 0] + im2_x_shift))
    inliers_y = np.stack((points1[inliers][:, 1], points2[inliers][:, 1]))
    plt.plot(inliers_x, inliers_y, mfc='r', mec='r', c='y', linewidth=.5, markersize=1, marker='o')

    plt.show()


def accumulate_homographies(H_successive, m):
    """
    Convert a list of successive homographies to a list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography matrices where H_successive[i] is a homography which transforms
           points from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to accumulate the given homographies.
    :return: A list of M 3x3 homography matrices, where H2m[i] transforms points from coordinate system i to coordinate
             system m.
    """
    H2m = [np.eye(3)] * (len(H_successive) + 1)

    # Update for i < m
    for i in range(m - 1, -1, -1):
        H2m[i] = np.dot(H2m[i + 1], H_successive[i])
        # to keep H[2,2]=1
        H2m[i] /= H2m[i][2, 2]

    # Update for i > m
    for i in range(m + 1, len(H_successive) + 1):
        H2m[i] = np.dot(H2m[i - 1], np.linalg.inv(H_successive[i - 1]))
        # to keep H[2,2]=1
        H2m[i] /= H2m[i][2, 2]

    return H2m


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: A 3x3 homography matrix transforming an image to common coordinate system.
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    corners = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
    new_corners = apply_homography(corners, homography)

    x_min = np.floor(np.min(new_corners[:, 0]))
    x_max = np.ceil(np.max(new_corners[:, 0]))
    y_min = np.floor(np.min(new_corners[:, 1]))
    y_max = np.ceil(np.max(new_corners[:, 1]))

    return np.array([[x_min, y_min], [x_max, y_max]]).astype(dtype=int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homography with which to warp the image.
    :return: A 2d warped image.
    """
    [[x_min, y_min], [x_max, y_max]] = compute_bounding_box(homography, image.shape[1], image.shape[0])
    X_coords, Y_coords = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
    XY_coords = np.array([X_coords.flatten(), Y_coords.flatten()]).T

    warped_coords = apply_homography(XY_coords, np.linalg.inv(homography))
    warped_coords = np.array([warped_coords[:, 1], warped_coords[:, 0]])

    warped_image = ndimage.map_coordinates(image, warped_coords, order=1, prefilter=False).reshape(X_coords.shape[0],
                                                                                                   X_coords.shape[1])

    return warped_image


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]
        # if crop_left < crop_right:
        #     self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = str(os.path.join('perspective_panoramic_frames', self.file_prefix))
        # out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            if os.path.exists(out_folder):
                shutil.rmtree(out_folder)
        except:
            print('could not remove folder %s', out_folder)
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'perspective_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s_panorama.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
