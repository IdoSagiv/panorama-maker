import numpy as np
import imageio
import skimage.color
from scipy.signal import convolve2d
from scipy.ndimage import filters

MAX_GRAY_LEVEL = 255
RGB_IMAGE = 2
GRAYSCALE_IMAGE = 1
MIN_RESOLUTION = 16
BLUR_FACTOR = 2


def read_image(filename, representation):
    """
    :param filename: image path.
    :param representation: 1=grayscale, 2=RGB.
    :return: a normalized [0,1] image in the requested representation.
    """
    assert filename
    image = imageio.imread(filename).astype(np.float64) / MAX_GRAY_LEVEL
    if representation == GRAYSCALE_IMAGE:
        return skimage.color.rgb2gray(image)
    elif representation == RGB_IMAGE:
        return image


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    :param im: – a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
        in constructing the pyramid filter (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25])
    :return: pyr - the gaussian pyramid as a standard python array (not numpy) with maximum length of max_levels,
                   where each element of the array is a grayscale image
             filter_vec - normalized vector of shape (1, filter_size) built using a consequent 1D convolutions of [1 1]
             with itself in order to derive a row of the binomial coefficients which is a good approximation to the
             Gaussian profile.
    """
    pyr = [im]
    filter_vec = normalize_pascal_coef(filter_size)
    i = 0
    while i < max_levels - 1 and can_reduce(pyr[i]):
        pyr.append(reduce(pyr[i], filter_vec))
        i += 1

    return pyr, filter_vec


def can_reduce(im):
    """
    :param im:
    :return: true if we can reduce the image and maintain good resolution
    """
    return im.shape[0] >= MIN_RESOLUTION * BLUR_FACTOR and im.shape[1] >= MIN_RESOLUTION * BLUR_FACTOR


def normalize_pascal_coef(n):
    """
    :return: a vector of the n-th pascal coefficient (normalize to sum of 1)
    """
    n_tag = n - 1
    coef = [1]

    for i in range(max(n_tag, 0)):
        coef.append(int(coef[i] * (n_tag - i) / (i + 1)))

    coef = np.array(coef) / np.sum(coef)
    return coef.reshape((1, n))


def blur(im, filter_vec):
    """
    :param im: image to blur
    :param filter_vec: normalized vector of shape (1, filter_size)
    :return: blurred image.
    """
    blur_im = filters.convolve(filters.convolve(im, filter_vec), filter_vec.T)
    return blur_im


def reduce(im, filter_vec):
    """
    blur + sub sample
    :param im: image to reduce
    :param filter_vec: normalized vector of shape (1, filter_size)
    """
    blur_im = blur(im, filter_vec)
    return blur_im[::BLUR_FACTOR, ::BLUR_FACTOR]


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img
