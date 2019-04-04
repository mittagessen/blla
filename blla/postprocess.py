from scipy.ndimage.filters import gaussian_filter
from skimage.filters import apply_hysteresis_threshold


def denoising_hysteresis_thresh(im, low, high, sigma):
    im = gaussian_filter(im, sigma)
    return apply_hysteresis_threshold(im, low, high)
