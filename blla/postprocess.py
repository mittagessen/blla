from scipy.ndimage.filters import gaussian_filter
from skimage.filters import apply_hysteresis_threshold


def hysteresis_thresh(im, low, high):
    return apply_hysteresis_threshold(im, low, high)
