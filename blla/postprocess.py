import math
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import label
from scipy.ndimage.filters import gaussian_filter

from skimage import filters
from skimage.draw import line
from skimage.graph import MCP_Connect
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import approximate_polygon
from skimage.transform import estimate_transform
from skimage.morphology import skeletonize_3d

from sklearn.metrics.pairwise import euclidean_distances

from itertools import combinations
from collections import defaultdict



def denoising_hysteresis_thresh(im, low, high, sigma):
    im = gaussian_filter(im, sigma)
    return apply_hysteresis_threshold(im, low, high)

def vectorize_lines(im: np.ndarray):
    """
    Vectorizes lines from a binarized array.
    """

    line_skel = skeletonize_3d(im)
    # find extremities by convolving with 3x3 filter (value == 2 on the line because of
    # 8-connected skeleton)
    line_skel = line_skel > 0
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    line_extrema = np.transpose(np.where((convolve2d(line_skel, kernel, mode='same') == 11) * line_skel))

    # this is the ugly hack from dhSegment. Instead calculating the graph
    # diameter to find the centerline of the skeleton (which is unbearably
    # slow) just take the two points with the largest euclidian distance as
    # endpoints. This breaks down in case of folded or spiral lines as the true
    # end points are closer closer than random branches on the skeleton.
    candidates = defaultdict(list)
    label_im, _ = label(line_skel, structure=np.ones((3, 3)))
    for pt in line_extrema:
        candidates[label_im[tuple(pt)]].append(pt)
    cc_extrema = []
    for pts in candidates.values():
        distance = euclidean_distances(np.stack(pts), np.stack(pts))
        i, j = np.unravel_index(distance.argmax(), distance.shape)
        cc_extrema.append(pts[i])
        cc_extrema.append(pts[j])

    class LineMCP(MCP_Connect):
        def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.connections = dict()
           self.scores = defaultdict(lambda: np.inf)

        def create_connection(self, id1, id2, pos1, pos2, cost1, cost2):
            k = (min(id1, id2), max(id1, id2))
            s = cost1 + cost2
            if self.scores[k] > s:
                self.connections[k] = (pos1, pos2, s)
                self.scores[k] = s

        def get_connections(self):
            results = []
            for k, (pos1, pos2, s) in self.connections.items():
                results.append(np.concatenate([self.traceback(pos1), self.traceback(pos2)[::-1]]))
            return results

        def goal_reached(self, int_index, float_cumcost):
            return 2 if float_cumcost else 0

    mcp = LineMCP(~line_skel)
    try:
        mcp.find_costs(cc_extrema)
    except ValueError as e:
        return []
    return [approximate_polygon(line, 3).tolist() for line in mcp.get_connections()]

