"""medpy.metric.binary stand-in: dc, jc, hd95, asd with medpy 0.4's definitions
(surface distances via a connectivity-1 boundary and an Euclidean distance transform),
so hybridTRTinf_helper.metric_2d scores identically without medpy installed."""
import numpy as np
from scipy import ndimage


def _surface_distances(result, reference, connectivity=1):
    result = np.atleast_1d(np.asarray(result).astype(bool))
    reference = np.atleast_1d(np.asarray(reference).astype(bool))
    if not result.any():
        raise RuntimeError("The first supplied array does not contain any binary object.")
    if not reference.any():
        raise RuntimeError("The second supplied array does not contain any binary object.")
    footprint = ndimage.generate_binary_structure(result.ndim, connectivity)
    result_border = result ^ ndimage.binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ ndimage.binary_erosion(reference, structure=footprint, iterations=1)
    dt = ndimage.distance_transform_edt(~reference_border)
    return dt[result_border]


def dc(result, reference):
    result, reference = np.asarray(result).astype(bool), np.asarray(reference).astype(bool)
    intersection = np.count_nonzero(result & reference)
    size = np.count_nonzero(result) + np.count_nonzero(reference)
    return 2.0 * intersection / float(size) if size else 0.0


def jc(result, reference):
    result, reference = np.asarray(result).astype(bool), np.asarray(reference).astype(bool)
    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)
    return float(intersection) / float(union) if union else 0.0


def hd95(result, reference, voxelspacing=None, connectivity=1):
    hd1 = _surface_distances(result, reference, connectivity)
    hd2 = _surface_distances(reference, result, connectivity)
    return float(np.percentile(np.hstack((hd1, hd2)), 95))


def asd(result, reference, voxelspacing=None, connectivity=1):
    return float(_surface_distances(result, reference, connectivity).mean())
