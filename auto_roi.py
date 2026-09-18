"""Automatic segmentation of irregular ROIs from labeled center points."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, disk

ROI_SUFFIXES = tuple(f"{row}{column}" for row in "ABCDE" for column in "123")
DETECTOR_CONFIG = {
    "lambda": {"title": "Lambda", "groups": ("VB", "HL", "VU", "VD"), "radius": 35},
    "minipix": {"title": "Minipix", "groups": ("HB",), "radius": 25},
}


def expected_labels(detector):
    """Return every crystal label for a detector in canonical order."""
    try:
        groups = DETECTOR_CONFIG[detector]["groups"]
    except KeyError as exc:
        raise ValueError(f"Unknown detector: {detector!r}") from exc
    return tuple(f"{group}-{suffix}" for group in groups for suffix in ROI_SUFFIXES)


def _summed_image(data, name):
    """Convert a frame stack or image into a finite float64 summed image."""
    image = np.asarray(data)
    if image.ndim == 3:
        if image.shape[0] == 0:
            raise ValueError(f"{name} contains no frames")
        image = image.sum(axis=0, dtype=np.float64)
    elif image.ndim == 2:
        image = image.astype(np.float64, copy=False)
    else:
        raise ValueError(f"{name} must be a 2D image or 3D frame stack")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{name} contains NaN or infinity")
    return image


def prepare_images(detectors):
    """Validate and sum both detector inputs."""
    if not hasattr(detectors, "keys"):
        raise TypeError("detectors must map 'lambda' and 'minipix' to arrays")
    missing = [name for name in DETECTOR_CONFIG if name not in detectors]
    if missing:
        raise KeyError(f"Missing detector data: {', '.join(missing)}")
    return {name: _summed_image(detectors[name], name) for name in DETECTOR_CONFIG}


def _smoothed(image, smooth_sigma):
    """Apply log scaling and Gaussian smoothing before segmentation."""
    return gaussian(
        np.log1p(np.clip(np.asarray(image, dtype=float), 0, None)),
        sigma=smooth_sigma,
        preserve_range=True,
    )


def segment_single_roi(
    smoothed_image,
    center_xy,
    max_radius,
    threshold_tightness=1.0,
    min_area=20,
):
    """Segment the connected component around one center point."""
    height, width = smoothed_image.shape
    x, y = center_xy
    if not (0 <= x < width and 0 <= y < height):
        return None, "center is outside image"

    x0, x1 = max(0, x - max_radius), min(width, x + max_radius + 1)
    y0, y1 = max(0, y - max_radius), min(height, y + max_radius + 1)
    crop = smoothed_image[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    search_area = (xx - x) ** 2 + (yy - y) ** 2 <= max_radius**2
    local_values = crop[search_area]
    if local_values.size == 0 or np.ptp(local_values) <= np.finfo(float).eps:
        return None, "insufficient intensity variation"

    auto_threshold = float(threshold_otsu(local_values))
    local_x, local_y = x - x0, y - y0
    seed = crop[
        max(0, local_y - 2) : local_y + 3,
        max(0, local_x - 2) : local_x + 3,
    ]
    seed_value = float(np.median(seed))
    threshold = auto_threshold + (threshold_tightness - 1.0) * (
        seed_value - auto_threshold
    )
    candidate = crop >= threshold if seed_value >= auto_threshold else crop <= threshold
    candidate = closing(candidate & search_area, disk(2))

    components = label(candidate, connectivity=2)
    seed_labels = components[
        max(0, local_y - 2) : local_y + 3,
        max(0, local_x - 2) : local_x + 3,
    ]
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        return None, "no connected component at the selected centre"

    component = binary_fill_holes(
        components == int(np.argmax(np.bincount(seed_labels)))
    )
    if np.count_nonzero(component) < min_area:
        return None, f"area is below {min_area} pixels"

    mask = np.zeros(smoothed_image.shape, dtype=bool)
    mask[y0:y1, x0:x1] = component
    return mask, ""


def detect_all_rois(
    image,
    centers,
    max_radius,
    smooth_sigma=2.0,
    threshold_tightness=1.0,
    min_area=20,
    log=print,
):
    """Segment all centers and assign overlapping pixels to the nearest one."""
    smoothed = _smoothed(image, smooth_sigma)
    successful, failed, reasons = [], [], {}
    for roi_name, center in centers.items():
        mask, reason = segment_single_roi(
            smoothed, center, max_radius, threshold_tightness, min_area
        )
        if mask is None:
            failed.append(roi_name)
            reasons[roi_name] = reason
            log(f"[FAILED] {roi_name}: {reason}")
        else:
            successful.append((roi_name, center, mask))

    label_image = np.zeros(image.shape, dtype=np.uint16)
    best_distance = np.full(image.shape, np.inf)
    overlap_count = np.zeros(image.shape, dtype=np.uint16)
    for number, (_name, (center_x, center_y), mask) in enumerate(successful, 1):
        overlap_count[mask] += 1
        ys, xs = np.nonzero(mask)
        distances = (xs - center_x) ** 2 + (ys - center_y) ** 2
        closer = distances < best_distance[ys, xs]
        label_image[ys[closer], xs[closer]] = number
        best_distance[ys[closer], xs[closer]] = distances[closer]

    overlap_pixels = int(np.count_nonzero(overlap_count > 1))
    if overlap_pixels:
        log(f"[INFO] {overlap_pixels} overlap pixels assigned to nearest centres")
    compact = np.zeros_like(label_image)
    retained = []
    for old_number, item in enumerate(successful, 1):
        pixels = label_image == old_number
        if np.any(pixels):
            retained.append(item)
            compact[pixels] = len(retained)
        else:
            roi_name = item[0]
            failed.append(roi_name)
            reasons[roi_name] = "all pixels were assigned to an overlapping ROI"
            log(f"[FAILED] {roi_name}: {reasons[roi_name]}")
    label_image = compact
    return {
        "label_image": label_image,
        "binary_mask": label_image > 0,
        "roi_labels": np.asarray([item[0] for item in retained], dtype=str),
        "centers_xy": np.asarray(
            [item[1] for item in retained], dtype=np.int32
        ).reshape(-1, 2),
        "failed_labels": np.asarray(failed, dtype=str),
        "failed_reasons": reasons,
        "overlap_pixels": overlap_pixels,
    }


def build_roi_session(
    detectors,
    centers_by_detector,
    source_label="in_memory",
    smooth_sigma=2.0,
    threshold_tightness=1.0,
    min_area=20,
    log=print,
):
    """Segment irregular ROIs independently for both detectors."""
    images = prepare_images(detectors)
    if not centers_by_detector:
        raise ValueError(
            "centers_by_detector is required; use pick-roi or propose-roi first"
        )
    selected, results = {}, {}
    for detector, config in DETECTOR_CONFIG.items():
        if detector not in centers_by_detector:
            raise KeyError(f"Missing ROI centres for {detector}")
        centers = dict(centers_by_detector[detector])
        if not centers:
            raise ValueError(f"{detector} has no ROI centers")
        selected[detector] = centers
        results[detector] = detect_all_rois(
            images[detector],
            centers,
            config["radius"],
            smooth_sigma,
            threshold_tightness,
            min_area,
            log=log,
        )
    return {
        "images": images,
        "centers": selected,
        "results": results,
        "source_label": source_label,
    }


# --------------------------------------------------------------------------
# Center reconstruction utilities for geometry changes.
# --------------------------------------------------------------------------


def _peaks(image, smooth_sigma, max_radius, threshold_rel, num_peaks):
    """Find candidate peaks in a smoothed image as ``(x, y)`` pairs."""
    from skimage.feature import peak_local_max

    smoothed = _smoothed(image, smooth_sigma)
    found = peak_local_max(
        smoothed,
        min_distance=max(2, int(max_radius) // 2),
        threshold_rel=threshold_rel,
        num_peaks=num_peaks,
        exclude_border=False,
    )
    # peak_local_max returns (row, column), which is (y, x).
    return [(int(col), int(row)) for row, col in found]


def _match_to_template(labels, template, peaks):
    """Match candidate peaks one-to-one with a geometry template."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    known = [label for label in labels if label in template]
    if not known or not peaks:
        return {}
    template_points = np.asarray([template[label] for label in known], dtype=float)
    peak_points = np.asarray(peaks, dtype=float)
    cost = cdist(template_points, peak_points)
    rows, cols = linear_sum_assignment(cost)
    return {
        known[row]: (int(peak_points[col][0]), int(peak_points[col][1]))
        for row, col in zip(rows, cols)
    }


def _assign_by_cluster(groups, peaks):
    """Cluster peaks by module and order each module as a 5-by-3 grid."""
    from scipy.cluster.vq import kmeans2

    if not peaks:
        return {}
    points = np.asarray(peaks, dtype=float)
    count = len(groups)
    if len(points) < count:
        return {}

    try:
        centroids, assignment = kmeans2(points, count, minit="++", seed=0)
    except Exception:
        return {}
    if len(set(assignment.tolist())) < count:
        return {}

    order = sorted(range(count), key=lambda index: (centroids[index][0], centroids[index][1]))
    centers = {}
    for group, cluster in zip(groups, order):
        members = points[assignment == cluster]
        # Ordering by y and then x maps to A1, A2, A3, B1, ..., E3.
        members = members[np.lexsort((members[:, 0], members[:, 1]))]
        for suffix, point in zip(ROI_SUFFIXES, members):
            centers[f"{group}-{suffix}"] = (int(point[0]), int(point[1]))
    return centers


def propose_centers(
    image,
    detector,
    template=None,
    smooth_sigma=2.0,
    threshold_rel=0.25,
):
    """Propose center points using a template match or cluster-based fallback."""
    labels = expected_labels(detector)
    groups = DETECTOR_CONFIG[detector]["groups"]
    radius = DETECTOR_CONFIG[detector]["radius"]
    peaks = _peaks(image, smooth_sigma, radius, threshold_rel, len(labels))

    if template:
        centers = _match_to_template(labels, template, peaks)
        mode = "template-match"
    else:
        centers = _assign_by_cluster(groups, peaks)
        mode = "cluster-guess"

    report = {
        "mode": mode,
        "num_peaks": len(peaks),
        "num_expected": len(labels),
        "resolved": sorted(centers),
        "missing": [label for label in labels if label not in centers],
    }
    return centers, report
