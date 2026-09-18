"""Standalone X-ray Raman scattering helpers used by the notebooks."""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


class XRS_roi:
    """A rectangular ROI using NumPy's half-open slice convention."""

    def __init__(
        self,
        x1,
        x2,
        y1,
        y2,
        num,
        name=None,
        add=True,
        q=None,
        dq=None,
        q_ave=None,
        dq_ave=None,
        q_range=None,
        dq_range=None,
        elastic_energy=9.685,
    ):
        self.x1, self.x2 = round(x1), round(x2)
        self.y1, self.y2 = round(y1), round(y2)
        if self.x1 < 0 or self.y1 < 0:
            raise ValueError("ROI lower bounds must be non-negative")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("ROI upper bounds must not be smaller than lower bounds")
        self.num = num
        self.name = name
        self.add = add
        self.q = q
        self.dq = dq
        self.q_ave = q_ave
        self.dq_ave = dq_ave
        self.q_range = q_range
        self.dq_range = dq_range
        self.elastic_energy = elastic_energy
        self._refresh_geometry()

    def _refresh_geometry(self):
        self.x_width = self.x2 - self.x1
        self.y_width = self.y2 - self.y1
        self.x_center = 0.5 * (self.x2 + self.x1)
        self.y_center = 0.5 * (self.y2 + self.y1)
        self.pixelNo = self.x_width * self.y_width

    def x_shift(self, shift):
        shift = max(round(shift), -self.x1)
        self.x1 += shift
        self.x2 += shift
        self._refresh_geometry()

    def y_shift(self, shift):
        shift = max(round(shift), -self.y1)
        self.y1 += shift
        self.y2 += shift
        self._refresh_geometry()

    def x_expand(self, expansion):
        expansion = round(expansion)
        new_x1, new_x2 = max(0, self.x1 - expansion), self.x2 + expansion
        if new_x2 < new_x1:
            raise ValueError("x expansion would produce a negative ROI width")
        self.x1, self.x2 = new_x1, new_x2
        self._refresh_geometry()

    def y_expand(self, expansion):
        expansion = round(expansion)
        new_y1, new_y2 = max(0, self.y1 - expansion), self.y2 + expansion
        if new_y2 < new_y1:
            raise ValueError("y expansion would produce a negative ROI height")
        self.y1, self.y2 = new_y1, new_y2
        self._refresh_geometry()

    def set_x_width(self, value):
        width = round(value)
        if width < 0:
            raise ValueError("ROI width must be non-negative")
        self.x1 = max(0, round(self.x_center - width / 2))
        self.x2 = self.x1 + width
        self._refresh_geometry()

    def set_y_width(self, value):
        width = round(value)
        if width < 0:
            raise ValueError("ROI height must be non-negative")
        self.y1 = max(0, round(self.y_center - width / 2))
        self.y2 = self.y1 + width
        self._refresh_geometry()


_MODULES = ("VB", "VU", "VD", "HB", "HL", "HR")
_MODULE_REVERSE = dict(zip(_MODULES, (-1, -1, -1, 1, 1, -1)))
_ROW_OFFSETS = dict(zip("ABCDE", (-2, -1, 0, 1, 2)))
_COLUMN_OFFSETS = dict(zip("123", (-1, 0, 1)))


def qCalc(moduleAngleList, roi_name, Ei, Ef):
    """Calculate momentum transfer and its geometrical uncertainty."""
    if len(moduleAngleList) != len(_MODULES):
        raise ValueError(f"Expected {len(_MODULES)} module angles")
    try:
        module, crystal = roi_name.split("-", maxsplit=1)
        row, column = crystal[0], crystal[1]
        module_index = _MODULES.index(module)
        reverse = _MODULE_REVERSE[module]
        row_offset = _ROW_OFFSETS[row]
        column_offset = _COLUMN_OFFSETS[column]
    except (AttributeError, IndexError, KeyError, ValueError) as exc:
        raise ValueError(
            f"Invalid ROI name {roi_name!r}; expected e.g. 'VB-A1'"
        ) from exc

    hc = 12.398419297617678
    crystal_diameter = 100.0
    crystal_distance = 1070.0
    angle_offset = np.deg2rad(7.0)
    theta = moduleAngleList[module_index] + angle_offset * row_offset * reverse
    delta = angle_offset * column_offset
    scattering_angle = np.arccos(np.clip(np.cos(theta) * np.cos(delta), -1.0, 1.0))
    ki = 2.0 * np.pi * np.asarray(Ei, dtype=float) / hc
    kf = 2.0 * np.pi * np.asarray(Ef, dtype=float) / hc
    q = np.sqrt(ki**2 + kf**2 - 2.0 * ki * kf * np.cos(scattering_angle))
    numerator = ki * kf * crystal_diameter * np.sin(abs(scattering_angle)) / crystal_distance
    dq = np.divide(
        numerator,
        q,
        out=np.full(np.shape(q), np.nan, dtype=float),
        where=q != 0,
    )
    q_ave = float(np.mean(q))
    q_range = float(np.ptp(q))
    if np.any(np.isfinite(dq)):
        dq_ave = float(np.nanmean(dq))
        dq_range = float(np.nanmax(dq) - np.nanmin(dq))
    else:
        dq_ave = dq_range = float("nan")
    return q, dq, q_ave, dq_ave, q_range, dq_range


def _normalise_scan_ids(scan_ids):
    if isinstance(scan_ids, (str, int, np.integer)):
        return [scan_ids]
    return list(scan_ids)


def _find_scan_files(scan_ids, path):
    root = Path(path).expanduser()
    if root.is_file():
        prefixes = tuple(f"{scan_id}_" for scan_id in scan_ids)
        return [root] if root.name.startswith(prefixes) else []
    if not root.is_dir():
        raise FileNotFoundError(f"Scan path does not exist or is not a directory: {root}")

    found, seen = [], set()
    for scan_id in scan_ids:
        matches = [item for item in root.glob(f"{scan_id}_*.nxs") if item.is_file()]
        for scan_dir in root.glob(f"{scan_id}_*"):
            if not scan_dir.is_dir():
                continue
            preferred = scan_dir / f"{scan_dir.name}.nxs"
            matches.extend([preferred] if preferred.is_file() else scan_dir.glob("*.nxs"))
        new_matches = []
        for item in sorted(matches, key=lambda candidate: candidate.name):
            key = str(item.resolve())
            if key not in seen:
                seen.add(key)
                found.append(item)
                new_matches.append(item)
        if not new_matches:
            warnings.warn(
                f"No NeXus file found for scan {scan_id!r} in {root}",
                RuntimeWarning,
                stacklevel=2,
            )
    return found


def _dataframe_from_1d_datasets(group):
    if not isinstance(group, h5py.Group):
        return [], pd.DataFrame()
    data = {
        name: dataset[()]
        for name, dataset in group.items()
        if isinstance(dataset, h5py.Dataset) and dataset.ndim == 1
    }
    try:
        frame = pd.DataFrame(data)
    except ValueError:
        frame = pd.DataFrame({name: pd.Series(values) for name, values in data.items()})
    return list(data), frame


def readH5(scanIDs, path, useROI=False, mute=False, detectors=None, log=print):
    """Read NeXus scans using the return structure expected by the notebooks."""
    scan_files = _find_scan_files(_normalise_scan_ids(scanIDs), path)
    if detectors is None:
        detector_filter = None
    elif isinstance(detectors, str):
        detector_filter = {detectors}
    else:
        detector_filter = set(detectors)

    mot_list, det1D_list, det2D_list, roi_list = [], [], [], []
    mot_data_list, det1D_data_list, det2D_data_list, roi_data_list = [], [], [], []
    for count, filename in enumerate(scan_files):
        try:
            with h5py.File(filename, "r") as nexus:
                mot_names, mot_data = _dataframe_from_1d_datasets(
                    nexus.get("entry/instrument")
                )
                det1D_names, det2D_names = [], []
                det1D_values, det2D_data = {}, {}
                data_group = nexus.get("entry/data")
                if isinstance(data_group, h5py.Group):
                    for det_name, dataset in data_group.items():
                        if detector_filter is not None and det_name not in detector_filter:
                            continue
                        if not isinstance(dataset, h5py.Dataset):
                            continue
                        if dataset.ndim == 1:
                            det1D_names.append(det_name)
                            det1D_values[det_name] = dataset[()]
                        elif dataset.ndim == 3:
                            det2D_names.append(det_name)
                            det2D_data[det_name] = dataset[()]
                try:
                    det1D_data = pd.DataFrame(det1D_values)
                except ValueError:
                    det1D_data = pd.DataFrame(
                        {name: pd.Series(values) for name, values in det1D_values.items()}
                    )
                if useROI:
                    roi_names, roi_data = _dataframe_from_1d_datasets(
                        nexus.get("entry/roi")
                    )
                else:
                    roi_names, roi_data = [], pd.DataFrame()
        except OSError as exc:
            warnings.warn(f"Could not read {filename}: {exc}", RuntimeWarning, stacklevel=2)
            continue

        mot_list.append(mot_names)
        det1D_list.append(det1D_names)
        det2D_list.append(det2D_names)
        roi_list.append(roi_names)
        mot_data_list.append(mot_data)
        det1D_data_list.append(det1D_data)
        det2D_data_list.append(det2D_data)
        roi_data_list.append(roi_data)
        if not mute:
            scan_parts = filename.stem.split("_")
            log(f"({count}) Scan {scan_parts[0]} - {scan_parts[-1]}")
            log(f"Motors: {mot_names}")
            log(f"1D detectors: {det1D_names}")
            log("2D detectors:")
            for det_name in det2D_names:
                log(f"  {det_name} Shape: {det2D_data[det_name].shape}")
            if useROI:
                log(f"Number of ROIs: {len(roi_names)}")

    if useROI:
        return (
            mot_list,
            det1D_list,
            det2D_list,
            roi_list,
            mot_data_list,
            det1D_data_list,
            det2D_data_list,
            roi_data_list,
        )
    return (
        mot_list,
        det1D_list,
        det2D_list,
        mot_data_list,
        det1D_data_list,
        det2D_data_list,
    )


def lorentzian(x, *parameters):
    amp, center, width, background = parameters
    return amp * width**2 / ((x - center) ** 2 + width**2) + background


_DETECTORS = ("lambda", "minipix")
_REGULAR_COLUMNS = (
    "roi_label",
    "x1",
    "x2",
    "y1",
    "y2",
    "x_shift",
    "y_shift",
    "x_expand",
    "y_expand",
)


def _summed_detector(data, name):
    image = np.asarray(data)
    if image.ndim == 3:
        image = image.sum(axis=0, dtype=np.float64)
    elif image.ndim == 2:
        image = image.astype(np.float64, copy=False)
    else:
        raise ValueError(f"{name} data must be a 2D image or 3D frame stack")
    if image.size == 0 or not np.all(np.isfinite(image)):
        raise ValueError(f"{name} image is empty or contains NaN/inf")
    return image


def plot_intensity_profile(detector_stacks, detector, x, y, figsize=(18, 5)):
    """Plot a detector image and horizontal/vertical profiles through ``(x, y)``."""
    if detector not in _DETECTORS:
        raise ValueError(f"detector must be one of {_DETECTORS}")
    if detector not in detector_stacks:
        raise KeyError(f"Missing detector data: {detector}")

    image = _summed_detector(detector_stacks[detector], detector)
    x, y = int(x), int(y)
    height, width = image.shape
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Point ({x}, {y}) is outside {detector} image {image.shape}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(np.log1p(np.clip(image, 0, None)), cmap="gray")
    axes[0].axvline(x, color="r")
    axes[0].axhline(y, color="r")
    axes[0].plot(x, y, "ro")
    axes[0].set_title(f"{detector}: ({x}, {y})")
    axes[1].plot(image[y, :])
    axes[1].axvline(x, color="r")
    axes[1].set(title=f"Horizontal profile at y={y}", xlabel="x", ylabel="intensity")
    axes[2].plot(image[:, x])
    axes[2].axvline(y, color="r")
    axes[2].set(title=f"Vertical profile at x={x}", xlabel="y", ylabel="intensity")
    fig.tight_layout()
    return fig, axes


def load_auto_roi_centers(filename):
    """Read ``roi_label, x, y`` centres separated by spaces, tabs, or commas."""
    table = pd.read_csv(filename, sep=r"[\s,]+", engine="python", comment="#")
    required = {"roi_label", "x", "y"}
    if not required.issubset(table.columns):
        raise ValueError(f"{filename} must contain columns: roi_label, x, y")
    if table.empty:
        raise ValueError(f"{filename} contains no ROI centres")
    if table["roi_label"].duplicated().any():
        raise ValueError(f"{filename} contains duplicate roi_label values")

    centers = {}
    for label, x, y in table[["roi_label", "x", "y"]].itertuples(index=False, name=None):
        x_value, y_value = float(x), float(y)
        if not x_value.is_integer() or not y_value.is_integer():
            raise ValueError(f"{filename}: coordinates for {label} must be integers")
        centers[str(label)] = (int(x_value), int(y_value))
    return centers


def write_auto_roi_centers(filename, centers, labels=None):
    """Write legacy center points as ``roi_label x y`` text columns."""
    path = Path(filename)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True)
    order = list(labels) if labels is not None else sorted(centers)
    lines = ["roi_label\tx\ty"]
    for label in order:
        if label in centers:
            x, y = centers[label]
            lines.append(f"{label}\t{int(x)}\t{int(y)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines) - 1


AUTO_ROI_SCHEMA_VERSION = 1


def write_auto_roi_hdf5(
    filename,
    result,
    *,
    detector,
    scan_ids,
    image_shape,
    parameters,
):
    """Write a versioned automatic-ROI geometry file atomically."""
    import json
    import os

    import h5py

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    labels = [str(item) for item in result["roi_labels"]]
    label_image = np.asarray(result["label_image"], dtype=np.uint16)
    centers = np.asarray(result["centers_xy"], dtype=np.int32).reshape(-1, 2)
    bounds = []
    for number in range(1, len(labels) + 1):
        y, x = np.where(label_image == number)
        if x.size == 0:
            raise ValueError(f"ROI {labels[number - 1]!r} has no pixels")
        bounds.append((int(y.min()), int(y.max()) + 1, int(x.min()), int(x.max()) + 1))
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = AUTO_ROI_SCHEMA_VERSION
        handle.attrs["detector"] = str(detector)
        handle.attrs["parameters_json"] = json.dumps(parameters, sort_keys=True)
        handle.create_dataset("scan_ids", data=np.asarray(scan_ids, dtype=np.int64))
        handle.create_dataset("image_shape", data=np.asarray(image_shape, dtype=np.int64))
        handle.create_dataset("label_image", data=label_image, compression="gzip")
        handle.create_dataset("roi_labels", data=np.asarray(labels, dtype=string_type))
        handle.create_dataset("centers_xy", data=centers)
        handle.create_dataset("bounding_boxes", data=np.asarray(bounds, dtype=np.int32))
        failed = [str(item) for item in result.get("failed_labels", [])]
        handle.create_dataset("failed_labels", data=np.asarray(failed, dtype=string_type))
        reasons = result.get("failed_reasons", {})
        handle.create_dataset(
            "failed_reasons",
            data=np.asarray([str(reasons.get(item, "")) for item in failed], dtype=string_type),
        )
        handle.attrs["overlap_pixels"] = int(result.get("overlap_pixels", 0))
    os.replace(temporary, path)
    return path


def load_auto_roi_hdf5(filename, *, detector=None, image_shape=None):
    """Load and validate an automatic-ROI HDF5 geometry file."""
    import json

    import h5py

    path = Path(filename)
    with h5py.File(path, "r") as handle:
        version = int(handle.attrs.get("schema_version", -1))
        if version != AUTO_ROI_SCHEMA_VERSION:
            raise ValueError(
                f"{path} uses unsupported auto ROI schema version {version}"
            )
        stored_detector = str(handle.attrs.get("detector", ""))
        if detector is not None and stored_detector != detector:
            raise ValueError(
                f"{path} is for detector {stored_detector!r}, not {detector!r}"
            )
        stored_shape = tuple(int(item) for item in handle["image_shape"][:])
        if image_shape is not None and stored_shape != tuple(image_shape):
            raise ValueError(
                f"{path} image shape {stored_shape} does not match {tuple(image_shape)}"
            )
        label_image = np.asarray(handle["label_image"][:], dtype=np.uint16)
        labels = np.asarray(handle["roi_labels"].asstr()[:], dtype=str)
        centers = np.asarray(handle["centers_xy"][:], dtype=np.int32).reshape(-1, 2)
        bounds = np.asarray(handle["bounding_boxes"][:], dtype=np.int32).reshape(-1, 4)
        failed = np.asarray(handle["failed_labels"].asstr()[:], dtype=str)
        failed_reasons = np.asarray(handle["failed_reasons"].asstr()[:], dtype=str)
        parameters = json.loads(str(handle.attrs.get("parameters_json", "{}")))
        scan_ids = np.asarray(handle["scan_ids"][:], dtype=np.int64)
        overlap_pixels = int(handle.attrs.get("overlap_pixels", 0))
    if label_image.shape != stored_shape:
        raise ValueError(f"{path} label_image shape does not match its metadata")
    if len(labels) != len(centers) or len(labels) != len(bounds):
        raise ValueError(f"{path} contains inconsistent ROI metadata lengths")
    present = set(int(item) for item in np.unique(label_image))
    expected = set(range(len(labels) + 1))
    if not present.issubset(expected) or not set(range(1, len(labels) + 1)).issubset(present):
        raise ValueError(f"{path} label_image does not match roi_labels")
    return {
        "label_image": label_image,
        "binary_mask": label_image > 0,
        "roi_labels": labels,
        "centers_xy": centers,
        "bounding_boxes": bounds,
        "failed_labels": failed,
        "failed_reasons": dict(zip(failed.tolist(), failed_reasons.tolist())),
        "overlap_pixels": overlap_pixels,
        "detector": stored_detector,
        "image_shape": stored_shape,
        "scan_ids": scan_ids,
        "parameters": parameters,
    }


def write_regular_rois(filename, rectangles):
    """Write rectangular ROI geometry using the editable TSV schema."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, x1, x2, y1, y2 in rectangles:
        rows.append(
            {
                "roi_label": str(label), "x1": int(x1), "x2": int(x2),
                "y1": int(y1), "y2": int(y2), "x_shift": 0, "y_shift": 0,
                "x_expand": 0, "y_expand": 0,
            }
        )
    pd.DataFrame(rows, columns=_REGULAR_COLUMNS).to_csv(path, sep="\t", index=False)
    return len(rows)


def _load_regular_rois(filename, detector):
    table = pd.read_csv(filename, sep="\t")
    missing = [column for column in _REGULAR_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"{filename} is missing columns: {', '.join(missing)}")

    rois = []
    for index, row in enumerate(table.itertuples(index=False)):
        roi = XRS_roi(row.x1, row.x2, row.y1, row.y2, index, name=str(row.roi_label))
        rois.append(roi)

    adjustments = {
        name: table[name].tolist()
        for name in ("x_shift", "y_shift", "x_expand", "y_expand")
    }
    return rois, adjustments


def _regular_masks(image, rois, filter_value):
    masks, stacked_rois, pixel_counts = [], [], []
    for roi in rois:
        roi_mask = np.zeros(image.shape, dtype=bool)
        roi_mask[roi.y1 : roi.y2, roi.x1 : roi.x2] = True
        stacked_roi = image * roi_mask
        threshold = float(stacked_roi.max()) * filter_value
        mask = roi_mask & (stacked_roi > threshold)
        masks.append(mask)
        stacked_rois.append(stacked_roi)
        pixel_counts.append(int(mask.sum()))
    return masks, stacked_rois, pixel_counts


def _build_regular_detector(
    image,
    filename,
    detector,
    use_auto_adjust,
    roi_size,
    filter_value,
):
    rois, adjustments = _load_regular_rois(filename, detector)
    if use_auto_adjust:
        _initial_masks, stacked_rois, _initial_counts = _regular_masks(
            image, rois, filter_value
        )
        for roi, stacked_roi in zip(rois, stacked_rois):
            peak_y, peak_x = np.unravel_index(np.argmax(stacked_roi), image.shape)
            roi.x_shift(peak_x - roi.x_center)
            roi.y_shift(peak_y - roi.y_center)
            roi.set_x_width(roi_size)
            roi.set_y_width(roi_size)

    for index, roi in enumerate(rois):
        roi.x_expand(adjustments["x_expand"][index])
        roi.y_expand(adjustments["y_expand"][index])
        roi.x_shift(adjustments["x_shift"][index])
        roi.y_shift(adjustments["y_shift"][index])
        roi.x1 = max(0, min(image.shape[1], roi.x1))
        roi.x2 = max(0, min(image.shape[1], roi.x2))
        roi.y1 = max(0, min(image.shape[0], roi.y1))
        roi.y2 = max(0, min(image.shape[0], roi.y2))
        roi._refresh_geometry()
        if roi.x2 <= roi.x1 or roi.y2 <= roi.y1:
            raise ValueError(f"Regular ROI {roi.name!r} is empty after adjustments")
    masks, stacked_rois, pixel_counts = _regular_masks(image, rois, filter_value)

    return {
        "image": image,
        "rois": rois,
        "masks": masks,
        "stacked_rois": stacked_rois,
        "num_pixels": pixel_counts,
        "adjustments": adjustments,
        "failed_labels": [],
    }


def _unpack_auto_detector(image, result, detector):
    labels = [str(label) for label in result["roi_labels"]]
    masks = [result["label_image"] == number for number in range(1, len(labels) + 1)]
    if not masks:
        failed = ", ".join(str(label) for label in result["failed_labels"])
        raise ValueError(f"No {detector} ROI was detected; failed labels: {failed}")

    rois = []
    for index, (name, mask) in enumerate(zip(labels, masks)):
        y, x = np.where(mask)
        rois.append(XRS_roi(x.min(), x.max() + 1, y.min(), y.max() + 1, index, name=name))
    zeros = [0] * len(rois)
    return {
        "image": image,
        "rois": rois,
        "masks": masks,
        "stacked_rois": [],
        "num_pixels": [int(mask.sum()) for mask in masks],
        "adjustments": {
            "x_shift": zeros.copy(),
            "y_shift": zeros.copy(),
            "x_expand": zeros.copy(),
            "y_expand": zeros.copy(),
        },
        "failed_labels": [str(label) for label in result["failed_labels"]],
        "failed_reasons": dict(result.get("failed_reasons", {})),
        "overlap_pixels": int(result.get("overlap_pixels", 0)),
    }


def build_roi_workflow(
    detector_stacks,
    mode,
    *,
    regular_files=None,
    auto_files=None,
    use_auto_adjust=False,
    roi_size=30,
    filter_value=0.15,
    source_label="in_memory",
    smooth_sigma=2.0,
    threshold_tightness=1.0,
    min_area=20,
    on_roi_failure="warn",
    log=print,
):
    """Build a unified ROI result for rectangular or automatic geometry."""
    if mode not in {"regular", "auto"}:
        raise ValueError("mode must be 'regular' or 'auto'")
    if not 0 <= filter_value <= 1:
        raise ValueError("filter_value must be between 0 and 1")
    if on_roi_failure not in {"warn", "error"}:
        raise ValueError("on_roi_failure must be 'warn' or 'error'")
    missing = [name for name in _DETECTORS if name not in detector_stacks]
    if missing:
        raise KeyError(f"Missing detector data: {', '.join(missing)}")

    images = {
        name: _summed_detector(detector_stacks[name], name)
        for name in _DETECTORS
    }
    workflow = {"mode": mode}

    if mode == "regular":
        if regular_files is None:
            raise ValueError("regular_files is required for regular mode")
        for detector in _DETECTORS:
            if detector not in regular_files:
                raise KeyError(f"Missing regular ROI file for {detector}")
            workflow[detector] = _build_regular_detector(
                images[detector],
                regular_files[detector],
                detector,
                use_auto_adjust,
                roi_size,
                filter_value,
            )
        return workflow

    if auto_files is None:
        raise ValueError("auto_files is required for auto mode")

    import auto_roi

    centers = {}
    stored_results = {}
    for detector in _DETECTORS:
        if detector not in auto_files:
            raise KeyError(f"Missing auto ROI file for {detector}")
        auto_path = Path(auto_files[detector])
        if auto_path.suffix.lower() in {".h5", ".hdf5"}:
            stored_results[detector] = load_auto_roi_hdf5(
                auto_path, detector=detector, image_shape=images[detector].shape
            )
            centers[detector] = {
                str(label): tuple(int(value) for value in point)
                for label, point in zip(
                    stored_results[detector]["roi_labels"],
                    stored_results[detector]["centers_xy"],
                )
            }
        else:
            centers[detector] = load_auto_roi_centers(auto_path)
        height, width = images[detector].shape
        for label, (x, y) in centers[detector].items():
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError(
                    f"{detector} {label}: ({x}, {y}) is outside image {images[detector].shape}"
                )

        # Reject unknown labels instead of silently losing misspelled ROIs.
        expected = set(auto_roi.expected_labels(detector))
        unknown = sorted(set(centers[detector]) - expected)
        if unknown:
            raise ValueError(
                f"{detector} center file contains unknown labels {unknown}; "
                f"valid labels include {sorted(expected)[:3]}"
            )

    if len(stored_results) == len(_DETECTORS):
        session = {"results": stored_results}
    else:
        session = auto_roi.build_roi_session(
            images,
            centers_by_detector=centers,
            source_label=source_label,
            smooth_sigma=smooth_sigma,
            threshold_tightness=threshold_tightness,
            min_area=min_area,
            log=log,
        )
    for detector in _DETECTORS:
        result = session["results"][detector]
        detected = set(str(label) for label in result["roi_labels"])
        expected = set(auto_roi.expected_labels(detector))
        not_found = sorted(expected - detected)
        workflow[detector] = _unpack_auto_detector(
            images[detector], result, detector
        )
        workflow[detector]["centers"] = centers[detector]
        workflow[detector]["missing_labels"] = not_found
        workflow[detector]["expected_labels"] = sorted(expected)
        log(
            f"[ROI] {detector}: segmented {len(workflow[detector]['rois'])}/"
            f"{len(centers[detector])} centers ({len(expected)} canonical labels)"
        )
        if not_found:
            log(f"[WARN] {detector} labels not segmented: {not_found}")
        if workflow[detector]["failed_labels"] and on_roi_failure == "error":
            reasons = workflow[detector]["failed_reasons"]
            detail = "; ".join(f"{k}: {v}" for k, v in reasons.items())
            raise ValueError(f"{detector} ROI segmentation failed: {detail}")
    return workflow


def plot_roi_masks(workflow, figsize=(20, 10)):
    """Plot combined masks and ROI bounding boxes for both detectors."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for axis, detector in zip(axes, _DETECTORS):
        result = workflow[detector]
        combined = np.zeros(result["image"].shape, dtype=np.uint16)
        for mask in result["masks"]:
            combined += np.asarray(mask, dtype=np.uint16)
        axis.imshow(combined)
        failed = [str(label) for label in result.get("failed_labels") or []]
        missing = [str(label) for label in result.get("missing_labels") or []]
        title = f"{detector}: {len(result['rois'])} ROIs"
        if failed:
            title += f"  |  {len(failed)} failed"
        if missing:
            title += f"  |  {len(missing)} not detected"
        axis.set_title(title)
        for roi in result["rois"]:
            axis.add_patch(
                Rectangle(
                    (roi.x1, roi.y1),
                    roi.x_width,
                    roi.y_width,
                    edgecolor="red",
                    facecolor="none",
                    lw=1,
                )
            )
            axis.text(roi.x2, roi.y2, roi.name, color="red")
        if failed or missing:
            # In headless runs this plot is the primary diagnostic artifact.
            note = []
            if failed:
                note.append("failed: " + ", ".join(failed))
            if missing:
                note.append("not detected: " + ", ".join(missing))
            axis.text(
                0.01,
                0.99,
                "\n".join(note),
                transform=axis.transAxes,
                va="top",
                ha="left",
                color="yellow",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
            )
    fig.tight_layout()
    return fig, axes


__all__ = [
    "XRS_roi",
    "build_roi_workflow",
    "load_auto_roi_centers",
    "lorentzian",
    "plot_intensity_profile",
    "plot_roi_masks",
    "qCalc",
    "readH5",
    "write_auto_roi_centers",
]
