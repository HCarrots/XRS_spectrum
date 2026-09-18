"""xrs_pipeline implementation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import xrs_processing as xrsp
from xrs_config import (
    STAGE_ORDER,
    Config,
    ConfigError,
    is_blank,
)
from xrs_ui import (
    ChoiceSpec,
    SliderSpec,
    UI,
    UiCancelled,
    configure_matplotlib,
)

_DETECTORS = ("lambda", "minipix")
_DATASETS = {"lambda": "D_LAMBDA", "minipix": "D_MINIPIX"}
_LABELS = {"lambda": "Lambda", "minipix": "Minipix"}

# English note.
# English note.
_ENERGY_PV_CANDIDATES = (
    "M_DCM_B5Energy_readback",
    "M_DCM_B5Link_Energy_readback",
)

_NOMINAL_ELASTIC_ENERGY = 9.685  # English note.
_DEFAULT_MODULE_ANGLES = [145.6, 79.03, 25, 118.8, 58.85, 67.862]
_MODULE_ORDER = ("VB", "VU", "VD", "HB", "HL", "HR")


# ==========================================================================
# English note.
# ==========================================================================


class StageStore:
    """Implementation notes for StageStore."""

    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir)
        self.figures = self.dir / "figures"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.dir / "meta.json"
        if self._meta_path.is_file():
            try:
                self._meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._meta = {"stages": {}}
        else:
            self._meta = {"stages": {}}
        self._meta.setdefault("stages", {})

# English note.

    def npz_path(self, stage: str) -> Path:
        return self.dir / f"{stage}.npz"

    def save_arrays(self, stage: str, data: dict) -> None:
        np.savez_compressed(self.npz_path(stage), **data)

    def load_arrays(self, stage: str) -> dict:
        path = self.npz_path(stage)
        if not path.is_file():
            raise ConfigError(
                f"Missing intermediate result {path}; run --stage {stage} first"
            )
        with np.load(path, allow_pickle=False) as handle:
            return {name: handle[name] for name in handle.files}

    def save_frame(self, stage: str, name: str, frame: pd.DataFrame) -> None:
        frame.to_csv(self.dir / f"{stage}_{name}.tsv", sep="\t", index=False)

    def load_frame(self, stage: str, name: str) -> pd.DataFrame:
        path = self.dir / f"{stage}_{name}.tsv"
        if not path.is_file():
            raise ConfigError(f"Missing {stage} table {path}; run --stage {stage} first")
        return pd.read_csv(path, sep="\t")

# English note.

    def stage_info(self, stage: str) -> dict:
        return dict(self._meta["stages"].get(stage, {}))

    def is_current(self, stage: str, fingerprint: str) -> bool:
        info = self._meta["stages"].get(stage)
        if not info or info.get("fingerprint") != fingerprint:
            return False
        return self.npz_path(stage).is_file()

    def record(self, stage: str, fingerprint: str, **extra) -> None:
        entry = {"fingerprint": fingerprint}
        entry.update(extra)
        self._meta["stages"][stage] = entry
        self.flush()

    def flush(self) -> None:
        self._meta_path.write_text(
            json.dumps(self._meta, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    def reset_from(self, stage: str) -> None:
        """Implementation notes for reset_from."""
        start = STAGE_ORDER.index(stage)
        for name in STAGE_ORDER[start:]:
            self._meta["stages"].pop(name, None)
            self.npz_path(name).unlink(missing_ok=True)
        self.flush()


# ==========================================================================
# English note.
# ==========================================================================


def _format_problems(stage: str, problems) -> str:
    lines = [f"Stage {stage} is missing {len(problems)} parameter(s):"]
    lines += [f"  - {item}" for item in problems]
    lines.append("")
    lines.append("Interactive mode prompts for these values; --no-ui requires them in config.yaml.")
    return "\n".join(lines)


def _pick_energy_pv(configured, motor_frame: pd.DataFrame, log) -> str:
    """Implementation notes for _pick_energy_pv."""
    columns = list(motor_frame.columns)
    if not is_blank(configured):
        if configured in columns:
            return str(configured)
        raise ConfigError(
            f"Configured energy_pv {configured!r} is not in motor data. Available: {columns}"
        )
    for candidate in _ENERGY_PV_CANDIDATES:
        if candidate in columns:
            log(f"[INFO] Automatically selected energy PV: {candidate}")
            return candidate
    raise ConfigError(
        f"Could not detect an energy PV from {_ENERGY_PV_CANDIDATES}. Available: {columns}"
    )


def _normalised_stacks(det2D_list, det1D_list, i0_pv, divide, log):
    """Implementation notes for _normalised_stacks."""
    if not divide:
        log("[INFO] divide_by_i0 = false; I0 normalization is disabled")
        return (
            [np.asarray(item[_DATASETS["lambda"]], dtype=float) for item in det2D_list],
            [np.asarray(item[_DATASETS["minipix"]], dtype=float) for item in det2D_list],
        )

    if det1D_list is None:
        raise ConfigError("divide_by_i0 = true, but no 1D I0 data was provided")

    i0_list = []
    for item in det1D_list:
        if i0_pv not in item:
            raise ConfigError(
                f"1D data does not contain {i0_pv!r}; available: {list(item.columns)}"
            )
        i0_list.append(np.abs(np.asarray(item[i0_pv], dtype=float)))

    def scale(stack, i0):
        if stack.shape[0] != i0.shape[0]:
            raise ConfigError(
                f"Detector frame count {stack.shape[0]} does not match I0 count {i0.shape[0]}"
            )
        return stack / i0[:, None, None]

    return (
        [scale(np.asarray(item[_DATASETS["lambda"]], dtype=float), i0)
         for item, i0 in zip(det2D_list, i0_list)],
        [scale(np.asarray(item[_DATASETS["minipix"]], dtype=float), i0)
         for item, i0 in zip(det2D_list, i0_list)],
    )


def _roi_spectra(stack, boxes, masks):
    """Implementation notes for _roi_spectra."""
    raw, masked = [], []
    for (y1, y2, x1, x2), mask in zip(boxes, masks):
        roi_stack = stack[:, y1:y2, x1:x2]
        raw.append(roi_stack.sum(axis=(1, 2)))
        masked.append((roi_stack * mask[y1:y2, x1:x2]).sum(axis=(1, 2)))
    if not raw:
        empty = np.empty((0, int(stack.shape[0])), dtype=float)
        return empty, empty.copy()
    return np.asarray(raw, dtype=float), np.asarray(masked, dtype=float)


def _lorentzian_fit(x_values, y_values, fit_type):
    """Implementation notes for _lorentzian_fit."""
    from scipy.optimize import OptimizeWarning, curve_fit
    import warnings as _warnings

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size:
        center = x[np.argmax(y)]
        p0 = np.array([np.ptp(y), center, 0.001, np.min(y)])
    else:
        p0 = np.array([0, np.nan, 0.001, 0])
    if x.size < 4 or np.ptp(y) <= np.finfo(float).eps:
        return p0, np.nan
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", OptimizeWarning)
            coeff, _ = curve_fit(fit_type, x, y, p0=p0, maxfev=10000)
    except (RuntimeError, ValueError, TypeError):
        return p0, np.nan
    residuals = y - fit_type(x, *coeff)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = np.nan if ss_tot <= np.finfo(float).eps else 1 - ss_res / ss_tot
    return coeff, r_squared


def _save_figure(fig, path: Path, log) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    log(f"[QC] {path}")


def _close(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def _pack_masks(masks, image_shape) -> np.ndarray:
    """Pack masks for one detector, including an empty detector."""
    height, width = (int(value) for value in image_shape)
    if not masks:
        return np.empty((0, (height * width + 7) // 8), dtype=np.uint8)
    stacked = np.stack([np.asarray(mask, dtype=bool) for mask in masks])
    if stacked.shape[1:] != (height, width):
        raise ValueError(
            f"Mask shape {stacked.shape[1:]} does not match detector shape {(height, width)}"
        )
    flat = stacked.reshape(stacked.shape[0], -1)
    return np.packbits(flat, axis=1)


def _unpack_masks(packed, shape, count) -> np.ndarray:
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1)
    height, width = int(shape[0]), int(shape[1])
    return bits[:, : height * width].reshape(count, height, width).astype(bool)


# ==========================================================================
# English note.
# ==========================================================================


def _acquire_elastic(cfg: Config, ui: UI, log) -> None:
    """Acquire only the data location and elastic scan identifiers."""
    if is_blank(cfg.get("data.root")):
        cfg.set(
            "data.root",
            ui.ask(
                "Data root (data.root)",
                None,
                help_text="Example: /hepsdatafs/ID33/202504/Data/GID33-250416-01/",
            ),
        )
    if is_blank(cfg.get("elastic.scan_ids")):
        cfg.set(
            "elastic.scan_ids",
            ui.ask_list("Elastic scan IDs (elastic.scan_ids)", None, int, "Example: 14522"),
        )


def _backup(path: Path, log) -> None:
    """Create a recoverable backup before replacing an ROI geometry file."""
    if path.is_file():
        backup = path.with_name(path.name + ".bak")
        shutil.copy2(path, backup)
        log(f"[ROI] Backed up {path} to {backup}")


def _regular_geometry(path: Path, detector: str):
    rois, _adjustments = xrsp._load_regular_rois(path, detector)
    labels = [roi.name for roi in rois]
    bounds = [(roi.y1, roi.y2, roi.x1, roi.x2) for roi in rois]
    return labels, bounds


def _result_bounds(result):
    if "bounding_boxes" in result:
        return [tuple(int(value) for value in row) for row in result["bounding_boxes"]]
    output = []
    label_image = np.asarray(result["label_image"])
    for number in range(1, len(result["roi_labels"]) + 1):
        y, x = np.where(label_image == number)
        output.append((int(y.min()), int(y.max()) + 1, int(x.min()), int(x.max()) + 1))
    return output


def _auto_target(cfg: Config, detector: str) -> Path:
    value = cfg.get(f"elastic.auto_files.{detector}")
    if is_blank(value):
        value = f"roi/auto_ROI_{detector}.h5"
        cfg.set(f"elastic.auto_files.{detector}", value)
    return cfg.resolve(value)


def _prepare_roi_inputs(cfg: Config, ui: UI, images, scan_ids, log) -> str:
    """Preview elastic images, select a mode, and create or reuse ROI files."""
    if ui.interactive:
        ui.preview_images(images, "Summed elastic detector images")
    mode = cfg.get("elastic.roi_mode")
    if is_blank(mode):
        mode = ui.ask(
            "ROI mode (elastic.roi_mode)", None, choices=("regular", "auto"),
            help_text="regular = rectangular ROIs; auto = segmented irregular ROIs",
        )
        cfg.set("elastic.roi_mode", mode)
    if mode not in {"regular", "auto"}:
        raise ConfigError("elastic.roi_mode must be 'regular' or 'auto'")

    import auto_roi

    parameters = dict(cfg.get("elastic.auto_params", {}) or {})
    for detector in _DETECTORS:
        image = images[detector]
        if mode == "regular":
            key = f"elastic.regular_files.{detector}"
            if is_blank(cfg.get(key)):
                cfg.set(key, f"roi/roi_{detector}.txt")
            target = cfg.resolve(cfg.get(key))
            reuse = False
            if target.is_file():
                labels, bounds = _regular_geometry(target, detector)
                reuse = not ui.interactive or ui.review_geometry(
                    image, labels, bounds, f"{_LABELS[detector]} rectangular ROIs"
                )
            if not reuse:
                rectangles = ui.pick_rectangles(
                    image, auto_roi.expected_labels(detector),
                    f"Draw {_LABELS[detector]} rectangular ROIs",
                )
                _backup(target, log)
                count = xrsp.write_regular_rois(target, rectangles)
                log(f"[ROI] Wrote {count} rectangular ROIs to {target}")
            cfg.save()
            if detector == "lambda":
                log("[ROI] Lambda ROI file is saved; opening the Minipix editor")
            continue

        target = _auto_target(cfg, detector)
        reuse = False
        if target.is_file():
            try:
                stored = xrsp.load_auto_roi_hdf5(
                    target, detector=detector, image_shape=image.shape
                )
            except ValueError as exc:
                if not ui.interactive:
                    raise
                log(f"[ROI] Existing {detector} HDF5 cannot be reused: {exc}")
            else:
                reuse = not ui.interactive or ui.review_geometry(
                    stored["label_image"], stored["roi_labels"],
                    _result_bounds(stored), f"{_LABELS[detector]} automatic ROIs",
                )
        if reuse:
            continue

        legacy_value = cfg.get(f"elastic.auto_centers.{detector}")
        legacy_path = None if is_blank(legacy_value) else cfg.resolve(legacy_value)
        use_legacy = legacy_path is not None and legacy_path.is_file()
        while True:
            if use_legacy:
                centers = xrsp.load_auto_roi_centers(legacy_path)
                log(f"[ROI] Migrating legacy centers from {legacy_path}")
            else:
                centers = ui.pick_points(
                    image, auto_roi.expected_labels(detector),
                    f"Select {_LABELS[detector]} ROI centers",
                )
            result = auto_roi.detect_all_rois(
                image,
                centers,
                auto_roi.DETECTOR_CONFIG[detector]["radius"],
                smooth_sigma=float(parameters.get("smooth_sigma", 2.0)),
                threshold_tightness=float(parameters.get("threshold_tightness", 1.0)),
                min_area=int(parameters.get("min_area", 20)),
                log=log,
            )
            if not len(result["roi_labels"]):
                if not ui.interactive:
                    raise ConfigError(f"No {detector} ROI could be segmented")
                log(f"[ROI] No {detector} ROI was segmented; select centers again")
                use_legacy = False
                continue
            if not ui.interactive or ui.review_geometry(
                result["label_image"], result["roi_labels"], _result_bounds(result),
                f"{_LABELS[detector]} generated automatic ROIs",
            ):
                break
            use_legacy = False
        _backup(target, log)
        xrsp.write_auto_roi_hdf5(
            target, result, detector=detector, scan_ids=scan_ids,
            image_shape=image.shape, parameters=parameters,
        )
        log(f"[ROI] Wrote automatic ROI geometry to {target}")
    cfg.save()
    return str(mode)


def _elastic_paths(cfg: Config, mode: str) -> dict:
    group = "regular_files" if mode == "regular" else "auto_files"
    return {
        detector: str(cfg.resolve(cfg.get(f"elastic.{group}.{detector}")))
        for detector in _DETECTORS
    }


def _elastic_controls(cfg: Config) -> list:
    controls = [
        SliderSpec(
            "filter_value",
            "filter_value (relative ROI threshold)",
            0.0,
            1.0,
            float(cfg.get("elastic.filter_value", 0.15)),
            0.01,
        ),
        SliderSpec(
            "e_lowlim",
            "Fit center lower bound (keV)",
            9.0,
            13.5,
            float(cfg.get("elastic.fit.e_lowlim", 9.67)),
            0.001,
        ),
        SliderSpec(
            "e_highlim",
            "Fit center upper bound (keV)",
            9.0,
            13.7,
            float(cfg.get("elastic.fit.e_highlim", 9.69)),
            0.001,
        ),
    ]
    if cfg.get("elastic.auto_adjust"):
        controls.append(
            SliderSpec(
                "roi_size",
                "roi_size (auto-adjusted side length)",
                5,
                120,
                int(cfg.get("elastic.roi_size", 30)),
                1,
                integer=True,
            )
        )
    return controls


def run_elastic(cfg: Config, ui: UI, log, force: bool = False, overwrite: bool = False) -> dict:
    _acquire_elastic(cfg, ui, log)
    cfg.save()

    scan_ids = [int(item) for item in cfg.get("elastic.scan_ids")]
    divide = bool(cfg.get("elastic.divide_by_i0"))
    i0_pv = cfg.get("elastic.i0_pv")

    wanted = [_DATASETS[name] for name in _DETECTORS]
    if divide:
        wanted.append(i0_pv)
    log(f"[elastic] Reading scans {scan_ids} from {cfg.raw_dir} ...")
    (_, _, _, mot_data_list, det1D_data_list, det2D_data_list) = xrsp.readH5(
        scan_ids, cfg.raw_dir, useROI=False, detectors=wanted, log=log
    )
    if not det2D_data_list:
        raise ConfigError(f"Scans {scan_ids} contain no 2D detector data")

    energy_pv = _pick_energy_pv(cfg.get("elastic.energy_pv"), mot_data_list[0], log)
    cfg.set("elastic.energy_pv", energy_pv)
    energies = []
    for index, frame in enumerate(mot_data_list):
        if energy_pv not in frame:
            raise ConfigError(
                f"Elastic scan {scan_ids[index]} does not contain energy PV {energy_pv!r}"
            )
        energies.append(np.asarray(frame[energy_pv], dtype=float))

    lambda_stacks, minipix_stacks = _normalised_stacks(
        det2D_data_list, det1D_data_list, i0_pv, divide, log
    )
    for detector, items in (("lambda", lambda_stacks), ("minipix", minipix_stacks)):
        shapes = {tuple(item.shape[1:]) for item in items}
        if len(shapes) != 1:
            raise ConfigError(
                f"Elastic {detector} image shapes differ across scans: {sorted(shapes)}"
            )
    energy = np.concatenate(energies)
    stacks = {
        "lambda": np.concatenate(lambda_stacks, axis=0),
        "minipix": np.concatenate(minipix_stacks, axis=0),
    }
    if any(stack.shape[0] != energy.size for stack in stacks.values()):
        raise ConfigError("Elastic frame counts do not match the combined energy axis")
    order = np.argsort(energy, kind="stable")
    energy = energy[order]
    stacks = {name: stack[order] for name, stack in stacks.items()}
    grid = {name: stacks[name].shape[1:] for name in _DETECTORS}
    log(f"[elastic] Detector image shapes: {grid}")

    images = {name: stacks[name].sum(axis=0, dtype=np.float64) for name in _DETECTORS}
    mode = _prepare_roi_inputs(cfg, ui, images, scan_ids, log)
    problems = cfg.problems("elastic")
    if problems:
        raise ConfigError(_format_problems("elastic", problems))

    paths = _elastic_paths(cfg, mode)
    fit_cfg = cfg.get("elastic.fit", {})
    auto_params = cfg.get("elastic.auto_params", {})
    use_filter = bool(cfg.get("elastic.use_filter", True))
    source_label = "_".join(str(item) for item in scan_ids) + "_elastic"

    def build(values: dict):
        """Implementation notes for build."""
        filter_value = float(values["filter_value"])
        roi_size = int(values.get("roi_size", cfg.get("elastic.roi_size", 30)))
        auto_adjust = bool(cfg.get("elastic.auto_adjust", False))
        kwargs = dict(
            use_auto_adjust=auto_adjust,
            roi_size=roi_size,
            filter_value=filter_value,
            source_label=source_label,
            smooth_sigma=float(auto_params.get("smooth_sigma", 2.0)),
            threshold_tightness=float(auto_params.get("threshold_tightness", 1.0)),
            min_area=int(auto_params.get("min_area", 20)),
            on_roi_failure=str(auto_params.get("on_roi_failure", "warn")),
            log=log,
        )
        if mode == "regular":
            kwargs["regular_files"] = paths
        else:
            kwargs["auto_files"] = paths
        return xrsp.build_roi_workflow(stacks, mode, **kwargs), filter_value

    def prepare(values: dict):
        """Implementation notes for prepare."""
        workflow, filter_value = build(values)
        names, detectors, boxes, masks = [], [], [], []
        curves = {"lambda": [], "minipix": []}
        for detector in _DETECTORS:
            bundle = workflow[detector]
            raw, masked = _roi_spectra(
                stacks[detector],
                [(roi.y1, roi.y2, roi.x1, roi.x2) for roi in bundle["rois"]],
                bundle["masks"],
            )
            curves[detector] = masked if use_filter else raw
            for roi, mask in zip(bundle["rois"], bundle["masks"]):
                names.append(roi.name)
                detectors.append(detector)
                boxes.append((roi.y1, roi.y2, roi.x1, roi.x2))
                masks.append(mask)
        if not names:
            raise ConfigError("At least one ROI is required across all detectors")
        finite = np.asarray(
            [row for detector in _DETECTORS for row in curves[detector]], dtype=float
        )
        fits, bad = _fit_all(
            np.asarray(names, dtype=str), energy, finite, fit_cfg, log
        )
        return {
            "workflow": workflow,
            "filter_value": filter_value,
            "names": np.asarray(names, dtype=str),
            "detectors": np.asarray(detectors, dtype=str),
            "boxes": boxes,
            "masks": masks,
            "curves": finite,
            "fits": fits,
            "bad": bad,
        }

    values = {spec.key: spec.value for spec in _elastic_controls(cfg)}
    state = None

    import matplotlib.pyplot as plt

# English note.
    qc_figure = plt.figure(figsize=(18, 6))
    qc_axes = qc_figure.subplots(1, 3, squeeze=False)

    if ui.interactive:
# English note.
# English note.
        tune_figure = plt.figure(figsize=(20, 6))
        tune_axes = tune_figure.subplots(
            1, 3, gridspec_kw={"right": 0.68}, squeeze=False
        )

        def draw(current: dict) -> None:
            nonlocal state
            state = prepare(current)
            _draw_elastic(tune_axes, state, energy, cfg)

        draw(values)
        values = ui.adjust(
            tune_figure,
            _elastic_controls(cfg),
            draw,
            title="Elastic ROI masks and fit parameters",
        )
        state = prepare(values)
        _close(tune_figure)
    else:
        state = prepare(values)

    _draw_elastic(qc_axes, state, energy, cfg)
    _save_figure(qc_figure, cfg.state_dir / "figures" / "elastic_qc.png", log)
    _close(qc_figure)

# English note.
    cfg.set("elastic.filter_value", float(values["filter_value"]))
    if "roi_size" in values:
        cfg.set("elastic.roi_size", int(values["roi_size"]))
    cfg.set("elastic.fit.e_lowlim", float(values["e_lowlim"]))
    cfg.set("elastic.fit.e_highlim", float(values["e_highlim"]))
    cfg.save()

    fits = state["fits"]
    fit_result = _fit_result_table(state)
    bad_names = fit_result.loc[fit_result["bad_fit"], "crystal"].tolist()
    log(f"[elastic] badFit = {bad_names}")
    log("[elastic] fitResult:\n" + fit_result.to_string(index=False))
    _inspect_elastic_fits(ui, state, energy, log)

    payload = {
        "roi_names": state["names"],
        "roi_detectors": state["detectors"],
        "boxes": np.asarray(state["boxes"], dtype=np.int64),
        "mask_count": np.asarray([len(state["masks"])]),
        "fit_center_kev": fits["center_kev"],
        "fit_fwhm_ev": fits["fwhm_ev"],
        "fit_amp": fits["amp"],
        "fit_bkg": fits["bkg"],
        "fit_r2": fits["r2"],
        "bad_fit": state["bad"],
        "failed_labels": np.asarray(
            [str(label) for name in _DETECTORS
             for label in state["workflow"][name]["failed_labels"]], dtype=str
        ),
        "missing_labels": np.asarray(
            [str(label) for name in _DETECTORS
             for label in state["workflow"][name].get("missing_labels", [])], dtype=str
        ),
        "energy_pv": np.asarray([energy_pv]),
        "elastic_energy": energy,
        "image_shape": np.asarray(grid["lambda"], dtype=np.int64),
        "roi_mode": np.asarray([mode]),
    }
    for detector in _DETECTORS:
        detector_masks = state["workflow"][detector]["masks"]
        payload[f"masks_packed_{detector}"] = _pack_masks(
            detector_masks, grid[detector]
        )
        payload[f"mask_count_{detector}"] = np.asarray([len(detector_masks)])
        payload[f"image_shape_{detector}"] = np.asarray(
            grid[detector], dtype=np.int64
        )
    return payload


def _fit_all(names, energy_kev, curves, fit_cfg, log) -> tuple[dict, np.ndarray]:
    """Implementation notes for _fit_all."""
    fit_type = xrsp.lorentzian
    low = float(fit_cfg.get("e_lowlim", 9.67))
    high = float(fit_cfg.get("e_highlim", 9.69))
    max_fwhm = float(fit_cfg.get("max_fwhm_ev", 2.0))
    min_r2 = float(fit_cfg.get("min_r_squared", 0.8))

    centers, widths, amps, bkgs, rsqs = [], [], [], [], []
    bad = np.zeros(len(names), dtype=bool)
    for index, name in enumerate(names):
        coeff, r_squared = _lorentzian_fit(energy_kev, curves[index], fit_type)
        center_ev = float(coeff[1]) * 1000.0
        fwhm_ev = 2.0 * abs(float(coeff[2])) * 1000.0
        amp = float(coeff[0])
        centers.append(center_ev)
        widths.append(fwhm_ev)
        amps.append(amp)
        bkgs.append(float(coeff[3]))
        rsqs.append(r_squared)
        is_bad = (
            not np.isfinite(r_squared)
            or not np.all(np.isfinite(coeff))
            or r_squared < min_r2
            or coeff[1] < low
            or coeff[1] > high
            or amp < 0
            or fwhm_ev > max_fwhm
        )
        if is_bad:
            bad[index] = True

    bad_names = [str(name) for name, flag in zip(names, bad) if flag]
    log(f"[elastic] Fitted {len(names)} ROIs; {len(bad_names)} bad fits")
    if bad_names:
        log(f"[elastic] Bad fits: {bad_names}")
    return (
        {
            "center_ev": np.asarray(centers, dtype=float),
            "center_kev": np.asarray(centers, dtype=float) / 1000.0,
            "fwhm_ev": np.asarray(widths, dtype=float),
            "amp": np.asarray(amps, dtype=float),
            "bkg": np.asarray(bkgs, dtype=float),
            "r2": np.asarray(rsqs, dtype=float),
        },
        bad,
    )


def _draw_elastic(axes, state, energy, cfg) -> None:
    """Draw detector masks and elastic data with fitted peak curves."""
    from matplotlib.patches import Rectangle

    for axis in np.asarray(axes).ravel():
        axis.clear()

    for column, detector in enumerate(_DETECTORS):
        axis = axes[0][column]
        bundle = state["workflow"][detector]
        combined = np.zeros(bundle["image"].shape, dtype=np.uint16)
        for mask in bundle["masks"]:
            combined += np.asarray(mask, dtype=np.uint16)
        axis.imshow(combined)
        axis.set_title(f"{_LABELS[detector]}: {len(bundle['rois'])} ROIs")
        for roi in bundle["rois"]:
            axis.add_patch(
                Rectangle(
                    (roi.x1, roi.y1), roi.x_width, roi.y_width,
                    edgecolor="red", facecolor="none", lw=0.8,
                )
            )
        failed = list(bundle.get("failed_labels") or [])
        missing = list(bundle.get("missing_labels") or [])
        if failed or missing:
            axis.text(
                0.01, 0.99,
                "\n".join(
                    ([f"failed: {', '.join(map(str, failed))}"] if failed else [])
                    + ([f"not detected: {', '.join(map(str, missing))}"] if missing else [])
                ),
                transform=axis.transAxes, va="top", ha="left", fontsize=7,
                color="yellow", bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
            )

    axis = axes[0][2]
    fit_energy = np.linspace(float(np.min(energy)), float(np.max(energy)), 800)
    for index, (name, curve) in enumerate(zip(state["names"], state["curves"])):
        suffix = " [bad]" if state["bad"][index] else ""
        line = axis.plot(
            energy, curve, "o", ms=2.0, alpha=0.65, label=f"{name}{suffix}"
        )[0]
        fit = state["fits"]
        fit_curve = xrsp.lorentzian(
            fit_energy,
            fit["amp"][index],
            fit["center_kev"][index],
            fit["fwhm_ev"][index] / 2000.0,
            fit["bkg"][index],
        )
        axis.plot(
            fit_energy,
            fit_curve,
            color="red" if state["bad"][index] else line.get_color(),
            ls="--" if state["bad"][index] else "-",
            lw=0.9,
        )
    axis.set_title(f"Elastic peak fits ({int(state['bad'].sum())} bad)")
    axis.set_xlabel(f"{cfg.get('elastic.energy_pv')} (keV)")
    axis.set_ylabel("counts")
    if len(state["names"]) <= 20:
        axis.legend(fontsize=6, ncol=2)


def _fit_result_table(state) -> pd.DataFrame:
    """Build the elastic fitResult table shown in the terminal."""
    boxes = np.asarray(state["boxes"], dtype=np.int64).reshape(-1, 4)
    fits = state["fits"]
    return pd.DataFrame(
        {
            "crystal": np.asarray(state["names"], dtype=str),
            "detector": np.asarray(state["detectors"], dtype=str),
            "x1": boxes[:, 2],
            "x2": boxes[:, 3],
            "y1": boxes[:, 0],
            "y2": boxes[:, 1],
            "center": fits["center_ev"],
            "width": fits["fwhm_ev"],
            "height": fits["amp"],
            "background": fits["bkg"],
            "r-square": fits["r2"],
            "bad_fit": np.asarray(state["bad"], dtype=bool),
        }
    )


def _draw_elastic_inspection(state, energy_kev, roi_name: str):
    """Draw one ROI fit and all curves shifted to zero energy transfer."""
    import matplotlib.pyplot as plt

    matches = np.flatnonzero(np.asarray(state["names"], dtype=str) == str(roi_name))
    if matches.size == 0:
        raise KeyError(roi_name)
    index = int(matches[0])
    fits = state["fits"]
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    figure.subplots_adjust(bottom=0.16, wspace=0.25)

    fit_energy = np.linspace(float(np.min(energy_kev)), float(np.max(energy_kev)), 1000)
    fit_curve = xrsp.lorentzian(
        fit_energy,
        fits["amp"][index],
        fits["center_kev"][index],
        fits["fwhm_ev"][index] / 2000.0,
        fits["bkg"][index],
    )
    axes[0].plot(energy_kev, state["curves"][index], "o", ms=4, label="Data")
    axes[0].plot(fit_energy, fit_curve, lw=1.4, label="Fitted")
    axes[0].set_title(
        f"{roi_name}: center={fits['center_ev'][index]:.3f} eV, "
        f"FWHM={fits['fwhm_ev'][index]:.3f} eV, "
        f"R²={fits['r2'][index]:.4f}"
    )
    axes[0].set_xlabel("Energy (keV)")
    axes[0].set_ylabel("counts")
    axes[0].legend()

    for curve_index, (name, curve) in enumerate(zip(state["names"], state["curves"])):
        transfer = np.asarray(energy_kev, dtype=float) * 1000.0 - fits["center_ev"][curve_index]
        axes[1].plot(transfer, curve, lw=0.8, label=str(name))
    axes[1].axvline(0.0, color="black", ls="--", lw=0.8)
    axes[1].set_title("All elastic curves centered at zero energy transfer")
    axes[1].set_xlabel("Energy Transfer (eV)")
    axes[1].set_ylabel("counts")
    if len(state["names"]) <= 30:
        axes[1].legend(fontsize=6, ncol=3)
    return figure


def _inspect_elastic_fits(ui: UI, state, energy_kev, log) -> None:
    """Prompt for ROI names and display detailed fit diagnostics."""
    if not ui.interactive:
        return
    available = [str(name) for name in state["names"]]
    while True:
        roi_name = ui.ask_optional(
            "ROI crystal name to inspect (press Enter to continue to XRS)",
            help_text="The left panel shows the selected fit; the right panel checks zeroed energy transfer.",
        )
        if roi_name is None:
            return
        if roi_name not in available:
            log(f"[elastic] Unknown ROI {roi_name!r}. Available: {', '.join(available)}")
            continue
        figure = _draw_elastic_inspection(state, energy_kev, roi_name)
        ui.show_figure(figure, f"Elastic fit inspection: {roi_name}", "Close inspection")


# ==========================================================================
# English note.
# ==========================================================================


def _remove_i0_glitches(i0_list, det2D_list, log) -> int:
    """Implementation notes for _remove_i0_glitches."""
    removed = 0
    for i0, det2D in zip(i0_list, det2D_list):
        for index in range(i0.shape[0] - 2):
            if (
                i0[index] != 0
                and i0[index + 2] != 0
                and i0[index + 1] / i0[index] < 0.7
                and i0[index + 1] / i0[index + 2] < 0.7
            ):
                i0[index + 1] = (i0[index] + i0[index + 2]) / 2
                for name in _DATASETS.values():
                    det2D[name][index + 1] = (
                        det2D[name][index] + det2D[name][index + 2]
                    ) / 2
                removed += 1
    if removed:
        log(f"[xrs] Repaired {removed} I0 glitches")
    return removed


def _acquire_xrs(cfg: Config, ui: UI, log) -> None:
    if is_blank(cfg.get("data.root")):
        cfg.set(
            "data.root",
            ui.ask("Data root (data.root)", None,
                   help_text="Example: /hepsdatafs/ID33/202504/Data/GID33-250416-01/"),
        )
    if is_blank(cfg.get("xrs.scan_ids")):
        cfg.set(
            "xrs.scan_ids",
            ui.ask_list("XRS scan IDs (xrs.scan_ids)", None, int, "Example: 14524"),
        )


def run_xrs(cfg: Config, ui: UI, log, force: bool = False, overwrite: bool = False) -> dict:
    _acquire_xrs(cfg, ui, log)
    cfg.save()

    problems = cfg.problems("xrs")
    if problems:
        raise ConfigError(_format_problems("xrs", problems))

    store_dir = cfg.state_dir
    elastic = _load_stage_arrays(store_dir, "elastic")
    scan_ids = [int(item) for item in cfg.get("xrs.scan_ids")]
    divide = bool(cfg.get("xrs.divide_by_i0"))
    i0_pv = cfg.get("xrs.i0_pv")
    remove_glitch = bool(cfg.get("xrs.remove_i0_glitches", True))

    wanted = [_DATASETS[name] for name in _DETECTORS]
    if divide:
        wanted.append(i0_pv)
    log(f"[xrs] Reading scans {scan_ids} from {cfg.raw_dir} ...")
    (_, _, _, mot_data_list, det1D_data_list, det2D_data_list) = xrsp.readH5(
        scan_ids, cfg.raw_dir, useROI=False, detectors=wanted, log=log
    )
    if not det2D_data_list:
        raise ConfigError(f"Scans {scan_ids} contain no 2D detector data")

    energy_pv = _pick_energy_pv(cfg.get("xrs.energy_pv"), mot_data_list[0], log)
    cfg.set("xrs.energy_pv", energy_pv)
    cfg.save()
    energies = [np.asarray(frame[energy_pv], dtype=float) for frame in mot_data_list]

    i0_list = None
    if divide:
        i0_list = []
        for item in det1D_data_list:
            if i0_pv not in item:
                raise ConfigError(
                    f"1D data does not contain {i0_pv!r}; available: {list(item.columns)}"
                )
            i0_list.append(np.abs(np.asarray(item[i0_pv], dtype=float)))

# English note.
    excluded = sorted({int(item) for item in (cfg.get("xrs.exclude_scans") or [])})
    if ui.interactive and i0_list is not None:
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(12, 6))
        try:
            excluded = ui.toggle_scans(
                list(zip(energies, i0_list)),
                excluded,
                "XRS scan exclusion (beam loss or anomalies)",
                figure=figure,
            )
            _save_figure(figure, cfg.state_dir / "figures" / "xrs_i0.png", log)
        finally:
            _close(figure)
        cfg.set("xrs.exclude_scans", excluded)
        cfg.save()
    elif excluded:
        log(f"[xrs] Excluding scans by configuration: {excluded}")

    if remove_glitch and i0_list is not None:
        _remove_i0_glitches(i0_list, det2D_data_list, log)

# English note.
    if i0_list is not None and not ui.interactive:
        _draw_i0_qc(cfg, energies, i0_list, excluded, log)

    keep = [index for index in range(len(det2D_data_list)) if index not in set(excluded)]
    if not keep:
        raise ConfigError(f"All scans were excluded: exclude_scans={excluded}")
    if len(keep) != len(det2D_data_list):
        log(f"[xrs] Keeping {len(keep)}/{len(det2D_data_list)} scans")

    kept_2D = [det2D_data_list[index] for index in keep]
    kept_1D = [det1D_data_list[index] for index in keep] if i0_list is not None else None
    kept_energy = [energies[index] for index in keep]

    lambda_stacks, minipix_stacks = _normalised_stacks(
        kept_2D, kept_1D, i0_pv, divide, log
    )

# English note.
    names = np.asarray(elastic["roi_names"], dtype=str)
    detectors = np.asarray(elastic["roi_detectors"], dtype=str)
    boxes = np.asarray(elastic["boxes"], dtype=np.int64)
    stack_by_detector = {"lambda": lambda_stacks, "minipix": minipix_stacks}
    masks_by_detector = {}
    has_detector_masks = all(
        f"masks_packed_{detector}" in elastic for detector in _DETECTORS
    )
    if has_detector_masks:
        for detector in _DETECTORS:
            expected_shape = tuple(
                int(value) for value in elastic[f"image_shape_{detector}"]
            )
            actual_shape = tuple(stack_by_detector[detector][0].shape[1:])
            if actual_shape != expected_shape:
                raise ConfigError(
                    f"{detector} XRS image shape {actual_shape} differs from its "
                    f"elastic image shape {expected_shape}; ROI geometry cannot be reused"
                )
            masks_by_detector[detector] = _unpack_masks(
                elastic[f"masks_packed_{detector}"],
                expected_shape,
                int(elastic[f"mask_count_{detector}"][0]),
            )
    else:
        expected_shape = tuple(int(value) for value in elastic["image_shape"])
        masks_all = _unpack_masks(
            elastic["masks_packed"], expected_shape, int(elastic["mask_count"][0])
        )
        for detector in _DETECTORS:
            selector = detectors == detector
            actual_shape = tuple(stack_by_detector[detector][0].shape[1:])
            if selector.any() and actual_shape != expected_shape:
                raise ConfigError(
                    "The elastic cache uses the legacy shared-mask format and cannot "
                    f"represent {detector} shape {actual_shape}. Rerun from the elastic "
                    "stage with --force to create detector-specific masks."
                )
            masks_by_detector[detector] = masks_all[selector]

    per_scan_raw, per_scan_masked = [], []
    for index in range(len(kept_2D)):
        raw_parts, masked_parts = [], []
        for detector, stack_source in (("lambda", lambda_stacks), ("minipix", minipix_stacks)):
            selector = detectors == detector
            raw, masked = _roi_spectra(
                stack_source[index], boxes[selector], masks_by_detector[detector]
            )
            raw_parts.append(raw)
            masked_parts.append(masked)
        per_scan_raw.append(np.concatenate(raw_parts, axis=0))
        per_scan_masked.append(np.concatenate(masked_parts, axis=0))

    payload = {
        "roi_names": names,
        "n_scans": np.asarray([len(kept_2D)]),
        "kept_indices": np.asarray(keep, dtype=np.int64),
        "raw_scan_ids": np.asarray(scan_ids, dtype=np.int64),
        "excluded_scans": np.asarray(excluded, dtype=np.int64),
        "energy_pv": np.asarray([energy_pv]),
    }
    for index in range(len(kept_2D)):
        payload[f"energy_{index}"] = np.asarray(kept_energy[index], dtype=float)
        payload[f"sums_{index}"] = per_scan_raw[index]
        payload[f"sums_filtered_{index}"] = per_scan_masked[index]
    if i0_list is not None:
        for index, position in enumerate(keep):
            payload[f"i0_{index}"] = np.asarray(i0_list[position], dtype=float)

    _draw_xrs_qc(cfg, names, detectors, per_scan_raw, per_scan_masked,
                 kept_energy, i0_list, keep, log)
    return payload


def _draw_i0_qc(cfg, energies, i0_list, excluded, log):
    """Implementation notes for _draw_i0_qc."""
    import matplotlib.pyplot as plt

    dropped = set(int(item) for item in excluded)
    figure, axis = plt.subplots(figsize=(12, 6))
    for index, (energy, i0) in enumerate(zip(energies, i0_list)):
        is_out = index in dropped
        axis.plot(
            energy,
            np.abs(i0),
            lw=2.0 if is_out else 0.9,
            alpha=0.35 if is_out else 0.9,
            color="0.6" if is_out else None,
            linestyle="--" if is_out else "-",
            label=f"scan {index}" + ("  [excluded]" if is_out else ""),
        )
    axis.set_yscale("log")
    axis.set_xlabel(f"{cfg.get('xrs.energy_pv')} (keV)")
    axis.set_ylabel(cfg.get("xrs.i0_pv"))
    axis.set_title(f"XRS I0 (excluded: {sorted(dropped) or 'none'})")
    axis.legend(fontsize=8, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    figure.tight_layout()
    _save_figure(figure, cfg.state_dir / "figures" / "xrs_i0.png", log)
    _close(figure)


def _draw_xrs_qc(cfg, names, detectors, raw_list, masked_list, energies, i0_list, keep, log):
    """Implementation notes for _draw_xrs_qc."""
    import matplotlib.pyplot as plt

    count = len(raw_list)
    figure = plt.figure(figsize=(15, 4 * max(count, 1)))
    axes = figure.subplots(max(count, 1), 2, squeeze=False)
    use_filter = bool(cfg.get("elastic.use_filter", True))
    for index in range(count):
        curves = masked_list if use_filter else raw_list
        axes[index][0].plot(energies[index], curves[index].T, lw=0.6)
        axes[index][0].set_title(f"scan {keep[index]}: ROI sums")
        axes[index][0].set_xlabel(f"{cfg.get('xrs.energy_pv')} (keV)")
        axes[index][1].plot(energies[index], curves[index].T, lw=0.6)
        axes[index][1].set_yscale("log")
        axes[index][1].set_title(f"scan {keep[index]}: log scale")
        axes[index][1].set_xlabel(f"{cfg.get('xrs.energy_pv')} (keV)")
    figure.tight_layout()
    _save_figure(figure, cfg.state_dir / "figures" / "xrs_roi_sums.png", log)
    _close(figure)


# ==========================================================================
# English note.
# ==========================================================================


def _load_stage_arrays(state_dir: Path, stage: str) -> dict:
    store = StageStore(state_dir)
    return store.load_arrays(stage)


def run_q(cfg: Config, ui: UI, log, force: bool = False, overwrite: bool = False) -> dict:
    problems = cfg.problems("q")
    if problems and ui.interactive:
        if is_blank(cfg.get("q.module_angles_deg")):
            cfg.set(
                "q.module_angles_deg",
                ui.ask_list(
                    "Module angles q.module_angles_deg (degrees)",
                    _DEFAULT_MODULE_ANGLES,
                    float,
                    f"Required order: {list(_MODULE_ORDER)}",
                ),
            )
            cfg.save()
        problems = cfg.problems("q")
    if problems:
        raise ConfigError(_format_problems("q", problems))

    elastic = _load_stage_arrays(cfg.state_dir, "elastic")
    names = np.asarray(elastic["roi_names"], dtype=str)
    centers_kev = np.asarray(elastic["fit_center_kev"], dtype=float)
    bad_fit = np.asarray(elastic["bad_fit"], dtype=bool)

    source = str(cfg.get("q.elastic_energy_source", "as_before"))
    if source == "nominal":
        elastic_energy = np.full(len(names), _NOMINAL_ELASTIC_ENERGY, dtype=float)
    elif source == "fitted":
        elastic_energy = centers_kev.astype(float, copy=True)
    else:  # English note.
        elastic_energy = np.full(len(names), _NOMINAL_ELASTIC_ENERGY, dtype=float)
        elastic_energy[bad_fit] = centers_kev[bad_fit]

    xrs = _load_stage_arrays(cfg.state_dir, "xrs")
    scan_count = int(xrs["n_scans"][0])
# English note.
    ef = np.asarray(xrs[f"energy_{scan_count - 1}"], dtype=float)

    angles = [np.deg2rad(float(item)) for item in cfg.get("q.module_angles_deg")]
    q_ave = np.zeros(len(names), dtype=float)
    dq_ave = np.zeros(len(names), dtype=float)
    q_range = np.zeros(len(names), dtype=float)
    dq_range = np.zeros(len(names), dtype=float)
    for index, name in enumerate(names):
        _, _, q_ave[index], dq_ave[index], q_range[index], dq_range[index] = xrsp.qCalc(
            angles, str(name), float(elastic_energy[index]), ef
        )
    log(f"[q] Calculated Q for {len(names)} ROIs (Ei source: {source})")
    return {
        "roi_names": names,
        "elastic_energy": elastic_energy,
        "q_ave": q_ave,
        "dq_ave": dq_ave,
        "q_range": q_range,
        "dq_range": dq_range,
    }


# ==========================================================================
# English note.
# ==========================================================================


def _build_roi_table(cfg, elastic, xrs, q) -> pd.DataFrame:
    """Implementation notes for _build_roi_table."""
    names = np.asarray(elastic["roi_names"], dtype=str)
    boxes = np.asarray(elastic["boxes"], dtype=np.int64)
    table = pd.DataFrame(
        {
            "crystal": names,
            "x1": boxes[:, 2],
            "x2": boxes[:, 3],
            "y1": boxes[:, 0],
            "y2": boxes[:, 1],
            "center": np.asarray(elastic["fit_center_kev"]) * 1000.0,
            "width": np.asarray(elastic["fit_fwhm_ev"]),
            "height": np.asarray(elastic["fit_amp"]),
            "background": np.asarray(elastic["fit_bkg"]),
            "r-square": np.asarray(elastic["fit_r2"]),
            "bad_fit": np.asarray(elastic["bad_fit"]),
            "elastic_energy_kev": np.asarray(q["elastic_energy"]),
            "q_ave": np.asarray(q["q_ave"]),
            "dq_ave": np.asarray(q["dq_ave"]),
            "q_range": np.asarray(q["q_range"]),
            "dq_range": np.asarray(q["dq_range"]),
        }
    )
    return table


def _roi_names_of(table: pd.DataFrame) -> np.ndarray:
    """Implementation notes for _roi_names_of."""
    return table["crystal"].to_numpy(dtype=str)


def _inclusion(cfg, table: pd.DataFrame, modules) -> np.ndarray:
    """Implementation notes for _inclusion."""
    names = _roi_names_of(table)
    module_of = np.asarray([name.split("-", maxsplit=1)[0] for name in names])
    excluded = {str(item) for item in (cfg.get("sum.exclude_rois") or [])}
    low, high = (float(v) for v in cfg.get("sum.q_range"))
    include = np.ones(len(names), dtype=bool)
    include &= np.isin(module_of, list(modules))
    include &= ~table["bad_fit"].to_numpy(dtype=bool)
    include &= ~np.isin(names, list(excluded))
    include &= (table["q_ave"].to_numpy(dtype=float) >= low) & (
        table["q_ave"].to_numpy(dtype=float) <= high
    )
    return include


def _interp_grid(energies, centers_ev, step):
    """Implementation notes for _interp_grid."""
    lowers, uppers = [], []
    for energy in energies:
        shifted = np.asarray(energy, dtype=float) * 1000.0
        for center in centers_ev:
            lowers.append(float(np.min(shifted - center)))
            uppers.append(float(np.max(shifted - center)))
    lower = round(max(lowers) / step) * step
    upper = round(min(uppers) / step) * step
    if upper <= lower:
        raise ConfigError(
            f"Shifted ROI energy ranges do not overlap ({lower:.3f} to {upper:.3f} eV); "
            "check elastic fits and energy spacing"
        )
    count = int((upper - lower) / step)
    return lower, upper, np.linspace(lower, upper, count + 1)


def _interpolate_all(cfg, table, xrs, grid, include) -> dict:
    """Implementation notes for _interpolate_all."""
    use_filter = bool(cfg.get("elastic.use_filter", True))
    scan_count = int(xrs["n_scans"][0])
    names = _roi_names_of(table)
    centers_ev = table["center"].to_numpy(dtype=float)
    modules = [str(item) for item in cfg.get("sum.modules")]

    combined = np.zeros_like(grid)
    per_crystal = np.zeros((len(names), grid.size), dtype=float)
    per_module = {module: np.zeros_like(grid) for module in modules}

    for scan in range(scan_count):
        energy = np.asarray(xrs[f"energy_{scan}"], dtype=float) * 1000.0
        spectra = np.asarray(
            xrs[f"sums_filtered_{scan}" if use_filter else f"sums_{scan}"], dtype=float
        )
        for index, name in enumerate(names):
            curve = np.interp(grid, energy - centers_ev[index], spectra[index])
            if include[index]:
                combined += curve
                per_crystal[index] += curve
                module = name.split("-", maxsplit=1)[0]
                if module in per_module:
                    per_module[module] += curve
    return {
        "combined": combined,
        "per_crystal": per_crystal,
        "per_module": per_module,
        "n_scans": scan_count,
    }


def run_sum(cfg: Config, ui: UI, log, force: bool = False, overwrite: bool = False) -> dict:
# English note.
    if is_blank(cfg.get("sum.modules")):
        cfg.set("sum.modules", ["VD"])
    if cfg.get("sum.energy_step_ev") is None:
        cfg.set("sum.energy_step_ev", 0.2)
    if is_blank(cfg.get("sum.q_range")):
        cfg.set("sum.q_range", [0, 10])

    problems = cfg.problems("sum")
    if problems:
        raise ConfigError(_format_problems("sum", problems))

    elastic = _load_stage_arrays(cfg.state_dir, "elastic")
    xrs = _load_stage_arrays(cfg.state_dir, "xrs")
    q = _load_stage_arrays(cfg.state_dir, "q")
    table = _build_roi_table(cfg, elastic, xrs, q)

    energies = [
        np.asarray(xrs[f"energy_{index}"], dtype=float)
        for index in range(int(xrs["n_scans"][0]))
    ]
    centers_ev = table["center"].to_numpy(dtype=float)
    available_modules = sorted({name.split("-", maxsplit=1)[0] for name in _roi_names_of(table)})

    requested = [str(item) for item in (cfg.get("sum.modules") or [])]
    dropped = [item for item in requested if item not in available_modules]
    if dropped:
        log(f"[WARN] Ignoring unavailable modules from sum.modules: {dropped}")
    if not [item for item in requested if item in available_modules]:
        log(f"[WARN] No requested module is available; using {available_modules[:1]}")

    def prepare(values: dict):
        modules = values["modules"]
        if isinstance(modules, str):
            modules = [modules]
        cfg.set("sum.modules", list(modules))
        cfg.set("sum.q_range", [float(values["q_low"]), float(values["q_high"])])
        cfg.set("sum.energy_step_ev", float(values["energy_step_ev"]))
        include = _inclusion(cfg, table, modules)
        _, _, grid = _interp_grid(energies, centers_ev, float(values["energy_step_ev"]))
        result = _interpolate_all(cfg, table, xrs, grid, include)
        result["grid"] = grid
        result["include"] = include
        result["modules"] = list(modules)
        return result

    initial = {
        "modules": [m for m in cfg.get("sum.modules") if m in available_modules] or available_modules[:1],
        "q_low": float(cfg.get("sum.q_range")[0]),
        "q_high": float(cfg.get("sum.q_range")[1]),
        "energy_step_ev": float(cfg.get("sum.energy_step_ev")),
    }
    state = None

    import matplotlib.pyplot as plt

    qc_figure = plt.figure(figsize=(14, 9))
    qc_axes = qc_figure.subplots(2, 1)

    if ui.interactive:
        tune_figure = plt.figure(figsize=(14, 9))
        tune_axes = tune_figure.subplots(2, 1, gridspec_kw={"right": 0.68})

        def draw(values: dict) -> None:
            nonlocal state
            state = prepare(values)
            _draw_sum(tune_axes, state, table)

        draw(initial)
        controls = [
            ChoiceSpec(
                "modules", "Modules to combine", tuple(available_modules),
                value=initial["modules"], multi=True,
            ),
            SliderSpec("q_low", "q lower bound (1/Å)", 0.0, 12.0, initial["q_low"], 0.05),
            SliderSpec("q_high", "q upper bound (1/Å)", 0.0, 12.0, initial["q_high"], 0.05),
            SliderSpec(
                "energy_step_ev", "Energy interpolation step (eV)", 0.05, 2.0,
                initial["energy_step_ev"], 0.05,
            ),
        ]
        values = ui.adjust(
            tune_figure, controls, draw, title="Sum: modules, q range, and interpolation step"
        )
        state = prepare(values)
        _close(tune_figure)
    else:
        state = prepare(initial)

    if state["include"].sum() == 0:
        raise ConfigError(
            "No ROI satisfies the combination criteria; "
            f"modules={state['modules']}, q_range={cfg.get('sum.q_range')}"
        )

    _draw_sum(qc_axes, state, table)
    _save_figure(qc_figure, cfg.state_dir / "figures" / "sum_qc.png", log)
    _close(qc_figure)
    _save_per_crystal_figure(cfg, state, table, log)

    payload = {
        "E_interp": state["grid"],
        "combined": state["combined"],
        "per_crystal": state["per_crystal"],
        "roi_names": _roi_names_of(table),
        "include": state["include"],
        "n_included": np.asarray([int(state["include"].sum())]),
        "n_scans": np.asarray([state["n_scans"]]),
        "modules": np.asarray(state["modules"], dtype=str),
    }
    for module, curve in state["per_module"].items():
        payload[f"module_{module}"] = curve
    log(
        f"[sum] Combined {state['n_scans']} scans x {int(state['include'].sum())} ROIs; "
        f"energy {state['grid'][0]:.2f} to {state['grid'][-1]:.2f} eV"
    )
    return payload


def _draw_sum(axes, state, table) -> None:
    """Implementation notes for _draw_sum."""
    for axis in np.asarray(axes).ravel():
        axis.clear()
    grid = state["grid"]
    axes[0].plot(grid, state["combined"], lw=0.8)
    axes[0].set_title(
        f"Combined result: modules={state['modules']}, "
        f"{int(state['include'].sum())} ROIs x {state['n_scans']} scans"
    )
    axes[0].set_xlabel("Energy Transfer (eV)")
    axes[0].set_ylabel("Intensity")
    for module, curve in state["per_module"].items():
        axes[1].plot(grid, curve, lw=0.8, label=str(module))
    axes[1].set_title("By module")
    axes[1].set_xlabel("Energy Transfer (eV)")
    axes[1].legend(fontsize=8)
# English note.
# English note.


def _save_per_crystal_figure(cfg, state, table, log) -> None:
    """Implementation notes for _save_per_crystal_figure."""
    import matplotlib.pyplot as plt

    names = _roi_names_of(table)
    include = state["include"]
    chosen = np.nonzero(include)[0]
    if chosen.size == 0:
        return
    limit = 40
    shown = chosen[:limit]
    columns = 5
    rows = max(1, int(np.ceil(shown.size / columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(25, 4 * rows), squeeze=False)
    grid = state["grid"]
    for slot, index in enumerate(shown):
        axis = axes[slot // columns][slot % columns]
        axis.plot(grid, state["per_crystal"][index], lw=0.8)
        axis.set_title(
            f"{names[index]}  q={table['q_ave'].to_numpy()[index]:.3f}", fontsize=10
        )
    for slot in range(shown.size, rows * columns):
        figure.delaxes(axes[slot // columns][slot % columns])
    if chosen.size > limit:
        figure.suptitle(f"Showing the first {limit} of {chosen.size} selected crystals", fontsize=14)
    figure.tight_layout()
    _save_figure(figure, cfg.state_dir / "figures" / "sum_per_crystal.png", log)
    _close(figure)


# ==========================================================================
# English note.
# ==========================================================================


def run_save(cfg: Config, ui: UI, log, force: bool = False, overwrite: bool = False) -> dict:
    if is_blank(cfg.get("output.filename")):
        cfg.set(
            "output.filename",
            ui.ask("Output directory name (output.filename)", None, help_text="Example: NiO_20260518"),
        )
        cfg.save()
    problems = cfg.problems("save")
    if problems:
        raise ConfigError(_format_problems("save", problems))

    filename = str(cfg.get("output.filename"))
    mode = str(cfg.get("output.mode", "combined"))
    out_dir = cfg.processed_dir / filename

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise ConfigError(
            f"Output directory already exists and is not empty: {out_dir}\n"
            "Choose another output.filename or add --overwrite."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    elastic = _load_stage_arrays(cfg.state_dir, "elastic")
    xrs = _load_stage_arrays(cfg.state_dir, "xrs")
    q = _load_stage_arrays(cfg.state_dir, "q")
    total = _load_stage_arrays(cfg.state_dir, "sum")
    table = _build_roi_table(cfg, elastic, xrs, q)

    grid = np.asarray(total["E_interp"], dtype=float)
    names = np.asarray(total["roi_names"], dtype=str)
    written = []

    def write_frame(frame: pd.DataFrame, path: Path) -> None:
        if path.exists() and not overwrite:
            raise ConfigError(f"File already exists: {path} (use --overwrite)")
        frame.to_csv(path, sep="\t", index=False)
        written.append(path)
        log(f"[save] {path}")

    if mode == "combined":
        frame = pd.DataFrame(
            {"Energy Transfer (eV)": grid, "Intensity": np.asarray(total["combined"])}
        )
        write_frame(frame, out_dir / f"{filename}_data.txt")
    elif mode == "per_module":
        frame = pd.DataFrame({"Energy Transfer (eV)": grid})
        for key in total:
            if key.startswith("module_"):
                frame[key[len("module_"):]] = np.asarray(total[key])
        write_frame(frame, out_dir / f"{filename}_data.txt")
    else:  # per_crystal
        for index, name in enumerate(names):
            if not bool(np.asarray(total["include"])[index]):
                continue
            frame = pd.DataFrame(
                {
                    "Energy Transfer (eV)": grid,
                    "Intensity": np.asarray(total["per_crystal"])[index],
                }
            )
            safe = str(name).replace("/", "_").replace("\\", "_")
            write_frame(frame, out_dir / f"{safe}_data.txt")

    write_frame(table, out_dir / f"{filename}_rois.txt")

    info = out_dir / f"{filename}_info.txt"
    if info.exists() and not overwrite:
        raise ConfigError(f"File already exists: {info} (use --overwrite)")
    info.write_text(_info_text(cfg, table, total), encoding="utf-8")
    written.append(info)
    log(f"[save] {info}")

    return {
        "output_dir": np.asarray([str(out_dir)]),
        "files": np.asarray([str(path) for path in written], dtype=str),
        "mode": np.asarray([mode]),
    }


def _info_text(cfg: Config, table: pd.DataFrame, total: dict) -> str:
    """Implementation notes for _info_text."""
    lines = [
        f"data.root = {cfg.get('data.root')}",
        f"roi_mode = {cfg.get('elastic.roi_mode')}",
        f"use_filter = {cfg.get('elastic.use_filter')}",
        f"auto_adjust = {cfg.get('elastic.auto_adjust')}",
        f"split: divide_by_i0 (elastic) = {cfg.get('elastic.divide_by_i0')}",
        f"scan_ids (elastic) = {cfg.get('elastic.scan_ids')}",
        f"energy_pv (elastic) = {cfg.get('elastic.energy_pv')}",
        f"i0_pv (elastic) = {cfg.get('elastic.i0_pv')}",
        f"regular_files = {cfg.get('elastic.regular_files')}",
        f"auto_centers = {cfg.get('elastic.auto_centers')}",
        f"auto_params = {cfg.get('elastic.auto_params')}",
        f"filter_value = {cfg.get('elastic.filter_value')}",
        f"roi_size = {cfg.get('elastic.roi_size')}",
        f"fit = {cfg.get('elastic.fit')}",
        f"divide_by_i0 (xrs) = {cfg.get('xrs.divide_by_i0')}",
        f"i0_pv (xrs) = {cfg.get('xrs.i0_pv')}",
        f"scan_ids (xrs) = {cfg.get('xrs.scan_ids')}",
        f"energy_pv (xrs) = {cfg.get('xrs.energy_pv')}",
        f"exclude_scans = {cfg.get('xrs.exclude_scans')}",
        f"remove_i0_glitches = {cfg.get('xrs.remove_i0_glitches')}",
        f"module_angles_deg = {cfg.get('q.module_angles_deg')}",
        f"elastic_energy_source = {cfg.get('q.elastic_energy_source')}",
        f"modules = {cfg.get('sum.modules')}",
        f"q_range = {cfg.get('sum.q_range')}",
        f"exclude_rois = {cfg.get('sum.exclude_rois')}",
        f"energy_step_ev = {cfg.get('sum.energy_step_ev')}",
        f"output_mode = {cfg.get('output.mode')}",
        f"Number of ROIs = {len(table)}",
        f"Number of ROIs included = {int(np.asarray(total['n_included'])[0])}",
        f"Number of scans summed = {int(np.asarray(total['n_scans'])[0])}",
        f"Bad fit ROIs = {sorted(table.loc[table['bad_fit'].astype(bool), 'crystal'].astype(str))}",
    ]
    return "\n".join(lines) + "\n"


# ==========================================================================
# English note.
# ==========================================================================

_RUNNERS = {
    "elastic": run_elastic,
    "xrs": run_xrs,
    "q": run_q,
    "sum": run_sum,
    "save": run_save,
}


def run_stage(
    cfg: Config,
    stage: str,
    ui: UI,
    log=print,
    force: bool = False,
    overwrite: bool = False,
) -> dict:
    """Implementation notes for run_stage."""
    if stage not in STAGE_ORDER:
        raise ConfigError(f"Unknown stage {stage!r}; available: {list(STAGE_ORDER)}")

    configure_matplotlib(int(cfg.get("ui.dpi", 150) or 150))
    store = StageStore(cfg.state_dir)

    if not force and store.is_current(stage, cfg.fingerprint(stage)):
        log(f"[skip] {stage} is current: {store.npz_path(stage)}")
        return store.load_arrays(stage)

    if force:
        store.reset_from(stage)

    log(f"===== Stage {stage}: {_stage_title(stage)} =====")
    log(f"[fingerprint] {cfg.fingerprint(stage)}")

    try:
        payload = _RUNNERS[stage](cfg, ui, log, force=force, overwrite=overwrite)
    except (ValueError, KeyError) as exc:
# English note.
        raise ConfigError(f"Stage {stage} failed: {exc}") from exc

    store.save_arrays(stage, payload)
# English note.
# English note.
# English note.
    if cfg.save():
        log(f"[config] Updated {cfg.path}")
    store.record(stage, cfg.fingerprint(stage))
    log(f"[done] {stage} → {store.npz_path(stage)}")
    return payload


def _stage_title(stage: str) -> str:
    from xrs_config import STAGE_TITLE

    return STAGE_TITLE.get(stage, stage)


# ==========================================================================
# English note.
# ==========================================================================


def load_elastic_images(cfg: Config, log=print) -> dict:
    """Sum every configured elastic scan into one image per detector."""
    if is_blank(cfg.get("data.root")):
        raise ConfigError("data.root is empty; detector images cannot be loaded")
    scan_ids = [int(item) for item in (cfg.get("elastic.scan_ids") or [])]
    if not scan_ids:
        raise ConfigError("elastic.scan_ids is empty; detector images cannot be loaded")

    divide = bool(cfg.get("elastic.divide_by_i0"))
    i0_pv = cfg.get("elastic.i0_pv")
    wanted = [_DATASETS[name] for name in _DETECTORS]
    if divide and not is_blank(i0_pv):
        wanted.append(i0_pv)
    log(f"[ROI] Reading elastic scans {scan_ids} for geometry calibration ...")
    (_, _, _, _, det1D_data_list, det2D_data_list) = xrsp.readH5(
        scan_ids, cfg.raw_dir, useROI=False, detectors=wanted, log=log
    )
    if not det2D_data_list:
        raise ConfigError(f"Scans {scan_ids} contain no 2D detector data")
    lambda_stacks, minipix_stacks = _normalised_stacks(
        det2D_data_list, det1D_data_list, i0_pv, divide, log
    )
    return {
        "lambda": np.concatenate(lambda_stacks, axis=0).sum(axis=0),
        "minipix": np.concatenate(minipix_stacks, axis=0).sum(axis=0),
    }


def save_centers_overlay(image, centers, detector, path: Path, log) -> None:
    """Save a center-point overlay for visual quality control."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 11))
    axis.imshow(np.log1p(np.clip(np.asarray(image, dtype=float), 0, None)), cmap="gray")
    for label, (x, y) in centers.items():
        axis.plot(x, y, "o", color="#ff3030", ms=4)
        axis.text(x + 5, y - 5, str(label), color="#ff3030", fontsize=7)
    axis.set_title(f"{_LABELS[detector]}: {len(centers)} centers")
    _save_figure(figure, path, log)
    _close(figure)


def centers_file_for(cfg: Config, detector: str) -> Path:
    return _auto_target(cfg, detector)


def _load_centers_if_present(path: Path):
    if path and Path(path).is_file():
        return xrsp.load_auto_roi_centers(path)
    return None


def run_pick_roi(cfg: Config, detector: str, ui: UI, write: bool, log=print) -> dict:
    """Pick centers, segment ROIs, preview them, and write HDF5 geometry."""
    import auto_roi

    images = load_elastic_images(cfg, log)
    if detector not in images:
        raise ConfigError(f"Unknown detector {detector!r}; available: {list(images)}")
    labels = auto_roi.expected_labels(detector)
    target = centers_file_for(cfg, detector)
    centers = None
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(11, 11))
    try:
        centers = ui.pick_points(
            images[detector], labels, f"{_LABELS[detector]} ROI centers", figure=figure
        )
    finally:
        _close(figure)
    if not centers:
        raise ConfigError("No ROI centers were selected")
    parameters = dict(cfg.get("elastic.auto_params", {}) or {})
    result = auto_roi.detect_all_rois(
        images[detector], centers, auto_roi.DETECTOR_CONFIG[detector]["radius"],
        smooth_sigma=float(parameters.get("smooth_sigma", 2.0)),
        threshold_tightness=float(parameters.get("threshold_tightness", 1.0)),
        min_area=int(parameters.get("min_area", 20)), log=log,
    )
    if not len(result["roi_labels"]):
        raise ConfigError("No automatic ROI could be segmented")
    if not ui.review_geometry(result["label_image"], result["roi_labels"],
                              _result_bounds(result), f"{_LABELS[detector]} generated ROIs"):
        raise UiCancelled("Automatic ROI selection must be repeated")
    _backup(target, log)
    written = xrsp.write_auto_roi_hdf5(
        target, result, detector=detector,
        scan_ids=[int(item) for item in cfg.get("elastic.scan_ids")],
        image_shape=images[detector].shape, parameters=parameters,
    )
    cfg.save()
    save_centers_overlay(
        images[detector], centers, detector,
        cfg.state_dir / "figures" / f"centers_{detector}.png", log,
    )
    return {"path": str(written), "count": len(result["roi_labels"])}


def run_propose_roi(cfg: Config, detector: str, write: bool, log=print) -> dict:
    """Propose centers and write candidate or official HDF5 ROI geometry."""
    import auto_roi

    images = load_elastic_images(cfg, log)
    if detector not in images:
        raise ConfigError(f"Unknown detector {detector!r}; available: {list(images)}")
    target = centers_file_for(cfg, detector)
    legacy = cfg.get(f"elastic.auto_centers.{detector}")
    template = _load_centers_if_present(cfg.resolve(legacy)) if not is_blank(legacy) else None
    if target.is_file():
        stored = xrsp.load_auto_roi_hdf5(target, detector=detector,
                                         image_shape=images[detector].shape)
        template = {
            str(label): tuple(int(value) for value in point)
            for label, point in zip(stored["roi_labels"], stored["centers_xy"])
        }

    centers, report = auto_roi.propose_centers(images[detector], detector, template=template)
    log(f"[ROI] Proposal mode: {report['mode']}; found {report['num_peaks']} peaks; "
        f"resolved {len(report['resolved'])}/{report['num_expected']} labels")
    if report["missing"]:
        log(f"[WARN] Unresolved labels ({len(report['missing'])}): {report['missing']}")
    if not centers:
        raise ConfigError("No centers were proposed; use pick-roi for manual selection")

    parameters = dict(cfg.get("elastic.auto_params", {}) or {})
    result = auto_roi.detect_all_rois(
        images[detector], centers, auto_roi.DETECTOR_CONFIG[detector]["radius"],
        smooth_sigma=float(parameters.get("smooth_sigma", 2.0)),
        threshold_tightness=float(parameters.get("threshold_tightness", 1.0)),
        min_area=int(parameters.get("min_area", 20)), log=log,
    )
    if not len(result["roi_labels"]):
        raise ConfigError("The proposal did not produce any segmented ROI")
    if write:
        _backup(target, log)
        written = target
    else:
        written = target.with_name(target.stem + ".proposed" + target.suffix)
    xrsp.write_auto_roi_hdf5(
        written, result, detector=detector,
        scan_ids=[int(item) for item in cfg.get("elastic.scan_ids")],
        image_shape=images[detector].shape, parameters=parameters,
    )
    if write:
        cfg.save()
    log(f"[ROI] Wrote {'official' if write else 'candidate'} geometry to {written}")

    save_centers_overlay(
        images[detector], centers, detector,
        cfg.state_dir / "figures" / f"centers_{detector}_proposed.png", log,
    )
    return {"path": str(written), "count": len(centers), "mode": report["mode"]}


def _commit_centers(target: Path, centers, labels, write: bool, log) -> Path:
    """Write a legacy center file and back up an existing target."""
    target = Path(target)
    if target.is_file() and write:
        backup = target.with_suffix(target.suffix + ".bak")
        backup.write_bytes(target.read_bytes())
        log(f"[ROI] Backed up the original file to {backup}")
    count = xrsp.write_auto_roi_centers(target, centers, labels=labels)
    log(f"[ROI] Wrote {count} centers to {target}")
    return target
