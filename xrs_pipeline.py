"""五个阶段的实现：``elastic`` → ``xrs`` → ``q`` → ``sum`` → ``save``。

每个阶段独立可跑、可重复跑：

1. 先用 :meth:`xrs_config.Config.problems` 看缺哪些参数；
2. 缺的就在 :class:`xrs_ui.UI` 里问（终端问答或弹出 matplotlib 窗口）；
3. 算完把中间结果写进 ``<processed>/.xrs_state/``，QC 图写进 ``figures/``；
4. 参数一旦被改动，:meth:`xrs_config.Config.save` 立刻回写 ``config.yaml``。

阶段间只通过状态目录里的数组传递数据，所以改了 ``sum`` 的参数之后
重跑不会再碰 HDF5。
"""

from __future__ import annotations

import json
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

#: 能量 PV 的探测顺序。原 notebook 里那段 ``try/except`` 其实永远走 except
#: （它在 readH5 赋值之前就访问 mot_data_list0），所以这里改成真探测。
_ENERGY_PV_CANDIDATES = (
    "M_DCM_B5Energy_readback",
    "M_DCM_B5Link_Energy_readback",
)

_NOMINAL_ELASTIC_ENERGY = 9.685  # keV，XRS_roi 的构造默认值
_DEFAULT_MODULE_ANGLES = [145.6, 79.03, 25, 118.8, 58.85, 67.862]
_MODULE_ORDER = ("VB", "VU", "VD", "HB", "HL", "HR")


# ==========================================================================
# 状态目录
# ==========================================================================


class StageStore:
    """``<processed>/.xrs_state/`` 的读写。

    ``meta.json`` 记录每个阶段完成时的参数指纹，用来判断"产物还在，
    但参数已经变了"。数组走 npz，表格走 tsv。
    """

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

    # -- 数组/表格 ----------------------------------------------------------

    def npz_path(self, stage: str) -> Path:
        return self.dir / f"{stage}.npz"

    def save_arrays(self, stage: str, data: dict) -> None:
        np.savez_compressed(self.npz_path(stage), **data)

    def load_arrays(self, stage: str) -> dict:
        path = self.npz_path(stage)
        if not path.is_file():
            raise ConfigError(
                f"缺少 {stage} 阶段的中间结果 {path}；请先运行 --stage {stage}"
            )
        with np.load(path, allow_pickle=False) as handle:
            return {name: handle[name] for name in handle.files}

    def save_frame(self, stage: str, name: str, frame: pd.DataFrame) -> None:
        frame.to_csv(self.dir / f"{stage}_{name}.tsv", sep="\t", index=False)

    def load_frame(self, stage: str, name: str) -> pd.DataFrame:
        path = self.dir / f"{stage}_{name}.tsv"
        if not path.is_file():
            raise ConfigError(f"缺少 {stage} 阶段的表格 {path}；请先运行 --stage {stage}")
        return pd.read_csv(path, sep="\t")

    # -- 指纹 ---------------------------------------------------------------

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
        """把 ``stage`` 及其下游记录清掉（``--force`` 或上游变更时用）。"""
        start = STAGE_ORDER.index(stage)
        for name in STAGE_ORDER[start:]:
            self._meta["stages"].pop(name, None)
            self.npz_path(name).unlink(missing_ok=True)
        self.flush()


# ==========================================================================
# 小工具
# ==========================================================================


def _format_problems(stage: str, problems) -> str:
    lines = [f"{stage} 阶段还缺 {len(problems)} 项参数："]
    lines += [f"  - {item}" for item in problems]
    lines.append("")
    lines.append("交互模式下会逐个询问；用 --no-ui 时请先手工填好 config.yaml。")
    return "\n".join(lines)


def _pick_energy_pv(configured, motor_frame: pd.DataFrame, log) -> str:
    """确定能量 PV：配置优先，否则按候选列表探测。"""
    columns = list(motor_frame.columns)
    if not is_blank(configured):
        if configured in columns:
            return str(configured)
        raise ConfigError(
            f"配置的 energy_pv = {configured!r} 不在电机数据里。"
            f"可用：{columns}"
        )
    for candidate in _ENERGY_PV_CANDIDATES:
        if candidate in columns:
            log(f"[INFO] 自动选用能量 PV：{candidate}")
            return candidate
    raise ConfigError(
        f"无法自动识别能量 PV，候选 {_ENERGY_PV_CANDIDATES} 都不在电机数据里。"
        f"可用：{columns}"
    )


def _normalised_stacks(det2D_list, det1D_list, i0_pv, divide, log):
    """按 I0 归一化（或原样返回）两个探测器的帧栈。"""
    if not divide:
        log("[INFO] divide_by_i0 = false，不做 I0 归一化")
        return (
            [np.asarray(item[_DATASETS["lambda"]], dtype=float) for item in det2D_list],
            [np.asarray(item[_DATASETS["minipix"]], dtype=float) for item in det2D_list],
        )

    if det1D_list is None:
        raise ConfigError("divide_by_i0 = true 但没有传入 1D 数据（I0）")

    i0_list = []
    for item in det1D_list:
        if i0_pv not in item:
            raise ConfigError(
                f"1D 数据里没有 {i0_pv!r}，无法做 I0 归一化；可用：{list(item.columns)}"
            )
        i0_list.append(np.abs(np.asarray(item[i0_pv], dtype=float)))

    def scale(stack, i0):
        if stack.shape[0] != i0.shape[0]:
            raise ConfigError(
                f"探测器帧数 {stack.shape[0]} 与 I0 点数 {i0.shape[0]} 不一致"
            )
        return stack / i0[:, None, None]

    return (
        [scale(np.asarray(item[_DATASETS["lambda"]], dtype=float), i0)
         for item, i0 in zip(det2D_list, i0_list)],
        [scale(np.asarray(item[_DATASETS["minipix"]], dtype=float), i0)
         for item, i0 in zip(det2D_list, i0_list)],
    )


def _roi_spectra(stack, boxes, masks):
    """按 ROI 切片求和。返回 ``(原始和, 掩膜过滤后的和)``，形状 ``(n_roi, n_frames)``。

    这就是 notebook 里反复出现的那段 ``roiStack.sum(axis=(1, 2))`` /
    ``(roiStack * mask).sum(axis=(1, 2))``。
    """
    raw, masked = [], []
    for (y1, y2, x1, x2), mask in zip(boxes, masks):
        roi_stack = stack[:, y1:y2, x1:x2]
        raw.append(roi_stack.sum(axis=(1, 2)))
        masked.append((roi_stack * mask[y1:y2, x1:x2]).sum(axis=(1, 2)))
    return np.asarray(raw, dtype=float), np.asarray(masked, dtype=float)


def _lorentzian_fit(x_values, y_values, fit_type):
    """把 notebook cell 18 里那个局部函数搬出来，异常类型写全。"""
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


def _pack_masks(masks) -> np.ndarray:
    """把布尔掩膜打包成位，避免 60 个 ROI × 整幅图把状态文件撑大。"""
    stacked = np.asarray(masks, dtype=bool)
    flat = stacked.reshape(stacked.shape[0], -1)
    return np.packbits(flat, axis=1)


def _unpack_masks(packed, shape, count) -> np.ndarray:
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1)
    height, width = int(shape[0]), int(shape[1])
    return bits[:, : height * width].reshape(count, height, width).astype(bool)


# ==========================================================================
# elastic 阶段
# ==========================================================================


def _acquire_elastic(cfg: Config, ui: UI, log) -> None:
    """把结构性参数补齐（路径、扫描号、ROI 模式、ROI 文件）。"""
    if is_blank(cfg.get("data.root")):
        cfg.set(
            "data.root",
            ui.ask(
                "数据根目录 data.root",
                None,
                help_text="例：/hepsdatafs/ID33/202504/Data/GID33-250416-01/",
            ),
        )
    if is_blank(cfg.get("elastic.scan_ids")):
        cfg.set(
            "elastic.scan_ids",
            ui.ask_list("弹性峰扫描编号 elastic.scan_ids", None, int, "例：14522"),
        )
    if is_blank(cfg.get("elastic.roi_mode")):
        cfg.set(
            "elastic.roi_mode",
            ui.ask(
                "ROI 模式 elastic.roi_mode",
                None,
                choices=("regular", "auto"),
                help_text="regular=读矩形 ROI 文件；auto=读中心点文件后自动分割",
            ),
        )
    mode = cfg.get("elastic.roi_mode")
    group = "regular_files" if mode == "regular" else "auto_centers"
    noun = "矩形 ROI 文件" if mode == "regular" else "中心点文件"
    for detector in _DETECTORS:
        key = f"elastic.{group}.{detector}"
        if is_blank(cfg.get(key)):
            default = f"roi/{'roi' if mode == 'regular' else 'auto_ROI'}_{detector}.txt"
            cfg.set(key, ui.ask(f"{_LABELS[detector]} 的{noun} {key}", default))


def _elastic_paths(cfg: Config, mode: str) -> dict:
    group = "regular_files" if mode == "regular" else "auto_centers"
    return {
        detector: str(cfg.resolve(cfg.get(f"elastic.{group}.{detector}")))
        for detector in _DETECTORS
    }


def _elastic_controls(cfg: Config) -> list:
    controls = [
        SliderSpec(
            "filter_value",
            "filter_value（ROI 内阈值系数）",
            0.0,
            1.0,
            float(cfg.get("elastic.filter_value", 0.15)),
            0.01,
        ),
        SliderSpec(
            "e_lowlim",
            "拟合中心下限 (keV)",
            9.0,
            13.5,
            float(cfg.get("elastic.fit.e_lowlim", 9.67)),
            0.001,
        ),
        SliderSpec(
            "e_highlim",
            "拟合中心上限 (keV)",
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
                "roi_size（自动调整后的边长）",
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

    problems = cfg.problems("elastic")
    if problems:
        raise ConfigError(_format_problems("elastic", problems))

    scan_ids = [int(item) for item in cfg.get("elastic.scan_ids")]
    if len(scan_ids) != 1:
        log(f"[WARN] elastic 阶段只使用第一个扫描 {scan_ids[0]}，其余被忽略")
    divide = bool(cfg.get("elastic.divide_by_i0"))
    i0_pv = cfg.get("elastic.i0_pv")
    mode = cfg.get("elastic.roi_mode")

    wanted = [_DATASETS[name] for name in _DETECTORS]
    if divide:
        wanted.append(i0_pv)
    log(f"[elastic] 读取 {cfg.raw_dir} 下的扫描 {scan_ids[0]} …")
    (_, _, _, mot_data_list, det1D_data_list, det2D_data_list) = xrsp.readH5(
        scan_ids, cfg.raw_dir, useROI=False, detectors=wanted, log=log
    )
    if not det2D_data_list:
        raise ConfigError(f"扫描 {scan_ids[0]} 没读到任何 2D 探测器数据")

    energy_pv = _pick_energy_pv(cfg.get("elastic.energy_pv"), mot_data_list[0], log)
    cfg.set("elastic.energy_pv", energy_pv)
    energy = np.asarray(mot_data_list[0][energy_pv], dtype=float)

    lambda_stacks, minipix_stacks = _normalised_stacks(
        det2D_data_list, det1D_data_list, i0_pv, divide, log
    )
    stacks = {"lambda": lambda_stacks[0], "minipix": minipix_stacks[0]}
    grid = {name: stacks[name].shape[1:] for name in _DETECTORS}
    log(f"[elastic] 探测器图像尺寸：{grid}")

    paths = _elastic_paths(cfg, mode)
    fit_cfg = cfg.get("elastic.fit", {})
    auto_params = cfg.get("elastic.auto_params", {})
    use_filter = bool(cfg.get("elastic.use_filter", True))
    source_label = f"{scan_ids[0]}_elastic"

    def build(values: dict):
        """按当前参数重建 ROI workflow。调参窗口每次拖动都会调它。"""
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
        """重建 ROI 并按当前拟合窗口重算弹性峰曲线与拟合结果。"""
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

    # QC 图单独一张，保证存盘时不会把调参控件也画进去
    qc_figure = plt.figure(figsize=(16, 9))
    qc_axes = qc_figure.subplots(2, 3, squeeze=False)

    if ui.interactive:
        # 调参窗口把右侧让给控件；draw 只清空自己那几根轴，
        # 绝不能 figure.clear()——那会把 adjust 刚建好的控件轴一起抹掉。
        tune_figure = plt.figure(figsize=(16, 9))
        tune_axes = tune_figure.subplots(
            2, 3, gridspec_kw={"right": 0.68}, squeeze=False
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
            title="elastic：拖滑杆看 ROI 与拟合效果",
        )
        state = prepare(values)
        _close(tune_figure)
    else:
        state = prepare(values)

    _draw_elastic(qc_axes, state, energy, cfg)
    _save_figure(qc_figure, cfg.state_dir / "figures" / "elastic_qc.png", log)
    _close(qc_figure)

    # 参数定稿后回写
    cfg.set("elastic.filter_value", float(values["filter_value"]))
    if "roi_size" in values:
        cfg.set("elastic.roi_size", int(values["roi_size"]))
    cfg.set("elastic.fit.e_lowlim", float(values["e_lowlim"]))
    cfg.set("elastic.fit.e_highlim", float(values["e_highlim"]))
    cfg.save()

    fits = state["fits"]
    payload = {
        "roi_names": state["names"],
        "roi_detectors": state["detectors"],
        "boxes": np.asarray(state["boxes"], dtype=np.int64),
        "masks_packed": _pack_masks(state["masks"]),
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
    return payload


def _fit_all(names, energy_kev, curves, fit_cfg, log) -> tuple[dict, np.ndarray]:
    """逐 ROI 拟合弹性峰，返回系数表与 bad-fit 掩码。"""
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
    log(f"[elastic] 拟合完成：{len(names)} 个 ROI，其中 {len(bad_names)} 个 bad fit")
    if bad_names:
        log(f"[elastic] bad fit：{bad_names}")
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
    """elastic 阶段的 QC 图：mask 叠加 + 各 ROI 曲线 + 拟合中心。

    ``axes`` 由调用方建好（2x3）。这里只清空这些轴，不动整张 figure——
    调参窗口的控件轴和它共用一个 figure。
    """
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
    axis.plot(energy, state["curves"].T, lw=0.6)
    axis.set_title("elastic ROI sums（每行一个 ROI）")
    axis.set_xlabel(f"{cfg.get('elastic.energy_pv')} (keV)")
    axis.set_ylabel("counts")

    axis = axes[1][0]
    orders = np.argsort(state["fits"]["center_ev"])
    axis.plot(orders, state["fits"]["center_ev"][orders], "o-", ms=3)
    axis.axhspan(
        float(cfg.get("elastic.fit.e_lowlim", 9.67)) * 1000,
        float(cfg.get("elastic.fit.e_highlim", 9.69)) * 1000,
        color="green", alpha=0.15, label="允许窗口",
    )
    bad_idx = np.nonzero(state["bad"])[0]
    axis.plot(bad_idx, state["fits"]["center_ev"][bad_idx], "rx", ms=8, label="bad fit")
    axis.set_title(f"拟合中心（bad fit {len(bad_idx)} 个）")
    axis.set_xlabel("ROI 序号")
    axis.set_ylabel("center (eV)")
    axis.legend(fontsize=8)

    axis = axes[1][1]
    axis.plot(np.arange(len(state["fits"]["r2"])), state["fits"]["r2"], "o-", ms=3)
    axis.axhline(
        float(cfg.get("elastic.fit.min_r_squared", 0.8)),
        color="red", ls="--", lw=1, label="R² 下限",
    )
    axis.set_title("拟合 R²")
    axis.set_xlabel("ROI 序号")
    axis.legend(fontsize=8)

    axis = axes[1][2]
    axis.plot(np.arange(len(state["fits"]["fwhm_ev"])), state["fits"]["fwhm_ev"], "o-", ms=3)
    axis.axhline(
        float(cfg.get("elastic.fit.max_fwhm_ev", 2.0)),
        color="red", ls="--", lw=1, label="FWHM 上限",
    )
    axis.set_title("拟合 FWHM (eV)")
    axis.set_xlabel("ROI 序号")
    axis.legend(fontsize=8)


# ==========================================================================
# xrs 阶段
# ==========================================================================


def _remove_i0_glitches(i0_list, det2D_list, log) -> int:
    """去掉 I0 上的"掉光"凹点，同步用相邻均值补上对应帧的探测器数据。

    逻辑与 notebook cell 22 一致：某点同时低于左右邻居的 0.7 倍即判为 glitch。
    """
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
        log(f"[xrs] 去掉 {removed} 个 I0 glitch")
    return removed


def _acquire_xrs(cfg: Config, ui: UI, log) -> None:
    if is_blank(cfg.get("data.root")):
        cfg.set(
            "data.root",
            ui.ask("数据根目录 data.root", None,
                   help_text="例：/hepsdatafs/ID33/202504/Data/GID33-250416-01/"),
        )
    if is_blank(cfg.get("xrs.scan_ids")):
        cfg.set(
            "xrs.scan_ids",
            ui.ask_list("XRS 扫描编号 xrs.scan_ids", None, int, "例：14524"),
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
    log(f"[xrs] 读取 {cfg.raw_dir} 下的扫描 {scan_ids} …")
    (_, _, _, mot_data_list, det1D_data_list, det2D_data_list) = xrsp.readH5(
        scan_ids, cfg.raw_dir, useROI=False, detectors=wanted, log=log
    )
    if not det2D_data_list:
        raise ConfigError(f"扫描 {scan_ids} 没读到任何 2D 探测器数据")

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
                    f"1D 数据里没有 {i0_pv!r}，无法做 I0 归一化；可用：{list(item.columns)}"
                )
            i0_list.append(np.abs(np.asarray(item[i0_pv], dtype=float)))

    # 剔除坏扫描：交互模式下点 I0 曲线切换，否则读配置
    excluded = sorted({int(item) for item in (cfg.get("xrs.exclude_scans") or [])})
    if ui.interactive and i0_list is not None:
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(12, 6))
        try:
            excluded = ui.toggle_scans(
                list(zip(energies, i0_list)),
                excluded,
                "xrs：剔除坏扫描（掉光/异常）",
                figure=figure,
            )
            _save_figure(figure, cfg.state_dir / "figures" / "xrs_i0.png", log)
        finally:
            _close(figure)
        cfg.set("xrs.exclude_scans", excluded)
        cfg.save()
    elif excluded:
        log(f"[xrs] 按配置剔除扫描 {excluded}")

    if remove_glitch and i0_list is not None:
        _remove_i0_glitches(i0_list, det2D_data_list, log)

    # --no-ui 时也要出 I0 QC 图：无头运行唯一的核对依据
    if i0_list is not None and not ui.interactive:
        _draw_i0_qc(cfg, energies, i0_list, excluded, log)

    keep = [index for index in range(len(det2D_data_list)) if index not in set(excluded)]
    if not keep:
        raise ConfigError(f"所有扫描都被剔除了：exclude_scans={excluded}")
    if len(keep) != len(det2D_data_list):
        log(f"[xrs] 保留 {len(keep)}/{len(det2D_data_list)} 个扫描")

    kept_2D = [det2D_data_list[index] for index in keep]
    kept_1D = [det1D_data_list[index] for index in keep] if i0_list is not None else None
    kept_energy = [energies[index] for index in keep]

    lambda_stacks, minipix_stacks = _normalised_stacks(
        kept_2D, kept_1D, i0_pv, divide, log
    )

    # 用弹性阶段的 ROI（几何 + 掩膜）来切 XRS 数据
    names = np.asarray(elastic["roi_names"], dtype=str)
    detectors = np.asarray(elastic["roi_detectors"], dtype=str)
    boxes = np.asarray(elastic["boxes"], dtype=np.int64)
    expected_shape = tuple(int(v) for v in elastic["image_shape"])
    masks_all = _unpack_masks(
        elastic["masks_packed"], expected_shape, int(elastic["mask_count"][0])
    )

    for detector in _DETECTORS:
        actual = {"lambda": lambda_stacks, "minipix": minipix_stacks}[detector][0].shape[1:]
        if tuple(actual) != expected_shape:
            raise ConfigError(
                f"{detector} 的 XRS 图像尺寸 {actual} 与弹性阶段的 {expected_shape} 不一致，"
                "ROI 无法复用；请确认用的是同一套探测器设置"
            )

    per_scan_raw, per_scan_masked = [], []
    for index in range(len(kept_2D)):
        raw_parts, masked_parts = [], []
        for detector, stack_source in (("lambda", lambda_stacks), ("minipix", minipix_stacks)):
            selector = detectors == detector
            raw, masked = _roi_spectra(
                stack_source[index], boxes[selector], masks_all[selector]
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
    """非交互模式下的 I0 检查图（剔除的点画成灰虚线）。"""
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
            label=f"scan {index}" + ("  [剔除]" if is_out else ""),
        )
    axis.set_yscale("log")
    axis.set_xlabel(f"{cfg.get('xrs.energy_pv')} (keV)")
    axis.set_ylabel(cfg.get("xrs.i0_pv"))
    axis.set_title(f"XRS I0（剔除 {sorted(dropped) or '无'}）")
    axis.legend(fontsize=8, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
    figure.tight_layout()
    _save_figure(figure, cfg.state_dir / "figures" / "xrs_i0.png", log)
    _close(figure)


def _draw_xrs_qc(cfg, names, detectors, raw_list, masked_list, energies, i0_list, keep, log):
    """XRS 阶段 QC：各扫描的 ROI 谱 + I0 概览。"""
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
# q 阶段
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
                    "模组角度 q.module_angles_deg（度）",
                    _DEFAULT_MODULE_ANGLES,
                    float,
                    f"顺序固定为 {list(_MODULE_ORDER)}",
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
    else:  # as_before —— 复刻 notebook：只有 bad fit 才用拟合中心
        elastic_energy = np.full(len(names), _NOMINAL_ELASTIC_ENERGY, dtype=float)
        elastic_energy[bad_fit] = centers_kev[bad_fit]

    xrs = _load_stage_arrays(cfg.state_dir, "xrs")
    scan_count = int(xrs["n_scans"][0])
    # Ef 用全部扫描的能量轴（notebook 传的就是最后一条 energy2）
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
    log(f"[q] 完成 {len(names)} 个 ROI 的 Q 计算（Ei 来源：{source}）")
    return {
        "roi_names": names,
        "elastic_energy": elastic_energy,
        "q_ave": q_ave,
        "dq_ave": dq_ave,
        "q_range": q_range,
        "dq_range": dq_range,
    }


# ==========================================================================
# sum 阶段
# ==========================================================================


def _build_roi_table(cfg, elastic, xrs, q) -> pd.DataFrame:
    """汇总成最终 ``_rois.txt`` 的那张表。"""
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
    """把 ``crystal`` 列取成 numpy 的 ``<U`` 字符串数组。

    pandas 3 的字符串列可能是 Arrow 后端，``.astype(str).to_numpy()`` 会给出
    ``dtype=object`` 的数组——存进 npz 之后 ``allow_pickle=False`` 就读不回来了。
    """
    return table["crystal"].to_numpy(dtype=str)


def _inclusion(cfg, table: pd.DataFrame, modules) -> np.ndarray:
    """哪些 ROI 参与叠加：模组、q 范围、手动剔除、bad fit 四个条件。"""
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
    """所有扫描 × 所有 ROI 平移后能量范围的交集，按 step 对齐。"""
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
            f"各 ROI 平移后的能量范围没有交集（{lower:.3f} ~ {upper:.3f} eV）；"
            "请检查弹性峰拟合结果或 Estep"
        )
    count = int((upper - lower) / step)
    return lower, upper, np.linspace(lower, upper, count + 1)


def _interpolate_all(cfg, table, xrs, grid, include) -> dict:
    """把所有 (扫描, ROI) 谱内插到公共能量轴，并按 mode 叠加。"""
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
    # 没填的先用默认值兜底，交互窗口里再让用户改
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
        log(f"[WARN] sum.modules 里的 {dropped} 在当前 ROI 集合中不存在，已忽略")
    if not [item for item in requested if item in available_modules]:
        log(f"[WARN] sum.modules 全部不可用，改用 {available_modules[:1]}")

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
                "modules", "参与叠加的模组", tuple(available_modules),
                value=initial["modules"], multi=True,
            ),
            SliderSpec("q_low", "q 下限 (1/Å)", 0.0, 12.0, initial["q_low"], 0.05),
            SliderSpec("q_high", "q 上限 (1/Å)", 0.0, 12.0, initial["q_high"], 0.05),
            SliderSpec(
                "energy_step_ev", "能量内插步长 (eV)", 0.05, 2.0,
                initial["energy_step_ev"], 0.05,
            ),
        ]
        values = ui.adjust(
            tune_figure, controls, draw, title="sum：选模组、调 q 范围与内插步长"
        )
        state = prepare(values)
        _close(tune_figure)
    else:
        state = prepare(initial)

    if state["include"].sum() == 0:
        raise ConfigError(
            "没有任何 ROI 满足叠加条件（模组 ∩ q 范围 ∩ 非 bad fit ∩ 未手动剔除）；"
            f"当前 modules={state['modules']}, q_range={cfg.get('sum.q_range')}"
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
        f"[sum] {state['n_scans']} 个扫描 × {int(state['include'].sum())} 个 ROI 叠加完成，"
        f"能量 {state['grid'][0]:.2f} ~ {state['grid'][-1]:.2f} eV"
    )
    return payload


def _draw_sum(axes, state, table) -> None:
    """只清空调用方给的这两根轴——调参窗口的控件轴和它共用一个 figure。"""
    for axis in np.asarray(axes).ravel():
        axis.clear()
    grid = state["grid"]
    axes[0].plot(grid, state["combined"], lw=0.8)
    axes[0].set_title(
        f"叠加结果：modules={state['modules']}，"
        f"{int(state['include'].sum())} 个 ROI × {state['n_scans']} 个扫描"
    )
    axes[0].set_xlabel("Energy Transfer (eV)")
    axes[0].set_ylabel("Intensity")
    for module, curve in state["per_module"].items():
        axes[1].plot(grid, curve, lw=0.8, label=str(module))
    axes[1].set_title("分模组")
    axes[1].set_xlabel("Energy Transfer (eV)")
    axes[1].legend(fontsize=8)
    # 版面由调用方在 subplots(gridspec_kw=...) 时定好；这里既不 clear figure
    # 也不调 tight_layout——figure 上可能还挂着调参控件轴。


def _save_per_crystal_figure(cfg, state, table, log) -> None:
    """逐晶体画一张大图存盘（不弹窗，ROI 多的时候只画入选的）。"""
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
        figure.suptitle(f"只画了前 {limit} / {chosen.size} 个入选晶体", fontsize=14)
    figure.tight_layout()
    _save_figure(figure, cfg.state_dir / "figures" / "sum_per_crystal.png", log)
    _close(figure)


# ==========================================================================
# save 阶段
# ==========================================================================


def run_save(cfg: Config, ui: UI, log, force: bool = False, overwrite: bool = False) -> dict:
    if is_blank(cfg.get("output.filename")):
        cfg.set(
            "output.filename",
            ui.ask("输出目录名 output.filename", None, help_text="例：NiO_20260518"),
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
            f"输出目录已存在且非空：{out_dir}\n"
            "换个 output.filename，或加 --overwrite 覆盖。"
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
            raise ConfigError(f"文件已存在：{path}（加 --force 覆盖）")
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
        raise ConfigError(f"文件已存在：{info}（加 --force 覆盖）")
    info.write_text(_info_text(cfg, table, total), encoding="utf-8")
    written.append(info)
    log(f"[save] {info}")

    return {
        "output_dir": np.asarray([str(out_dir)]),
        "files": np.asarray([str(path) for path in written], dtype=str),
        "mode": np.asarray([mode]),
    }


def _info_text(cfg: Config, table: pd.DataFrame, total: dict) -> str:
    """处理过程记录。原 notebook 这里写的是从未定义过的 ``scanIDs2``。"""
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
# 调度
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
    """跑一个阶段，必要时先补参数、弹调参窗口，再落盘中间结果。

    ``force`` 只管"忽略参数指纹、强制重算"；能否覆盖已存在的输出文件由
    ``overwrite`` 单独控制——这两件事混在一起会让人想重算时被迫同意覆盖。
    """
    if stage not in STAGE_ORDER:
        raise ConfigError(f"未知阶段：{stage!r}；可用：{list(STAGE_ORDER)}")

    configure_matplotlib(int(cfg.get("ui.dpi", 150) or 150))
    store = StageStore(cfg.state_dir)

    if not force and store.is_current(stage, cfg.fingerprint(stage)):
        log(f"[skip] {stage} 已是最新（参数未变），产物在 {store.npz_path(stage)}")
        return store.load_arrays(stage)

    if force:
        store.reset_from(stage)

    log(f"===== 阶段 {stage}：{_stage_title(stage)} =====")
    log(f"[指纹] {cfg.fingerprint(stage)}")

    try:
        payload = _RUNNERS[stage](cfg, ui, log, force=force, overwrite=overwrite)
    except (ValueError, KeyError) as exc:
        # 把这些"输入不合法"的异常统一成 ConfigError，CLI 才能给出干净的报错
        raise ConfigError(f"{stage} 阶段失败：{exc}") from exc

    store.save_arrays(stage, payload)
    # 阶段自己可能刚补全/回写了参数（energy_pv、sum.modules、output.filename …），
    # 必须先把它们落盘再记指纹。否则下次运行 Config.load 读到的还是旧值，
    # 指纹对不上，这个阶段就会被无谓地重算一遍。
    if cfg.save():
        log(f"[config] 参数已回写 {cfg.path}")
    store.record(stage, cfg.fingerprint(stage))
    log(f"[done] {stage} → {store.npz_path(stage)}")
    return payload


def _stage_title(stage: str) -> str:
    from xrs_config import STAGE_TITLE

    return STAGE_TITLE.get(stage, stage)


# ==========================================================================
# 中心点重建（pick-roi / propose-roi 用）
# ==========================================================================


def load_elastic_images(cfg: Config, log=print) -> dict:
    """把弹性扫描的帧栈叠成每个探测器一张图，供选点/提案用。

    只依赖 ``data.root`` 和 ``elastic.scan_ids``——几何校准不该要求
    ROI 文件已经存在。
    """
    if is_blank(cfg.get("data.root")):
        raise ConfigError("data.root 为空，无法读取图像")
    scan_ids = [int(item) for item in (cfg.get("elastic.scan_ids") or [])]
    if not scan_ids:
        raise ConfigError("elastic.scan_ids 为空，无法读取图像")

    divide = bool(cfg.get("elastic.divide_by_i0"))
    i0_pv = cfg.get("elastic.i0_pv")
    wanted = [_DATASETS[name] for name in _DETECTORS]
    if divide and not is_blank(i0_pv):
        wanted.append(i0_pv)
    log(f"[roi] 读取扫描 {scan_ids[0]} 用于几何校准 …")
    (_, _, _, _, det1D_data_list, det2D_data_list) = xrsp.readH5(
        scan_ids[:1], cfg.raw_dir, useROI=False, detectors=wanted, log=log
    )
    if not det2D_data_list:
        raise ConfigError(f"扫描 {scan_ids[0]} 没读到任何 2D 探测器数据")
    lambda_stacks, minipix_stacks = _normalised_stacks(
        det2D_data_list, det1D_data_list, i0_pv, divide, log
    )
    return {
        "lambda": lambda_stacks[0].sum(axis=0),
        "minipix": minipix_stacks[0].sum(axis=0),
    }


def save_centers_overlay(image, centers, detector, path: Path, log) -> None:
    """把中心点画在图像上存成 PNG——无头运行时这是唯一的核对依据。"""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 11))
    axis.imshow(np.log1p(np.clip(np.asarray(image, dtype=float), 0, None)), cmap="gray")
    for label, (x, y) in centers.items():
        axis.plot(x, y, "o", color="#ff3030", ms=4)
        axis.text(x + 5, y - 5, str(label), color="#ff3030", fontsize=7)
    axis.set_title(f"{_LABELS[detector]}: {len(centers)} 个中心点")
    _save_figure(figure, path, log)
    _close(figure)


def centers_file_for(cfg: Config, detector: str) -> Path:
    return cfg.resolve(cfg.get(f"elastic.auto_centers.{detector}"))


def _load_centers_if_present(path: Path):
    if path and Path(path).is_file():
        return xrsp.load_auto_roi_centers(path)
    return None


def run_pick_roi(cfg: Config, detector: str, ui: UI, write: bool, log=print) -> dict:
    """交互式点选中心点，替代原来 ipycanvas 那个 Jupyter 部件。"""
    import auto_roi

    images = load_elastic_images(cfg, log)
    if detector not in images:
        raise ConfigError(f"未知探测器 {detector!r}；可用：{list(images)}")
    labels = auto_roi.expected_labels(detector)
    target = centers_file_for(cfg, detector)
    previous = _load_centers_if_present(target)
    if previous:
        log(f"[roi] 已有 {len(previous)} 个中心点，将按标签顺序重新点选以覆盖")

    figure_hint = None
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(11, 11))
    try:
        figure_hint = ui.pick_points(
            images[detector], labels, f"{_LABELS[detector]} ROI 中心点", figure=figure
        )
    finally:
        _close(figure)
    if not figure_hint:
        raise ConfigError("没有点到任何中心点")
    if len(figure_hint) != len(labels):
        log(f"[WARN] 只点了 {len(figure_hint)}/{len(labels)} 个，缺失的标签不会写入文件")

    written = _commit_centers(target, figure_hint, labels, write, log)
    save_centers_overlay(
        images[detector], figure_hint, detector,
        cfg.state_dir / "figures" / f"centers_{detector}.png", log,
    )
    return {"path": str(written), "count": len(figure_hint)}


def run_propose_roi(cfg: Config, detector: str, write: bool, log=print) -> dict:
    """用峰值检测给中心点文件生成候选值。默认只写 ``*.proposed.txt``。"""
    import auto_roi

    images = load_elastic_images(cfg, log)
    if detector not in images:
        raise ConfigError(f"未知探测器 {detector!r}；可用：{list(images)}")
    target = centers_file_for(cfg, detector)
    template = _load_centers_if_present(target)

    centers, report = auto_roi.propose_centers(images[detector], detector, template=template)
    log(f"[roi] 提案方式：{report['mode']}，找到 {report['num_peaks']} 个峰，"
        f"解析出 {len(report['resolved'])}/{report['num_expected']} 个标签")
    if report["missing"]:
        log(f"[WARN] 未能解析的标签（{len(report['missing'])} 个）：{report['missing']}")
    if not centers:
        raise ConfigError("没能生成任何候选中心点，请改用 pick-roi 手工点选")

    labels = auto_roi.expected_labels(detector)
    if write:
        written = _commit_centers(target, centers, labels, True, log)
    else:
        written = target.with_suffix(target.suffix + ".proposed.txt")
        n = xrsp.write_auto_roi_centers(written, centers, labels=labels)
        log(f"[roi] 候选中心点写到 {written}（{n} 个）。核对无误后再加 --write 覆盖正式文件。")

    save_centers_overlay(
        images[detector], centers, detector,
        cfg.state_dir / "figures" / f"centers_{detector}_proposed.png", log,
    )
    return {"path": str(written), "count": len(centers), "mode": report["mode"]}


def _commit_centers(target: Path, centers, labels, write: bool, log) -> Path:
    """写中心点文件。覆盖前先留一份 ``.bak``。"""
    target = Path(target)
    if target.is_file() and write:
        backup = target.with_suffix(target.suffix + ".bak")
        backup.write_bytes(target.read_bytes())
        log(f"[roi] 原文件已备份到 {backup}")
    count = xrsp.write_auto_roi_centers(target, centers, labels=labels)
    log(f"[roi] 写入 {target}（{count} 个中心点）")
    return target
