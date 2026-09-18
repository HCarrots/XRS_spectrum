"""test_pipeline implementation."""

from __future__ import annotations

import re
import shutil

import numpy as np
import pandas as pd
import pytest

import auto_roi
import xrs_processing as xrsp
from conftest import (
    BLOBS,
    PROJECT_ROOT,
    REGULAR_BOXES,
    TEMPLATE,
    ELASTIC_SCAN_ID,
    XRS_SCAN_ID,
    build_case,
)
from run_xrs import main
from xrs_config import STAGE_ORDER, Config, ConfigError, environment_parity
from xrs_pipeline import _prepare_roi_inputs, _remove_i0_glitches, run_stage
from xrs_ui import UI


# ==========================================================================
# English note.
# ==========================================================================


def test_config_roundtrip_preserves_comments(work_dir):
    """Implementation notes for test_config_roundtrip_preserves_comments."""
    path = work_dir / "config.yaml"
    shutil.copy(TEMPLATE, path)
    before = path.read_text(encoding="utf-8")

    cfg = Config.load(path)
    assert cfg.save() is False, ""

    cfg.set("elastic.scan_ids", [14522])
    cfg.set("elastic.roi_mode", "regular")
    cfg.set("sum.q_range", [0.5, 9.5])
    cfg.set("q.module_angles_deg", [145.6, 79.03, 25, 118.8, 58.85, 67.862])
    assert cfg.save() is True
    assert cfg.save() is False, ""

    after = path.read_text(encoding="utf-8")
    for marker in (
        "# XRS spectrum processing configuration",
        "# Elastic scans used to establish ROI geometry",
        "# Empty values (null or []) are requested interactively",
    ):
        assert marker in after, f"Missing preserved comment: {marker}"
    assert "scan_ids: [14522]" in after
    assert "q_range: [0.5, 9.5]" in after
    assert after.count("elastic:") == before.count("elastic:")

    reloaded = Config.load(path)
    assert reloaded.get("elastic.roi_mode") == "regular"
    assert reloaded.get("sum.q_range") == [0.5, 9.5]
    assert not (work_dir / "config.yaml.tmp").exists()


def test_config_problems_point_at_yaml_paths():
    cfg = Config.load(TEMPLATE)
    problems = cfg.problems("elastic")
    joined = "\n".join(problems)
    assert "data.root" in joined
    assert "elastic.scan_ids" in joined
    assert "elastic.roi_mode" in joined
# English note.
    assert "regular_files" not in joined

    cfg.set("elastic.roi_mode", "regular")
    joined = "\n".join(cfg.problems("elastic"))
    assert "elastic.regular_files.lambda" in joined
    assert "elastic.regular_files.minipix" in joined
    assert "auto_centers" not in joined


def test_fingerprint_tracks_upstream_changes():
    cfg = Config.load(TEMPLATE)
    cfg.set("data.root", "/tmp/x")
    cfg.set("elastic.scan_ids", [1])
    cfg.set("elastic.roi_mode", "regular")
    cfg.set("elastic.regular_files.lambda", "roi/a.txt")
    cfg.set("elastic.regular_files.minipix", "roi/b.txt")

    before_sum = cfg.fingerprint("sum")
    cfg.set("sum.q_range", [1, 2])
    after_sum = cfg.fingerprint("sum")
    assert before_sum != after_sum
# English note.
    assert cfg.fingerprint("elastic") == cfg.fingerprint("elastic")

    before_elastic = cfg.fingerprint("elastic")
    cfg.set("elastic.filter_value", 0.31)
    assert cfg.fingerprint("elastic") != before_elastic
# English note.
    assert cfg.fingerprint("sum") != after_sum


def test_environment_manifests_agree():
    only_pixi, only_env = environment_parity(PROJECT_ROOT)
    assert not only_pixi, f"Only in pixi.toml: {only_pixi}"
    assert not only_env, f"Only in environment.yml: {only_env}"


# ==========================================================================
# English note.
# ==========================================================================

_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(ipycanvas|ipywidgets|IPython|ipykernel|jupyter\w*)",
    re.MULTILINE,
)


def test_shipped_code_has_no_jupyter_imports():
    offenders = []
    for path in sorted(PROJECT_ROOT.glob("*.py")):
        if _IMPORT_RE.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert not offenders, f"Jupyter imports remain: {offenders}"

    for name in ("pixi.toml", "environment.yml"):
        text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        assert "ipycanvas" not in text
        assert "ipywidgets" not in text
        assert "jupyter" not in text.lower()


def test_auto_roi_module_has_no_widget_api():
    """Implementation notes for test_auto_roi_module_has_no_widget_api."""
    assert not hasattr(auto_roi, "select_roi_centers")
    assert not hasattr(auto_roi, "_canvas_image")
    assert not hasattr(auto_roi, "asyncio")


def test_roi_workflow_is_synchronous():
    """Implementation notes for test_roi_workflow_is_synchronous."""
    import inspect

    assert not inspect.iscoroutinefunction(xrsp.build_roi_workflow)
    assert not inspect.iscoroutinefunction(auto_roi.build_roi_session)


# ==========================================================================
# English note.
# ==========================================================================


def test_center_file_roundtrip(work_dir):
    centers = {"HB-A1": (16, 14), "HB-A2": (41, 14)}
    path = work_dir / "centers.txt"
    count = xrsp.write_auto_roi_centers(
        path, centers, labels=auto_roi.expected_labels("minipix")
    )
    assert count == 2
    assert xrsp.load_auto_roi_centers(path) == centers
    assert path.read_text(encoding="utf-8").splitlines()[0] == "roi_label\tx\ty"


def test_auto_roi_hdf5_roundtrip_and_shape_validation(work_dir):
    label_image = np.zeros((12, 14), dtype=np.uint16)
    label_image[2:5, 3:7] = 1
    label_image[7:10, 8:12] = 2
    result = {
        "label_image": label_image,
        "roi_labels": np.asarray(["HB-A1", "HB-A2"]),
        "centers_xy": np.asarray([[4, 3], [9, 8]], dtype=np.int32),
        "failed_labels": np.asarray(["HB-A3"]),
        "failed_reasons": {"HB-A3": "area is below 20 pixels"},
        "overlap_pixels": 3,
    }
    path = work_dir / "auto.h5"
    xrsp.write_auto_roi_hdf5(
        path, result, detector="minipix", scan_ids=[1, 2],
        image_shape=label_image.shape,
        parameters={"smooth_sigma": 2.0, "threshold_tightness": 1.0, "min_area": 20},
    )
    loaded = xrsp.load_auto_roi_hdf5(
        path, detector="minipix", image_shape=label_image.shape
    )
    assert np.array_equal(loaded["label_image"], label_image)
    assert list(loaded["roi_labels"]) == ["HB-A1", "HB-A2"]
    assert loaded["failed_reasons"]["HB-A3"] == "area is below 20 pixels"
    assert loaded["parameters"]["min_area"] == 20
    with pytest.raises(ValueError, match="image shape"):
        xrsp.load_auto_roi_hdf5(path, detector="minipix", image_shape=(1, 1))


def test_regular_roi_text_roundtrip(work_dir):
    path = work_dir / "regular.txt"
    rectangles = [("HB-A1", 2, 8, 3, 9), ("HB-A2", 10, 15, 4, 11)]
    assert xrsp.write_regular_rois(path, rectangles) == 2
    rois, adjustments = xrsp._load_regular_rois(path, "minipix")
    assert [roi.name for roi in rois] == ["HB-A1", "HB-A2"]
    assert [(roi.x1, roi.x2, roi.y1, roi.y2) for roi in rois] == [
        (2, 8, 3, 9), (10, 15, 4, 11)
    ]
    assert all(value == [0, 0] for value in adjustments.values())


def test_python_and_config_text_is_english_only():
    han = re.compile(
        r"[\u3400-\u9fff\u3002\uff0c\uff1a\uff1b\uff08\uff09"
        r"\u3010\u3011\u300a\u300b\u3001]|\u2014\u2014"
    )
    paths = list(PROJECT_ROOT.glob("*.py")) + list((PROJECT_ROOT / "tests").glob("*.py"))
    paths.extend(PROJECT_ROOT.glob("*.yaml"))
    paths.extend(PROJECT_ROOT.glob("*.yml"))
    offenders = [path.name for path in paths if han.search(path.read_text(encoding="utf-8"))]
    assert not offenders, f"CJK characters remain in source/config files: {offenders}"


def test_expected_labels_are_canonical():
    assert len(auto_roi.expected_labels("lambda")) == 60
    assert len(auto_roi.expected_labels("minipix")) == 15
    assert auto_roi.expected_labels("minipix")[:3] == ("HB-A1", "HB-A2", "HB-A3")
    with pytest.raises(ValueError):
        auto_roi.expected_labels("nope")


def test_glitch_removal_fixes_i0_and_frames():
    i0 = [np.full(12, 100.0)]
    i0[0][5] = 10.0
    frames = [
        {
            "D_LAMBDA": np.ones((12, 4, 4)),
            "D_MINIPIX": np.ones((12, 4, 4)),
        }
    ]
    removed = _remove_i0_glitches(i0, frames, log=lambda _message: None)
    assert removed == 1
    assert i0[0][5] == pytest.approx(100.0)
    assert frames[0]["D_LAMBDA"][5].mean() == pytest.approx(1.0)


def test_q_calc_is_in_a_sane_range():
    angles = [np.deg2rad(value) for value in (145.6, 79.03, 25, 118.8, 58.85, 67.862)]
    ef = np.linspace(9.9, 10.1, 50)
    _, _, q_ave, dq_ave, q_range, _ = xrsp.qCalc(angles, "HB-A1", 9.68, ef)
    assert 5.0 < q_ave < 11.0
    assert dq_ave > 0
    assert q_range > 0
    with pytest.raises(ValueError):
        xrsp.qCalc(angles, "NOT-A-ROI", 9.68, ef)


def test_propose_centers_prefers_template():
    """Implementation notes for test_propose_centers_prefers_template."""
    image = np.zeros((48, 60))
    yy, xx = np.mgrid[0:48, 0:60]
    for cx, cy in BLOBS:
        image += 100.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0**2))
    template = {"HB-A1": (14, 12), "HB-A2": (39, 12)}  #  2 px
    centers, report = auto_roi.propose_centers(image, "minipix", template=template)
    assert report["mode"] == "template-match"
    for label, (x, y) in centers.items():
        assert abs(x - template[label][0]) <= 4, (label, x, y)
        assert abs(y - template[label][1]) <= 4, (label, x, y)


# ==========================================================================
# English note.
# ==========================================================================


def _run(config_path, *extra):
    return main(["run", str(config_path), "--no-ui", *extra])


@pytest.mark.parametrize("mode", ["regular", "auto"])
def test_end_to_end(work_dir, mode, capsys):
    config_path = build_case(work_dir / "case", mode=mode)
    assert _run(config_path) == 0, capsys.readouterr().out

    root = config_path.parent
    out = root / "processed" / "synthetic_run"

    data = pd.read_csv(out / "synthetic_run_data.txt", sep="\t")
    assert list(data.columns) == ["Energy Transfer (eV)", "Intensity"]
    assert len(data) > 100
    assert np.allclose(np.diff(data["Energy Transfer (eV)"]), 0.2, atol=1e-9)
    assert np.isfinite(data["Intensity"]).all()
    assert (data["Intensity"] > 0).all()

# English note.
    rois = pd.read_csv(out / "synthetic_run_rois.txt", sep="\t")
    assert len(rois) == 2 * len(REGULAR_BOXES)
    assert {"crystal", "x1", "x2", "y1", "y2", "q_ave", "center", "width"} <= set(rois.columns)
    assert not rois["bad_fit"].any(), rois[["crystal", "center", "width", "r-square"]]

# English note.
    assert np.allclose(rois["center"], 9680.0, atol=5.0)
# English note.
    assert rois["q_ave"].between(0, 12).all()

    info = (out / "synthetic_run_info.txt").read_text(encoding="utf-8")
    assert f"scan_ids (xrs) = [{XRS_SCAN_ID}]" in info
    assert "Number of ROIs included = " in info

    figures = root / "processed" / ".xrs_state" / "figures"
    for name in ("elastic_qc.png", "xrs_i0.png", "xrs_roi_sums.png",
                 "sum_qc.png", "sum_per_crystal.png"):
        assert (figures / name).is_file(), f" QC  {name}"


@pytest.mark.parametrize("empty_detector", ["lambda", "minipix"])
def test_regular_end_to_end_allows_one_empty_detector(
    work_dir, capsys, empty_detector
):
    config_path = build_case(work_dir / "case", mode="regular")
    cfg = Config.load(config_path)
    empty_path = cfg.resolve(cfg.get(f"elastic.regular_files.{empty_detector}"))
    xrsp.write_regular_rois(empty_path, [])

    assert _run(config_path) == 0, capsys.readouterr().out
    elastic = np.load(cfg.state_dir / "elastic.npz", allow_pickle=False)
    expected_detector = "minipix" if empty_detector == "lambda" else "lambda"
    assert set(elastic["roi_detectors"].tolist()) == {expected_detector}
    assert int(elastic["mask_count"][0]) == len(REGULAR_BOXES)


def test_elastic_uses_all_configured_scans(work_dir):
    config_path = build_case(work_dir / "case", mode="regular")
    raw = config_path.parent / "raw"
    second_id = ELASTIC_SCAN_ID + 1
    shutil.copy2(
        raw / f"{ELASTIC_SCAN_ID}_elastic.nxs",
        raw / f"{second_id}_elastic.nxs",
    )
    cfg = Config.load(config_path)
    cfg.set("elastic.scan_ids", [ELASTIC_SCAN_ID, second_id])
    cfg.save()
    payload = run_stage(cfg, "elastic", UI(interactive=False), log=lambda _message: None)
    assert payload["elastic_energy"].size == 402
    assert np.all(np.diff(payload["elastic_energy"]) >= 0)
    assert payload["roi_names"].size == 2 * len(REGULAR_BOXES)


def test_resume_skips_unchanged_stages(work_dir, capsys):
    config_path = build_case(work_dir / "case", mode="regular")
    assert _run(config_path) == 0
    capsys.readouterr()

    assert _run(config_path) == 0
    output = capsys.readouterr().out
    assert output.count("[skip]") == len(STAGE_ORDER), output

# English note.
    cfg = Config.load(config_path)
    cfg.set("sum.energy_step_ev", 0.5)
    cfg.save()
    capsys.readouterr()
    assert _run(config_path, "--overwrite") == 0
    output = capsys.readouterr().out
    assert output.count("[skip]") == STAGE_ORDER.index("sum")

    data = pd.read_csv(
        config_path.parent / "processed" / "synthetic_run" / "synthetic_run_data.txt",
        sep="\t",
    )
    assert np.allclose(np.diff(data["Energy Transfer (eV)"]), 0.5, atol=1e-9)


def test_check_reports_ready_case(work_dir, capsys):
    config_path = build_case(work_dir / "case", mode="regular")
    assert main(["check", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "not yet run" in output
    assert "Environment manifests match" in output


def test_check_flags_missing_parameters(capsys):
    assert main(["check", str(TEMPLATE)]) == 1
    output = capsys.readouterr().out
    assert "data.root" in output
    assert "elastic.roi_mode" in output


def test_unknown_roi_label_is_rejected(work_dir, capsys):
    config_path = build_case(work_dir / "case", mode="auto")
    (config_path.parent / "roi" / "auto_lambda.txt").write_text(
        "roi_label\tx\ty\nZZ-A1\t16\t14\n", encoding="utf-8"
    )
    assert _run(config_path) == 1
    output = capsys.readouterr().out
    assert "unknown labels" in output


def test_out_of_range_center_is_rejected(work_dir, capsys):
    config_path = build_case(work_dir / "case", mode="auto")
    (config_path.parent / "roi" / "auto_minipix.txt").write_text(
        "roi_label\tx\ty\nHB-A1\t9999\t14\n", encoding="utf-8"
    )
    assert _run(config_path) == 1
    output = capsys.readouterr().out
    assert "outside image" in output


def test_save_refuses_to_overwrite_without_force(work_dir, capsys):
    config_path = build_case(work_dir / "case", mode="regular")
    assert _run(config_path) == 0
    capsys.readouterr()

    out = config_path.parent / "processed" / "synthetic_run"
    assert (out / "synthetic_run_data.txt").is_file()

# English note.
    (config_path.parent / "processed" / ".xrs_state" / "save.npz").unlink()

    assert _run(config_path, "--stage", "save") == 1
    assert "already exists" in capsys.readouterr().out

# English note.
    assert _run(config_path, "--stage", "save", "--force") == 1
    assert "already exists" in capsys.readouterr().out

    assert _run(config_path, "--stage", "save", "--overwrite") == 0
    assert "[save]" in capsys.readouterr().out


def test_no_ui_reports_all_missing_parameters(capsys):
    assert main(["run", str(TEMPLATE), "--no-ui"]) == 1
    output = capsys.readouterr().out
    assert "required for --no-ui" in output
    for stage in STAGE_ORDER:
        assert f"[{stage}]" in output


def test_single_stage_requires_upstream_state(work_dir, capsys):
    config_path = build_case(work_dir / "case", mode="regular")
    assert _run(config_path, "--stage", "q") == 1
    output = capsys.readouterr().out
    assert "Missing intermediate result" in output


def test_glitch_removal_end_to_end(work_dir, capsys):
    """Implementation notes for test_glitch_removal_end_to_end."""
    from conftest import build_case as _build

    config_path = _build(work_dir / "case", mode="regular", glitch_index=50)
    assert _run(config_path) == 0
    output = capsys.readouterr().out
    assert "glitch" in output


def test_ui_required_without_interaction():
    ui = UI(interactive=False)
    from xrs_ui import UiRequired

    with pytest.raises(UiRequired):
        ui.require("window")
    assert ui.confirm("continue?", default=True) is True


def test_run_stage_returns_arrays(work_dir):
    config_path = build_case(work_dir / "case", mode="regular")
    cfg = Config.load(config_path)
    ui = UI(interactive=False)
    payload = run_stage(cfg, "elastic", ui, log=lambda _m: None)
    assert payload["roi_names"].shape == (2 * len(REGULAR_BOXES),)
    assert payload["bad_fit"].shape == (2 * len(REGULAR_BOXES),)
    assert int(payload["mask_count"][0]) == 2 * len(REGULAR_BOXES)
    assert payload["image_shape"].tolist() == [48, 60]

    with pytest.raises(ConfigError):
        run_stage(cfg, "not-a-stage", ui, log=lambda _m: None)


# ==========================================================================
# English note.
# ==========================================================================


def _mouse(figure, axis, x, y, button=1):
    from matplotlib.backend_bases import MouseEvent

    event = MouseEvent("button_press_event", figure.canvas, x, y, button=button)
    event.inaxes = axis
    event.xdata, event.ydata = float(x), float(y)
    figure.canvas.callbacks.process("button_press_event", event)


def _key(figure, key):
    from matplotlib.backend_bases import KeyEvent

    figure.canvas.callbacks.process(
        "key_press_event", KeyEvent("key_press_event", figure.canvas, key)
    )


def _interactive_ui(monkeypatch, action):
    """Implementation notes for _interactive_ui."""
    import matplotlib.pyplot as plt

    import xrs_ui

    monkeypatch.setattr(xrs_ui, "_run_event_loop", lambda figure, state: action(figure, state))
    ui = xrs_ui.UI(interactive=True)
    monkeypatch.setattr(ui, "require", lambda _what: None)  # Skip backend checks.
    return ui, plt


def test_preview_continue_button_remains_clickable(monkeypatch):
    import gc

    import numpy as np

    def action(figure, state):
        gc.collect()
        assert [widget.label.get_text() for widget in figure._xrs_widgets] == ["Continue"]
        figure._xrs_widgets[0]._observers.process("clicked", None)
        assert state == {"done": True, "accepted": True}

    ui, plt = _interactive_ui(monkeypatch, action)
    ui.preview_images({"lambda": np.zeros((8, 8)), "minipix": np.zeros((8, 8))})
    plt.close("all")


def test_review_redraw_button_remains_clickable(monkeypatch):
    import gc

    import numpy as np

    def action(figure, state):
        gc.collect()
        assert [widget.label.get_text() for widget in figure._xrs_widgets] == [
            "Use existing",
            "Redraw",
        ]
        figure._xrs_widgets[1]._observers.process("clicked", None)

    ui, plt = _interactive_ui(monkeypatch, action)
    reuse = ui.review_geometry(
        np.zeros((8, 8)), ["A1"], [(1, 4, 2, 6)], "Review"
    )
    assert reuse is False
    plt.close("all")


def test_rectangle_editor_redraws_last_roi_and_preserves_selector(monkeypatch):
    from types import SimpleNamespace

    def selection(x1, y1, x2, y2):
        return (
            SimpleNamespace(xdata=x1, ydata=y1),
            SimpleNamespace(xdata=x2, ydata=y2),
        )

    def action(figure, state):
        selector = figure._xrs_widgets[0]
        selector.onselect(*selection(2, 3, 8, 9))
        assert selector._selection_artist.axes is figure.axes[0]
        _key(figure, "r")
        selector.onselect(*selection(4, 5, 10, 12))
        selector.onselect(*selection(14, 6, 20, 13))
        _key(figure, "enter")

    ui, plt = _interactive_ui(monkeypatch, action)
    rectangles = ui.pick_rectangles(
        np.zeros((30, 40)), ("A1", "A2"), "Draw Lambda rectangular ROIs"
    )
    assert rectangles == [
        ("A1", 4, 10, 5, 12),
        ("A2", 14, 20, 6, 13),
    ]
    plt.close("all")


@pytest.mark.parametrize("detector", ["Lambda", "Minipix"])
def test_rectangle_editor_enter_accepts_empty_detector(monkeypatch, detector):
    def action(figure, state):
        assert getattr(figure.canvas.manager, "key_press_handler_id", None) is None
        _key(figure, "enter")
        assert state == {"done": True, "accepted": True}

    ui, plt = _interactive_ui(monkeypatch, action)
    rectangles = ui.pick_rectangles(
        np.zeros((30, 40)), ("A1", "A2"), f"Draw {detector} rectangular ROIs"
    )
    assert rectangles == []
    plt.close("all")


def test_regular_roi_saves_lambda_before_opening_minipix(work_dir):
    config_path = build_case(work_dir / "case", mode="regular")
    cfg = Config.load(config_path)
    lambda_path = cfg.resolve(cfg.get("elastic.regular_files.lambda"))
    minipix_path = cfg.resolve(cfg.get("elastic.regular_files.minipix"))
    lambda_path.unlink()
    minipix_path.unlink()
    opened = []

    class SequentialUI:
        interactive = True

        @staticmethod
        def preview_images(_images, _title):
            return None

        @staticmethod
        def pick_rectangles(_image, labels, title):
            opened.append(title)
            if "Lambda" in title:
                return []
            if "Minipix" in title:
                assert lambda_path.is_file()
                assert pd.read_csv(lambda_path, sep="\t").empty
            return [(str(labels[0]), 1, 4, 2, 6)]

    mode = _prepare_roi_inputs(
        cfg,
        SequentialUI(),
        {"lambda": np.zeros((10, 12)), "minipix": np.zeros((10, 12))},
        [ELASTIC_SCAN_ID],
        log=lambda _message: None,
    )
    assert mode == "regular"
    assert lambda_path.is_file() and minipix_path.is_file()
    assert opened == [
        "Draw Lambda rectangular ROIs",
        "Draw Minipix rectangular ROIs",
    ]


def test_pick_points_collects_clicks_in_label_order(monkeypatch):
    import numpy as np

    labels = ("HB-A1", "HB-A2")
    wanted = {"HB-A1": (16, 14), "HB-A2": (41, 14)}

    def action(figure, state):
        axis = figure.axes[0]
        for x, y in wanted.values():
            _mouse(figure, axis, x, y)
# English note.
        _mouse(figure, axis, 0, 0, button=3)
        _mouse(figure, axis, 41, 14)
        _key(figure, "enter")

    ui, plt = _interactive_ui(monkeypatch, action)
    image = np.zeros((48, 60))
    image[14, 16] = 100
    picked = ui.pick_points(image, labels, "test", figure=plt.figure())
    assert picked == wanted
    plt.close("all")


def test_rectangle_bounds_are_half_open_clamped_and_non_empty():
    from xrs_ui import _clamp_rectangle

    assert _clamp_rectangle(-3, -2, 8.2, 9.7, (20, 30)) == (0, 9, 0, 10)
    assert _clamp_rectangle(8.2, 9.7, 1.2, 2.2, (20, 30)) == (1, 9, 2, 10)
    with pytest.raises(ValueError, match="non-zero"):
        _clamp_rectangle(-3, -2, -1, -1, (20, 30))


def test_pick_points_allows_partial_selection(monkeypatch):
    import numpy as np

    calls = {"n": 0}

    def action(figure, state):
        axis = figure.axes[0]
        _mouse(figure, axis, 10, 10)
        _key(figure, "enter")
        assert state["done"] is True
        assert state["accepted"] is True
        calls["n"] += 1

    ui, plt = _interactive_ui(monkeypatch, action)
    result = ui.pick_points(
        np.zeros((48, 60)), ("HB-A1", "HB-A2"), "test", figure=plt.figure()
    )
    assert result == {"HB-A1": (10, 10)}
    assert calls["n"] == 1
    plt.close("all")


def test_toggle_scans_picks_nearest_curve(monkeypatch):
    import numpy as np

    energy = np.linspace(0, 10, 11)
    series = [(energy, np.full(11, level)) for level in (100.0, 200.0, 300.0)]

    def action(figure, state):
        axis = figure.axes[0]
        _mouse(figure, axis, 5.0, 200.0)  # Select the nearest, second curve.
        _key(figure, "enter")

    ui, plt = _interactive_ui(monkeypatch, action)
    dropped = ui.toggle_scans(series, [], "test", figure=plt.figure())
    assert dropped == [1]
    plt.close("all")


def test_adjust_returns_widget_values(monkeypatch):
    import xrs_ui

    drawn = []

    def action(figure, state):
        state["done"] = True
        state["accepted"] = True

    ui, plt = _interactive_ui(monkeypatch, action)
    controls = [
        xrs_ui.SliderSpec("a", "A", 0, 10, 3, 1, integer=True),
        xrs_ui.SliderSpec("b", "B", 0.0, 1.0, 0.25, 0.05),
        xrs_ui.ChoiceSpec("c", "C", ("x", "y"), value="y"),
        xrs_ui.ChoiceSpec("d", "D", ("p", "q"), value=["p"], multi=True),
    ]
    values = ui.adjust(plt.figure(), controls, lambda v: drawn.append(dict(v)))
    assert values["a"] == 3 and isinstance(values["a"], int)
    assert values["b"] == pytest.approx(0.25)
    assert values["c"] == "y"
    assert values["d"] == ["p"]
    assert drawn, "draw "
    plt.close("all")


def test_adjust_abort_raises(monkeypatch):
    import xrs_ui
    from xrs_ui import UiCancelled

    def action(figure, state):
        state["done"] = True
        state["accepted"] = False

    ui, plt = _interactive_ui(monkeypatch, action)
    with pytest.raises(UiCancelled):
        ui.adjust(
            plt.figure(),
            [xrs_ui.SliderSpec("a", "A", 0, 1, 0.5)],
            lambda _v: None,
        )
    plt.close("all")


# ==========================================================================
# English note.
# ==========================================================================


def _accept_immediately(monkeypatch):
    return _interactive_ui(
        monkeypatch, lambda figure, state: state.update(done=True, accepted=True)
    )


def test_elastic_stage_interactive_tuning(work_dir, monkeypatch):
    """Implementation notes for test_elastic_stage_interactive_tuning."""
    config_path = build_case(work_dir / "case", mode="regular")
    cfg = Config.load(config_path)
    ui, plt = _accept_immediately(monkeypatch)

    payload = run_stage(cfg, "elastic", ui, log=lambda _m: None)
    assert payload["roi_names"].shape == (2 * len(REGULAR_BOXES),)
    assert int(payload["mask_count"][0]) == 2 * len(REGULAR_BOXES)
# English note.
    assert cfg.get("elastic.filter_value") == pytest.approx(0.15)
    assert cfg.get("elastic.fit.e_lowlim") == pytest.approx(9.67)
    plt.close("all")


def test_xrs_stage_interactive_scan_removal(work_dir, monkeypatch):
    """Implementation notes for test_xrs_stage_interactive_scan_removal."""
    config_path = build_case(work_dir / "case", mode="regular")
    cfg = Config.load(config_path)
    ui, plt = _accept_immediately(monkeypatch)
    run_stage(cfg, "elastic", ui, log=lambda _m: None)
    payload = run_stage(cfg, "xrs", ui, log=lambda _m: None)
    assert int(payload["n_scans"][0]) == 1
    assert list(payload["excluded_scans"]) == []
    assert (cfg.state_dir / "figures" / "xrs_i0.png").is_file()
    plt.close("all")


def test_sum_stage_interactive_tuning(work_dir, monkeypatch):
    config_path = build_case(work_dir / "case", mode="auto")
    cfg = Config.load(config_path)
    quiet = UI(interactive=False)
    for stage in ("elastic", "xrs", "q"):
        run_stage(cfg, stage, quiet, log=lambda _m: None)

    ui, plt = _accept_immediately(monkeypatch)
    payload = run_stage(cfg, "sum", ui, log=lambda _m: None)
    assert int(payload["n_included"][0]) > 0
    assert np.allclose(np.diff(payload["E_interp"]), 0.2, atol=1e-9)
    assert (cfg.state_dir / "figures" / "sum_qc.png").is_file()
    plt.close("all")


# ==========================================================================
# English note.
# ==========================================================================


def test_propose_roi_writes_candidate_not_target(work_dir, capsys):
    config_path = build_case(work_dir / "case", mode="auto")
    roi_dir = config_path.parent / "roi"
    original = xrsp.load_auto_roi_centers(roi_dir / "auto_minipix.txt")

    assert main(["propose-roi", str(config_path), "--detector", "minipix"]) == 0
    capsys.readouterr()

    candidate = roi_dir / "auto_ROI_minipix.proposed.h5"
    assert candidate.is_file(), "The default must write a proposed HDF5 file"
    proposed = xrsp.load_auto_roi_hdf5(candidate, detector="minipix")
    assert len(proposed["roi_labels"]) > 0

# English note.
    assert xrsp.load_auto_roi_centers(roi_dir / "auto_minipix.txt") == original

    figures = config_path.parent / "processed" / ".xrs_state" / "figures"
    assert (figures / "centers_minipix_proposed.png").is_file()


def test_propose_roi_with_write_backs_up(work_dir, capsys):
    config_path = build_case(work_dir / "case", mode="auto")
    roi_dir = config_path.parent / "roi"

    assert main(["propose-roi", str(config_path), "--detector", "minipix", "--write"]) == 0
    assert main(["propose-roi", str(config_path), "--detector", "minipix", "--write"]) == 0
    capsys.readouterr()

    assert (roi_dir / "auto_ROI_minipix.h5.bak").is_file()
    official = xrsp.load_auto_roi_hdf5(
        roi_dir / "auto_ROI_minipix.h5", detector="minipix"
    )
    assert len(official["roi_labels"]) > 0


def test_pick_roi_writes_clicked_centers(work_dir, capsys, monkeypatch):
    import auto_roi

    config_path = build_case(work_dir / "case", mode="auto")
    roi_dir = config_path.parent / "roi"
    labels = auto_roi.expected_labels("minipix")
# English note.
    clicks = [(16, 14), (41, 14)]
    calls = {"count": 0}

    def action(figure, state):
        if calls["count"] == 0:
            axis = figure.axes[0]
            for x, y in clicks:
                _mouse(figure, axis, x, y)
        _key(figure, "enter")
        calls["count"] += 1

    import xrs_ui

    monkeypatch.setattr(xrs_ui, "_run_event_loop", lambda figure, state: action(figure, state))
    assert main(["pick-roi", str(config_path), "--detector", "minipix"]) == 0
    capsys.readouterr()

    written = xrsp.load_auto_roi_hdf5(
        roi_dir / "auto_ROI_minipix.h5", detector="minipix"
    )
    assert list(written["roi_labels"]) == list(labels[:2])
    assert calls["count"] == 2
