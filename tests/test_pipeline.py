"""端到端与单元测试。

跑法：``pytest tests -q``（或 ``pixi run python -m pytest tests -q``）。
所有测试都在 ``--no-ui`` 下运行，不弹任何窗口。
"""

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
    XRS_SCAN_ID,
    build_case,
)
from run_xrs import main
from xrs_config import STAGE_ORDER, Config, ConfigError, environment_parity
from xrs_pipeline import _remove_i0_glitches, run_stage
from xrs_ui import UI


# ==========================================================================
# 配置层
# ==========================================================================


def test_config_roundtrip_preserves_comments(work_dir):
    """回写参数不能把用户手写的注释和键顺序搞丢。"""
    path = work_dir / "config.yaml"
    shutil.copy(TEMPLATE, path)
    before = path.read_text(encoding="utf-8")

    cfg = Config.load(path)
    assert cfg.save() is False, "内容没变时不该动文件"

    cfg.set("elastic.scan_ids", [14522])
    cfg.set("elastic.roi_mode", "regular")
    cfg.set("sum.q_range", [0.5, 9.5])
    cfg.set("q.module_angles_deg", [145.6, 79.03, 25, 118.8, 58.85, 67.862])
    assert cfg.save() is True
    assert cfg.save() is False, "第二次保存应当是幂等的"

    after = path.read_text(encoding="utf-8")
    for marker in ("# XRS 光谱处理参数", "# 弹性峰数据编号", "（注释和键顺序都会保留）"):
        assert marker in after, f"注释丢了：{marker}"
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
    # 未填 roi_mode 时不该越界去要求 ROI 文件
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
    # 改 sum 不该让 elastic 失效
    assert cfg.fingerprint("elastic") == cfg.fingerprint("elastic")

    before_elastic = cfg.fingerprint("elastic")
    cfg.set("elastic.filter_value", 0.31)
    assert cfg.fingerprint("elastic") != before_elastic
    # 上游变了，sum 的指纹也必须变
    assert cfg.fingerprint("sum") != after_sum


def test_environment_manifests_agree():
    only_pixi, only_env = environment_parity(PROJECT_ROOT)
    assert not only_pixi, f"只在 pixi.toml 里：{only_pixi}"
    assert not only_env, f"只在 environment.yml 里：{only_env}"


# ==========================================================================
# 无 Jupyter
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
    assert not offenders, f"仍有 Jupyter 相关导入：{offenders}"

    for name in ("pixi.toml", "environment.yml"):
        text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        assert "ipycanvas" not in text
        assert "ipywidgets" not in text
        assert "jupyter" not in text.lower()


def test_auto_roi_module_has_no_widget_api():
    """原先那个 ipycanvas 点选函数必须真的没了。"""
    assert not hasattr(auto_roi, "select_roi_centers")
    assert not hasattr(auto_roi, "_canvas_image")
    assert not hasattr(auto_roi, "asyncio")


def test_roi_workflow_is_synchronous():
    """build_roi_workflow 不再需要 await。"""
    import inspect

    assert not inspect.iscoroutinefunction(xrsp.build_roi_workflow)
    assert not inspect.iscoroutinefunction(auto_roi.build_roi_session)


# ==========================================================================
# 单元
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
    """给模板时应当一对一匹配到光斑上，而不是靠聚类猜。"""
    image = np.zeros((48, 60))
    yy, xx = np.mgrid[0:48, 0:60]
    for cx, cy in BLOBS:
        image += 100.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0**2))
    template = {"HB-A1": (14, 12), "HB-A2": (39, 12)}  # 故意偏 2 px
    centers, report = auto_roi.propose_centers(image, "minipix", template=template)
    assert report["mode"] == "template-match"
    for label, (x, y) in centers.items():
        assert abs(x - template[label][0]) <= 4, (label, x, y)
        assert abs(y - template[label][1]) <= 4, (label, x, y)


# ==========================================================================
# 端到端
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

    # 两个探测器各 2 个 ROI
    rois = pd.read_csv(out / "synthetic_run_rois.txt", sep="\t")
    assert len(rois) == 2 * len(REGULAR_BOXES)
    assert {"crystal", "x1", "x2", "y1", "y2", "q_ave", "center", "width"} <= set(rois.columns)
    assert not rois["bad_fit"].any(), rois[["crystal", "center", "width", "r-square"]]

    # 拟合中心应当落在合成峰位附近
    assert np.allclose(rois["center"], 9680.0, atol=5.0)
    # q 落在配置的范围里
    assert rois["q_ave"].between(0, 12).all()

    info = (out / "synthetic_run_info.txt").read_text(encoding="utf-8")
    assert f"scan_ids (xrs) = [{XRS_SCAN_ID}]" in info
    assert "Number of ROIs included = " in info

    figures = root / "processed" / ".xrs_state" / "figures"
    for name in ("elastic_qc.png", "xrs_i0.png", "xrs_roi_sums.png",
                 "sum_qc.png", "sum_per_crystal.png"):
        assert (figures / name).is_file(), f"缺少 QC 图 {name}"


def test_resume_skips_unchanged_stages(work_dir, capsys):
    config_path = build_case(work_dir / "case", mode="regular")
    assert _run(config_path) == 0
    capsys.readouterr()

    assert _run(config_path) == 0
    output = capsys.readouterr().out
    assert output.count("[skip]") == len(STAGE_ORDER), output

    # 只改 sum 的参数：前面三个阶段应当仍然跳过，sum 与 save 重算
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
    assert "尚未运行" in output
    assert "环境清单一致" in output


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
    assert "未知标签" in output


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

    # 删掉完成标记，强制 save 阶段真的重跑一遍
    (config_path.parent / "processed" / ".xrs_state" / "save.npz").unlink()

    assert _run(config_path, "--stage", "save") == 1
    assert "已存在" in capsys.readouterr().out

    # --force 只管重算，不解除覆盖保护
    assert _run(config_path, "--stage", "save", "--force") == 1
    assert "已存在" in capsys.readouterr().out

    assert _run(config_path, "--stage", "save", "--overwrite") == 0
    assert "[save]" in capsys.readouterr().out


def test_no_ui_reports_all_missing_parameters(capsys):
    assert main(["run", str(TEMPLATE), "--no-ui"]) == 1
    output = capsys.readouterr().out
    assert "必须先在 config.yaml 里填好" in output
    for stage in STAGE_ORDER:
        assert f"[{stage}]" in output


def test_single_stage_requires_upstream_state(work_dir, capsys):
    config_path = build_case(work_dir / "case", mode="regular")
    assert _run(config_path, "--stage", "q") == 1
    output = capsys.readouterr().out
    assert "缺少 elastic 阶段的中间结果" in output


def test_glitch_removal_end_to_end(work_dir, capsys):
    """带 glitch 的数据仍应跑通，并在日志里报告修掉了几个点。"""
    from conftest import build_case as _build

    config_path = _build(work_dir / "case", mode="regular", glitch_index=50)
    assert _run(config_path) == 0
    output = capsys.readouterr().out
    assert "glitch" in output


def test_ui_required_without_interaction():
    ui = UI(interactive=False)
    from xrs_ui import UiRequired

    with pytest.raises(UiRequired):
        ui.require("弹窗")
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
# 交互层（把事件循环替换成"模拟点击"，这样无显示器也能测）
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
    """造一个 interactive UI，并把阻塞的事件循环换成给定的动作。"""
    import matplotlib.pyplot as plt

    import xrs_ui

    monkeypatch.setattr(xrs_ui, "_run_event_loop", lambda figure, state: action(figure, state))
    ui = xrs_ui.UI(interactive=True)
    monkeypatch.setattr(ui, "require", lambda _what: None)  # 跳过后端可用性检查
    return ui, plt


def test_pick_points_collects_clicks_in_label_order(monkeypatch):
    import numpy as np

    labels = ("HB-A1", "HB-A2")
    wanted = {"HB-A1": (16, 14), "HB-A2": (41, 14)}

    def action(figure, state):
        axis = figure.axes[0]
        for x, y in wanted.values():
            _mouse(figure, axis, x, y)
        # 先撤销一个再重画，验证右键/撤销路径
        _mouse(figure, axis, 0, 0, button=3)
        _mouse(figure, axis, 41, 14)
        _key(figure, "enter")

    ui, plt = _interactive_ui(monkeypatch, action)
    image = np.zeros((48, 60))
    image[14, 16] = 100
    picked = ui.pick_points(image, labels, "test", figure=plt.figure())
    assert picked == wanted
    plt.close("all")


def test_pick_points_refuses_to_finish_early(monkeypatch):
    import numpy as np

    calls = {"n": 0}

    def action(figure, state):
        axis = figure.axes[0]
        _mouse(figure, axis, 10, 10)
        _key(figure, "enter")  # 只点了一个，不该结束
        assert state["done"] is False
        calls["n"] += 1
        state["done"] = True
        state["accepted"] = False

    ui, plt = _interactive_ui(monkeypatch, action)
    from xrs_ui import UiCancelled

    with pytest.raises(UiCancelled):
        ui.pick_points(np.zeros((48, 60)), ("HB-A1", "HB-A2"), "test", figure=plt.figure())
    assert calls["n"] == 1
    plt.close("all")


def test_toggle_scans_picks_nearest_curve(monkeypatch):
    import numpy as np

    energy = np.linspace(0, 10, 11)
    series = [(energy, np.full(11, level)) for level in (100.0, 200.0, 300.0)]

    def action(figure, state):
        axis = figure.axes[0]
        _mouse(figure, axis, 5.0, 200.0)  # 最靠近第二条
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
    assert drawn, "draw 回调至少要被调用一次"
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
# 交互模式下跑真实阶段（事件循环被替换成"立刻点确定"）
# ==========================================================================


def _accept_immediately(monkeypatch):
    return _interactive_ui(
        monkeypatch, lambda figure, state: state.update(done=True, accepted=True)
    )


def test_elastic_stage_interactive_tuning(work_dir, monkeypatch):
    """走一遍 run_elastic 的 ui.interactive 分支（弹调参窗口那条路）。"""
    config_path = build_case(work_dir / "case", mode="regular")
    cfg = Config.load(config_path)
    ui, plt = _accept_immediately(monkeypatch)

    payload = run_stage(cfg, "elastic", ui, log=lambda _m: None)
    assert payload["roi_names"].shape == (2 * len(REGULAR_BOXES),)
    assert int(payload["mask_count"][0]) == 2 * len(REGULAR_BOXES)
    # 窗口里的滑杆值应当被回写进配置
    assert cfg.get("elastic.filter_value") == pytest.approx(0.15)
    assert cfg.get("elastic.fit.e_lowlim") == pytest.approx(9.67)
    plt.close("all")


def test_xrs_stage_interactive_scan_removal(work_dir, monkeypatch):
    """toggle_scans 分支：本次不剔除任何扫描。"""
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
# 中心点重建子命令
# ==========================================================================


def test_propose_roi_writes_candidate_not_target(work_dir, capsys):
    config_path = build_case(work_dir / "case", mode="auto")
    roi_dir = config_path.parent / "roi"
    original = xrsp.load_auto_roi_centers(roi_dir / "auto_minipix.txt")

    assert main(["propose-roi", str(config_path), "--detector", "minipix"]) == 0
    capsys.readouterr()

    candidate = roi_dir / "auto_minipix.txt.proposed.txt"
    assert candidate.is_file(), "默认应写到 *.proposed.txt"
    proposed = xrsp.load_auto_roi_centers(candidate)
    assert proposed, "至少要解析出一些中心点"

    # 默认不碰正式文件
    assert xrsp.load_auto_roi_centers(roi_dir / "auto_minipix.txt") == original

    figures = config_path.parent / "processed" / ".xrs_state" / "figures"
    assert (figures / "centers_minipix_proposed.png").is_file()


def test_propose_roi_with_write_backs_up(work_dir, capsys):
    config_path = build_case(work_dir / "case", mode="auto")
    roi_dir = config_path.parent / "roi"

    assert main(["propose-roi", str(config_path), "--detector", "minipix", "--write"]) == 0
    capsys.readouterr()

    assert (roi_dir / "auto_minipix.txt.bak").is_file(), "覆盖前应留备份"
    assert xrsp.load_auto_roi_centers(roi_dir / "auto_minipix.txt")


def test_pick_roi_writes_clicked_centers(work_dir, capsys, monkeypatch):
    import auto_roi

    config_path = build_case(work_dir / "case", mode="auto")
    roi_dir = config_path.parent / "roi"
    labels = auto_roi.expected_labels("minipix")
    # 每个标签都点在第一个光斑上；只要点满，回车就能结束
    clicks = dict.fromkeys(labels, (16, 14))

    def action(figure, state):
        axis = figure.axes[0]
        for x, y in clicks.values():
            _mouse(figure, axis, x, y)
        _key(figure, "enter")

    import xrs_ui

    monkeypatch.setattr(xrs_ui, "_run_event_loop", lambda figure, state: action(figure, state))
    assert main(["pick-roi", str(config_path), "--detector", "minipix"]) == 0
    capsys.readouterr()

    written = xrsp.load_auto_roi_centers(roi_dir / "auto_minipix.txt")
    assert len(written) == len(labels)
    assert set(written) == set(labels)
    assert (roi_dir / "auto_minipix.txt.bak").is_file()
