"""交互层：缺参数时弹窗问你，调完立刻写回 YAML。

这个模块替代了原来 notebook 里"改一个 cell 顶部的变量 → 重跑"的循环。
三类交互：

* :meth:`UI.ask` / :meth:`UI.ask_list` —— 终端问答，用于路径、扫描编号这类
  不适合画图的标量。
* :meth:`UI.adjust` —— matplotlib 滑杆/单选/多选窗口，配一张实时重画的 QC 图，
  用于 ``filter_value``、``q_range``、``modules`` 这类"看效果调"的参数。
* :meth:`UI.pick_points` / :meth:`UI.toggle_scans` —— 图像上点选，用于
  重建 ROI 中心点和剔除坏扫描。

``--no-ui`` 时调用任何交互方法都会抛 :class:`UiRequired`，由上层拼出
"你还缺哪些参数"的报错。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from xrs_config import is_blank

#: matplotlib 默认字体没有中文字形，按优先级挑一个能用的。
_CJK_FONTS = (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Noto Sans SC",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Heiti SC",
)

_NON_INTERACTIVE_BACKENDS = {
    "agg",
    "cairo",
    "pdf",
    "pgf",
    "ps",
    "svg",
    "template",
}

_CANDIDATE_BACKENDS = ("TkAgg", "QtAgg", "Qt5Agg")


class UiRequired(Exception):
    """需要交互，但当前处于 ``--no-ui`` 或没有可用图形后端。"""


class UiCancelled(Exception):
    """用户在交互界面里按了"中止"。"""


# --------------------------------------------------------------------------
# 控件描述
# --------------------------------------------------------------------------


@dataclass
class SliderSpec:
    """一个滑杆。``step`` 为 1 且 ``integer`` 为真时取整。"""

    key: str
    label: str
    low: float
    high: float
    value: float
    step: float = 0.01
    integer: bool = False


@dataclass
class ChoiceSpec:
    """一组单选框（``multi=False``）或复选框（``multi=True``）。"""

    key: str
    label: str
    options: tuple
    value: object = None
    multi: bool = False

    def __post_init__(self):
        if self.value is None:
            self.value = [] if self.multi else self.options[0]


# --------------------------------------------------------------------------
# matplotlib 设置
# --------------------------------------------------------------------------


def configure_matplotlib(dpi: int = 150) -> None:
    """设置中文字体和分辨率。不切换后端。"""
    import logging

    import matplotlib
    from matplotlib import font_manager

    # 缺字重时 matplotlib 会对每个字形刷一条 findfont 警告，直接把日志压掉
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    try:
        installed = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:  # 字体缓存损坏时不该拖垮整个流程
        installed = set()
    usable = [name for name in _CJK_FONTS if name in installed]
    if usable:
        matplotlib.rcParams["font.sans-serif"] = usable + list(
            matplotlib.rcParams.get("font.sans-serif", [])
        )
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = dpi
    matplotlib.rcParams["savefig.dpi"] = dpi
    matplotlib.rcParams["savefig.bbox"] = "tight"


def has_display() -> bool:
    """粗略判断当前进程能不能开窗口。"""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    # Windows 上的 MSYS/Cygwin 会设这两个变量
    if os.environ.get("TERM_PROGRAM"):
        return True
    return False


def ensure_interactive_backend() -> str:
    """确保 pyplot 用的是能开窗口的后端，返回后端名。

    失败时抛 :class:`UiRequired`，由调用方决定是报错还是退回 ``--no-ui``。
    """
    import matplotlib
    import matplotlib.pyplot as plt

    current = matplotlib.get_backend()
    if current.lower() not in _NON_INTERACTIVE_BACKENDS:
        return current
    if not has_display():
        raise UiRequired(
            f"当前后端是 {current}，而且没有检测到显示器（DISPLAY/WAYLAND_DISPLAY 都为空）。"
            "纯 SSH 会话里无法弹窗，请用 --no-ui 并手工填好参数。"
        )
    for candidate in _CANDIDATE_BACKENDS:
        try:
            plt.switch_backend(candidate)
        except Exception:
            continue
        if matplotlib.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS:
            return matplotlib.get_backend()
    raise UiRequired(
        f"当前后端是 {current}，且切换到 {'/'.join(_CANDIDATE_BACKENDS)} 都失败。"
        "请安装 python3-tk（或 PyQt）后重试，或改用 --no-ui。"
    )


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------


class UI:
    """终端问答 + matplotlib 窗口的集合。

    ``interactive=False`` 时（``--no-ui``）所有交互方法都抛 :class:`UiRequired`。
    """

    def __init__(self, interactive: bool = True, dpi: int = 150):
        self.interactive = interactive
        self.dpi = dpi
        self._backend_checked = False

    # -- 基础设施 -----------------------------------------------------------

    def require(self, what: str) -> None:
        if not self.interactive:
            raise UiRequired(f"{what} 需要交互，但当前是 --no-ui")
        if not self._backend_checked:
            self._backend_checked = True
            ensure_interactive_backend()

    def note(self, message: str) -> None:
        print(message)

    # -- 终端问答 -----------------------------------------------------------

    def ask(self, label: str, current=None, cast=str, choices=None, help_text: str = ""):
        """问一个标量。直接回车表示沿用 ``current``。"""
        self.require(label)
        if help_text:
            print(f"  {help_text}")
        while True:
            suffix = f" [{current}]" if not is_blank(current) else ""
            try:
                raw = input(f"{label}{suffix}: ").strip()
            except EOFError as exc:
                raise UiCancelled(f"读取 {label} 时遇到 EOF") from exc
            if not raw:
                if not is_blank(current):
                    return current
                print("  不能为空，请重新输入。")
                continue
            if choices is not None and raw not in choices:
                print(f"  只能是 {list(choices)} 之一。")
                continue
            try:
                return cast(raw)
            except (TypeError, ValueError):
                print(f"  无法把 {raw!r} 解析成 {getattr(cast, '__name__', '所需类型')}。")

    def ask_list(self, label: str, current=None, cast=int, help_text: str = ""):
        """问一个列表。``"1,2,3"`` / ``"1 2 3"`` 都行；直接回车表示不改。"""
        self.require(label)
        if help_text:
            print(f"  {help_text}")
        while True:
            suffix = f" [{current}]" if not is_blank(current) else ""
            try:
                raw = input(f"{label}{suffix}: ").strip()
            except EOFError as exc:
                raise UiCancelled(f"读取 {label} 时遇到 EOF") from exc
            if not raw:
                if not is_blank(current):
                    return list(current)
                return []
            if raw.lower() in {"none", "null", "空"}:
                return []
            pieces = [piece for piece in raw.replace(",", " ").split() if piece]
            try:
                return [cast(piece) for piece in pieces]
            except (TypeError, ValueError):
                print(f"  无法把 {raw!r} 解析成 {getattr(cast, '__name__', '列表')} 列表。")

    def confirm(self, message: str, default: bool = True) -> bool:
        """y/N 确认。``--no-ui`` 时直接返回 ``default``（脚本化运行不该卡住）。"""
        if not self.interactive:
            return default
        hint = "Y/n" if default else "y/N"
        try:
            raw = input(f"{message} [{hint}]: ").strip().lower()
        except EOFError:
            return default
        if not raw:
            return default
        return raw in {"y", "yes"}

    # -- 滑杆/单选框窗口 ----------------------------------------------------

    def adjust(self, fig, controls, draw, title: str = "", left: float = 0.70):
        """在 ``fig`` 右侧加一列控件，实时调参。

        参数
        ----
        fig
            已经画好 QC 图的 Figure。
        controls
            :class:`SliderSpec` / :class:`ChoiceSpec` 的列表。
        draw
            回调 ``draw(values: dict) -> None``，负责按新参数重画。

        返回用户确认后的取值字典。窗口被关掉、或点了"中止"时抛
        :class:`UiCancelled`。
        """
        self.require(title or "调参窗口")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

        if not controls:
            raise ValueError("adjust() 至少需要一个控件")

        values = {spec.key: spec.value for spec in controls}
        state = {"done": False, "accepted": False}

        top, bottom = 0.90, 0.16
        slot = (top - bottom) / len(controls)
        widgets = []
        for index, spec in enumerate(controls):
            y = top - (index + 1) * slot
            height = slot * 0.78
            axes = fig.add_axes([left, y, 0.27, height])
            if isinstance(spec, SliderSpec):
                axes.set_title(spec.label, fontsize=9, loc="left")
                # 给标题腾地方
                position = axes.get_position()
                axes.set_position([position.x0, position.y0, position.width, position.height])
                widgets.append(
                    (
                        spec,
                        Slider(
                            axes,
                            "",
                            spec.low,
                            spec.high,
                            valinit=spec.value,
                            valstep=spec.step,
                        ),
                    )
                )
            elif isinstance(spec, ChoiceSpec):
                axes.set_title(spec.label, fontsize=9, loc="left")
                if spec.multi:
                    active = [option in spec.value for option in spec.options]
                    widgets.append(
                        (spec, CheckButtons(axes, spec.options, actives=active))
                    )
                else:
                    active = (
                        spec.options.index(spec.value)
                        if spec.value in spec.options
                        else 0
                    )
                    widgets.append(
                        (spec, RadioButtons(axes, spec.options, active=active))
                    )
            else:  # pragma: no cover - 编程错误
                raise TypeError(f"不认识的控件类型：{type(spec).__name__}")

        def collect() -> dict:
            for spec, widget in widgets:
                if isinstance(spec, SliderSpec):
                    value = widget.val
                    values[spec.key] = (
                        int(round(value)) if spec.integer else float(value)
                    )
                elif spec.multi:
                    status = widget.get_status()
                    values[spec.key] = [
                        option
                        for option, on in zip(spec.options, status)
                        if on
                    ]
                else:
                    values[spec.key] = widget.value_selected
            return values

        def redraw(_event=None) -> None:
            try:
                draw(collect())
            except Exception as exc:  # 参数组合非法时不该把窗口弄崩
                print(f"  [重画失败] {exc.__class__.__name__}: {exc}")
            fig.canvas.draw_idle()

        for _spec, widget in widgets:
            if hasattr(widget, "on_changed"):
                widget.on_changed(redraw)

        ax_ok = fig.add_axes([left, 0.10, 0.12, 0.05])
        ax_skip = fig.add_axes([left + 0.14, 0.10, 0.12, 0.05])
        ax_abort = fig.add_axes([left, 0.03, 0.26, 0.05])
        button_ok = Button(ax_ok, "保存并继续", color="0.85", hovercolor="0.75")
        button_skip = Button(ax_skip, "沿用原值", color="0.92", hovercolor="0.85")
        button_abort = Button(ax_abort, "中止", color="0.95", hovercolor="0.85")

        def finish(accepted: bool):
            def handler(_event):
                state["done"] = True
                state["accepted"] = accepted

            return handler

        button_ok.on_clicked(finish(True))
        button_skip.on_clicked(finish(False))
        button_abort.on_clicked(finish(False))

        if title:
            fig.suptitle(title, fontsize=13)

        redraw()
        fig.canvas.draw_idle()
        _run_event_loop(fig, state)

        if not state["accepted"]:
            raise UiCancelled(f"{title or '调参'}被取消")
        collect()
        # 图窗留给调用方：它通常还要把最终状态存成 QC PNG
        return values

    # -- 图像上点选 ---------------------------------------------------------

    def pick_points(self, image, labels, title: str, figure=None, figsize=(11, 11)):
        """在图像上按 ``labels`` 顺序点中心点。

        左键落点，右键或 Backspace 撤销，回车结束（必须点满），Esc 放弃。
        返回 ``{label: (x, y)}``。图窗不关闭，由调用方负责关（通常还要存 QC 图）。
        """
        self.require("ROI 中心点点选")
        import matplotlib.pyplot as plt

        import numpy as np

        if figure is None:
            figure = plt.figure(figsize=figsize)
        figure.clear()
        axis = figure.add_subplot(111)
        axis.imshow(np.log1p(np.clip(np.asarray(image, dtype=float), 0, None)), cmap="gray")
        axis.set_title(
            f"{title}\n左键落点 / 右键撤销 / 回车完成 / Esc 放弃",
            fontsize=11,
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")

        placed: list[tuple[str, int, int]] = []
        state = {"done": False, "accepted": False}
        height, width = np.asarray(image).shape

        def redraw() -> None:
            for artist in list(axis.lines) + list(axis.texts):
                artist.remove()
            for roi_label, x, y in placed:
                axis.axvline(x, color="#ff3030", lw=0.8, alpha=0.7)
                axis.axhline(y, color="#ff3030", lw=0.8, alpha=0.7)
                axis.plot(x, y, "o", color="#ff3030", ms=4)
                axis.text(x + 5, y - 5, roi_label, color="#ff3030", fontsize=8)
            remaining = len(labels) - len(placed)
            figure.suptitle(
                f"{title}   已选 {len(placed)}/{len(labels)}"
                + (f"，下一个 {labels[len(placed)]}" if remaining else "  —— 已点满，回车结束"),
                fontsize=12,
            )
            figure.canvas.draw_idle()

        def on_click(event) -> None:
            if event.inaxes is not axis or event.xdata is None:
                return
            if event.button == 3:
                if placed:
                    placed.pop()
                    redraw()
                return
            if event.button != 1:
                return
            if len(placed) >= len(labels):
                print("  已经点满了，回车结束（右键可撤销）")
                return
            x = int(min(max(round(event.xdata), 0), width - 1))
            y = int(min(max(round(event.ydata), 0), height - 1))
            placed.append((labels[len(placed)], x, y))
            redraw()

        def on_key(event) -> None:
            if event.key in {"backspace", "delete"} and placed:
                placed.pop()
                redraw()
            elif event.key == "enter":
                if len(placed) == len(labels):
                    state["done"] = True
                    state["accepted"] = True
                else:
                    print(f"  还差 {len(labels) - len(placed)} 个点：{labels[len(placed):]}")
            elif event.key == "escape":
                state["done"] = True
                state["accepted"] = False

        figure.canvas.mpl_connect("button_press_event", on_click)
        figure.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        _run_event_loop(figure, state)

        if not state["accepted"]:
            raise UiCancelled(f"{title} 的中心点选择被取消")
        return {roi_label: (x, y) for roi_label, x, y in placed}

    def toggle_scans(self, series, excluded, title: str, figure=None, figsize=(12, 6)):
        """点曲线来切换"剔除/保留"，替代手填 ``index_to_remove``。

        ``series`` 是 ``[(x, y), ...]``，每个扫描一条曲线——各个扫描的能量轴
        可以不一样。返回剔除的序号列表。图窗不关闭，由调用方负责关。
        """
        self.require("坏扫描剔除")
        import matplotlib.pyplot as plt
        import numpy as np

        if figure is None:
            figure = plt.figure(figsize=figsize)
        figure.clear()
        axis = figure.add_subplot(111)
        prepared = []
        for x_values, y_values in series:
            prepared.append(
                (
                    np.asarray(x_values, dtype=float),
                    np.abs(np.asarray(y_values, dtype=float)),
                )
            )
        dropped = set(int(index) for index in excluded)
        state = {"done": False, "accepted": False}

        def redraw() -> None:
            axis.clear()
            for index, (x_values, y_values) in enumerate(prepared):
                is_out = index in dropped
                axis.plot(
                    x_values,
                    y_values,
                    lw=2.0 if is_out else 0.9,
                    alpha=0.35 if is_out else 0.9,
                    color="0.6" if is_out else None,
                    linestyle="--" if is_out else "-",
                    label=f"scan {index}" + ("  [剔除]" if is_out else ""),
                )
            axis.set_yscale("log")
            axis.set_xlabel("energy")
            axis.set_ylabel("I0")
            axis.legend(fontsize=8, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
            axis.set_title(
                f"{title}\n点曲线切换剔除状态；当前剔除 {sorted(dropped) or '无'}",
                fontsize=11,
            )
            figure.canvas.draw_idle()

        def on_click(event) -> None:
            if event.inaxes is not axis or event.xdata is None or event.ydata is None:
                return
            if event.button != 1 or not prepared:
                return
            # 在点击的 x 处比较各曲线离点击点的纵向距离，取最近的一条
            best_index, best_distance = None, np.inf
            for index, (x_values, y_values) in enumerate(prepared):
                if x_values.size == 0 or y_values.size != x_values.size:
                    continue
                value = float(np.interp(event.xdata, x_values, y_values))
                span = float(np.nanmax(y_values) - np.nanmin(y_values)) or 1.0
                distance = abs(value - event.ydata) / span
                if distance < best_distance:
                    best_index, best_distance = index, distance
            if best_index is None:
                return
            if best_index in dropped:
                dropped.discard(best_index)
            else:
                dropped.add(best_index)
            redraw()

        def on_key(event) -> None:
            if event.key == "enter":
                state["done"] = True
                state["accepted"] = True
            elif event.key == "escape":
                state["done"] = True
                state["accepted"] = False

        ax_ok = figure.add_axes([0.02, 0.01, 0.13, 0.05])
        ax_keep = figure.add_axes([0.17, 0.01, 0.13, 0.05])
        from matplotlib.widgets import Button

        Button(ax_ok, "完成", color="0.85", hovercolor="0.75").on_clicked(
            lambda _e: (state.update(done=True, accepted=True))
        )
        Button(ax_keep, "全部保留", color="0.92", hovercolor="0.85").on_clicked(
            lambda _e: (dropped.clear(), redraw())
        )

        figure.canvas.mpl_connect("button_press_event", on_click)
        figure.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        _run_event_loop(figure, state)

        if not state["accepted"]:
            raise UiCancelled(f"{title} 被取消")
        return sorted(dropped)

    # -- 内部 ---------------------------------------------------------------


def _run_event_loop(figure, state: dict) -> None:
    """转到用户点完按钮/按完键。窗口被关掉时按取消处理。"""
    import matplotlib.pyplot as plt

    while not state["done"]:
        if not plt.fignum_exists(figure.number):
            state["done"] = True
            state["accepted"] = False
            return
        try:
            plt.pause(0.05)
        except Exception:
            # 窗口被销毁时 pause 会抛，等价于取消
            state["done"] = True
            state["accepted"] = False
            return
