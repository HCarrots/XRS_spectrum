"""xrs_ui implementation."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from xrs_config import is_blank

# English note.
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


def _clamp_rectangle(x0, y0, x1, y1, image_shape):
    """Convert two drag points to non-empty, half-open image bounds."""
    import numpy as np

    height, width = image_shape
    left = max(0, min(width, int(np.floor(min(x0, x1)))))
    right = max(0, min(width, int(np.ceil(max(x0, x1)))))
    top = max(0, min(height, int(np.floor(min(y0, y1)))))
    bottom = max(0, min(height, int(np.ceil(max(y0, y1)))))
    if right <= left or bottom <= top:
        raise ValueError("Rectangle must have a non-zero area")
    return left, right, top, bottom


class UiRequired(Exception):
    """Implementation notes for UiRequired."""


class UiCancelled(Exception):
    """Implementation notes for UiCancelled."""


# --------------------------------------------------------------------------
# English note.
# --------------------------------------------------------------------------


@dataclass
class SliderSpec:
    """Implementation notes for SliderSpec."""

    key: str
    label: str
    low: float
    high: float
    value: float
    step: float = 0.01
    integer: bool = False


@dataclass
class ChoiceSpec:
    """Implementation notes for ChoiceSpec."""

    key: str
    label: str
    options: tuple
    value: object = None
    multi: bool = False

    def __post_init__(self):
        if self.value is None:
            self.value = [] if self.multi else self.options[0]


# --------------------------------------------------------------------------
# English note.
# --------------------------------------------------------------------------


def configure_matplotlib(dpi: int = 150) -> None:
    """Implementation notes for configure_matplotlib."""
    import logging

    import matplotlib
    from matplotlib import font_manager

# English note.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    try:
        installed = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:  # A broken font cache must not stop processing.
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
    """Implementation notes for has_display."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
# English note.
    if os.environ.get("TERM_PROGRAM"):
        return True
    return False


def ensure_interactive_backend() -> str:
    """Implementation notes for ensure_interactive_backend."""
    import matplotlib
    import matplotlib.pyplot as plt

    current = matplotlib.get_backend()
    if current.lower() not in _NON_INTERACTIVE_BACKENDS:
        return current
    if not has_display():
        raise UiRequired(
            f"Backend {current} is non-interactive and no display was detected. "
            "Use --no-ui with a complete configuration in a headless session."
        )
    for candidate in _CANDIDATE_BACKENDS:
        try:
            plt.switch_backend(candidate)
        except Exception:
            continue
        if matplotlib.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS:
            return matplotlib.get_backend()
    raise UiRequired(
        f"Backend {current} is non-interactive and none of "
        f"{'/'.join(_CANDIDATE_BACKENDS)} could be enabled. Install Tk or Qt, or use --no-ui."
    )


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------


class UI:
    """Implementation notes for UI."""

    def __init__(self, interactive: bool = True, dpi: int = 150):
        self.interactive = interactive
        self.dpi = dpi
        self._backend_checked = False

# English note.

    def require(self, what: str) -> None:
        if not self.interactive:
            raise UiRequired(f"{what} requires interaction, but --no-ui is active")
        if not self._backend_checked:
            self._backend_checked = True
            ensure_interactive_backend()

    def note(self, message: str) -> None:
        print(message)

# English note.

    def ask(self, label: str, current=None, cast=str, choices=None, help_text: str = ""):
        """Implementation notes for ask."""
        self.require(label)
        if help_text:
            print(f"  {help_text}")
        while True:
            suffix = f" [{current}]" if not is_blank(current) else ""
            try:
                raw = input(f"{label}{suffix}: ").strip()
            except EOFError as exc:
                raise UiCancelled(f"Reached EOF while reading {label}") from exc
            if not raw:
                if not is_blank(current):
                    return current
                print("  A value is required.")
                continue
            if choices is not None and raw not in choices:
                print(f"  Choose one of {list(choices)}.")
                continue
            try:
                return cast(raw)
            except (TypeError, ValueError):
                print(f"  Could not parse {raw!r} as {getattr(cast, '__name__', 'the required type')}.")

    def ask_list(self, label: str, current=None, cast=int, help_text: str = ""):
        """Implementation notes for ask_list."""
        self.require(label)
        if help_text:
            print(f"  {help_text}")
        while True:
            suffix = f" [{current}]" if not is_blank(current) else ""
            try:
                raw = input(f"{label}{suffix}: ").strip()
            except EOFError as exc:
                raise UiCancelled(f"Reached EOF while reading {label}") from exc
            if not raw:
                if not is_blank(current):
                    return list(current)
                return []
            if raw.lower() in {"none", "null"}:
                return []
            pieces = [piece for piece in raw.replace(",", " ").split() if piece]
            try:
                return [cast(piece) for piece in pieces]
            except (TypeError, ValueError):
                print(f"  Could not parse {raw!r} as a list of {getattr(cast, '__name__', 'values')}.")

    def confirm(self, message: str, default: bool = True) -> bool:
        """Implementation notes for confirm."""
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

    def ask_optional(self, label: str, help_text: str = ""):
        """Return optional text input, or None when the user presses Enter."""
        if not self.interactive:
            return None
        self.require(label)
        if help_text:
            print(f"  {help_text}")
        try:
            value = input(f"{label}: ").strip()
        except EOFError:
            return None
        return value or None

    def show_figure(self, figure, title: str, button_label: str = "Continue") -> None:
        """Show an inspection figure until the user continues or closes it."""
        self.require(title)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        state = {"done": False, "accepted": False}
        button_axis = figure.add_axes([0.43, 0.02, 0.14, 0.055])
        continue_button = Button(button_axis, button_label, color="0.85")
        continue_button.on_clicked(
            lambda _event: state.update(done=True, accepted=True)
        )
        _keep_widgets_alive(figure, continue_button)
        figure.suptitle(title)
        figure.canvas.mpl_connect(
            "key_press_event",
            lambda event: state.update(done=True, accepted=True)
            if event.key in {"enter", "escape"} else None,
        )
        _run_event_loop(figure, state)
        plt.close(figure)

    def preview_images(self, images, title="Elastic detector preview") -> None:
        """Show summed detector images and wait for explicit confirmation."""
        self.require(title)
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.widgets import Button

        figure, axes = plt.subplots(1, len(images), figsize=(14, 7), squeeze=False)
        for axis, (name, image) in zip(axes[0], images.items()):
            axis.imshow(np.log1p(np.clip(np.asarray(image, dtype=float), 0, None)), cmap="gray")
            axis.set_title(str(name).title())
            axis.set_xlabel("x")
            axis.set_ylabel("y")
        state = {"done": False, "accepted": False}
        button_axis = figure.add_axes([0.43, 0.02, 0.14, 0.05])
        continue_button = Button(button_axis, "Continue", color="0.85")
        continue_button.on_clicked(
            lambda _event: state.update(done=True, accepted=True)
        )
        _keep_widgets_alive(figure, continue_button)
        figure.suptitle(title)
        figure.canvas.mpl_connect(
            "key_press_event",
            lambda event: state.update(done=True, accepted=True)
            if event.key == "enter" else None,
        )
        _run_event_loop(figure, state)
        plt.close(figure)
        if not state["accepted"]:
            raise UiCancelled(f"{title} was cancelled")

    def review_geometry(self, image, labels, bounds, title: str) -> bool:
        """Display stored ROI geometry; return True to reuse or False to redraw."""
        self.require(title)
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Button

        figure, axis = plt.subplots(figsize=(11, 9))
        axis.imshow(np.log1p(np.clip(np.asarray(image, dtype=float), 0, None)), cmap="gray")
        for label, (y1, y2, x1, x2) in zip(labels, bounds):
            axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     edgecolor="red", facecolor="none", lw=0.9))
            axis.text(x1, y1, str(label), color="yellow", fontsize=7)
        axis.set_title(f"{title}\nEnter: use existing | R: redraw | Esc: cancel")
        state = {"done": False, "accepted": False, "redraw": False}
        use_axis = figure.add_axes([0.34, 0.02, 0.14, 0.05])
        redraw_axis = figure.add_axes([0.52, 0.02, 0.14, 0.05])
        use_button = Button(use_axis, "Use existing", color="0.85")
        redraw_button = Button(redraw_axis, "Redraw", color="0.92")
        use_button.on_clicked(
            lambda _event: state.update(done=True, accepted=True)
        )
        redraw_button.on_clicked(
            lambda _event: state.update(done=True, accepted=True, redraw=True)
        )
        _keep_widgets_alive(figure, use_button, redraw_button)
        def on_key(event):
            if event.key == "enter":
                state.update(done=True, accepted=True)
            elif event.key in {"r", "R"}:
                state.update(done=True, accepted=True, redraw=True)
            elif event.key == "escape":
                state.update(done=True, accepted=False)
        figure.canvas.mpl_connect("key_press_event", on_key)
        _run_event_loop(figure, state)
        plt.close(figure)
        if not state["accepted"]:
            raise UiCancelled(f"{title} was cancelled")
        return not state["redraw"]

    def pick_rectangles(self, image, labels, title: str, figure=None, figsize=(11, 11)):
        """Draw labeled rectangular ROIs in canonical order."""
        self.require("rectangular ROI selection")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import RectangleSelector

        if figure is None:
            figure = plt.figure(figsize=figsize)
        figure.clear()
        _disable_default_key_handler(figure)
        axis = figure.add_subplot(111)
        source = np.asarray(image, dtype=float)
        height, width = source.shape
        axis.imshow(np.log1p(np.clip(source, 0, None)), cmap="gray")
        placed = []
        actions = []
        roi_artists = []
        state = {"done": False, "accepted": False}

        def next_label():
            position = len(actions)
            return labels[position] if position < len(labels) else None

        def redraw():
            while roi_artists:
                artist = roi_artists.pop()
                if artist.axes is not None:
                    artist.remove()
            for label, x1, x2, y1, y2 in placed:
                rectangle = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    edgecolor="red", facecolor="none", lw=1.2,
                )
                roi_artists.append(axis.add_patch(rectangle))
                roi_artists.append(
                    axis.text(x1, y1, label, color="yellow", fontsize=7)
                )
            upcoming = next_label()
            status = f"Next: {upcoming}" if upcoming else "All labels handled"
            figure.suptitle(
                f"{title} | {status}\n"
                "Drag: draw | S: skip | C/U/Backspace: cancel last | "
                "R: redraw last | A: start over | Enter: save and continue | "
                "Esc: cancel detector"
            )
            figure.canvas.draw_idle()

        def undo_last():
            if not actions:
                return
            action, _label = actions.pop()
            if action == "place":
                placed.pop()
            redraw()

        def on_select(click, release):
            label = next_label()
            if label is None or click.xdata is None or release.xdata is None:
                return
            try:
                x1, x2, y1, y2 = _clamp_rectangle(
                    click.xdata, click.ydata, release.xdata, release.ydata,
                    (height, width),
                )
            except ValueError:
                print("  Rectangle must have a non-zero area.")
                return
            placed.append((str(label), x1, x2, y1, y2))
            actions.append(("place", str(label)))
            redraw()

        selector = RectangleSelector(
            axis,
            on_select,
            useblit=True,
            button=[1],
            minspanx=1,
            minspany=1,
            spancoords="data",
            props={
                "facecolor": "tab:orange",
                "edgecolor": "yellow",
                "alpha": 0.30,
                "fill": True,
            },
        )
        _keep_widgets_alive(figure, selector)

        def on_key(event):
            if event.key in {"backspace", "delete", "c", "C", "u", "U"}:
                undo_last()
            elif event.key in {"s", "S"} and next_label() is not None:
                actions.append(("skip", str(next_label())))
                redraw()
            elif event.key in {"r", "R"}:
                undo_last()
            elif event.key in {"a", "A"}:
                placed.clear()
                actions.clear()
                redraw()
            elif event.key == "enter":
                state.update(done=True, accepted=True)
            elif event.key == "escape":
                state.update(done=True, accepted=False)
        figure.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        _run_event_loop(figure, state)
        selector.set_active(False)
        plt.close(figure)
        if not state["accepted"]:
            raise UiCancelled(f"{title} was cancelled")
        return list(placed)

# English note.

    def adjust(self, fig, controls, draw, title: str = "", left: float = 0.70):
        """Implementation notes for adjust."""
        self.require(title or "parameter adjustment")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

        if not controls:
            raise ValueError("adjust() requires at least one control")

        values = {spec.key: spec.value for spec in controls}
        state = {"done": False, "accepted": False, "keep_original": False}

        top, bottom = 0.90, 0.16
        slot = (top - bottom) / len(controls)
        widgets = []
        for index, spec in enumerate(controls):
            y = top - (index + 1) * slot
            height = slot * 0.78
            axes = fig.add_axes([left, y, 0.27, height])
            if isinstance(spec, SliderSpec):
                axes.set_title(spec.label, fontsize=9, loc="left")
# English note.
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
            else:  # pragma: no cover - programmer error
                raise TypeError(f"Unsupported control type: {type(spec).__name__}")

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
            except Exception as exc:  # Keep the adjustment window usable.
                print(f"  [REDRAW FAILED] {exc.__class__.__name__}: {exc}")
            fig.canvas.draw_idle()

        for _spec, widget in widgets:
            if hasattr(widget, "on_changed"):
                widget.on_changed(redraw)

        ax_ok = fig.add_axes([left, 0.10, 0.12, 0.05])
        ax_skip = fig.add_axes([left + 0.14, 0.10, 0.12, 0.05])
        ax_abort = fig.add_axes([left, 0.03, 0.26, 0.05])
        button_ok = Button(ax_ok, "Save and continue", color="0.85", hovercolor="0.75")
        button_skip = Button(ax_skip, "Keep values", color="0.92", hovercolor="0.85")
        button_abort = Button(ax_abort, "Cancel", color="0.95", hovercolor="0.85")

        def finish(accepted: bool, keep_original: bool = False):
            def handler(_event):
                state["done"] = True
                state["accepted"] = accepted
                state["keep_original"] = keep_original

            return handler

        button_ok.on_clicked(finish(True))
        button_skip.on_clicked(finish(True, keep_original=True))
        button_abort.on_clicked(finish(False))

        if title:
            fig.suptitle(title, fontsize=13)

        redraw()
        fig.canvas.draw_idle()
        _run_event_loop(fig, state)

        if not state["accepted"]:
            raise UiCancelled(f"{title or 'Parameter adjustment'} was cancelled")
        if state["keep_original"]:
            return {spec.key: spec.value for spec in controls}
        collect()
# English note.
        return values

# English note.

    def pick_points(self, image, labels, title: str, figure=None, figsize=(11, 11)):
        """Pick ROI centers in label order, with undo and skip support."""
        self.require("ROI center selection")
        import matplotlib.pyplot as plt

        import numpy as np

        if figure is None:
            figure = plt.figure(figsize=figsize)
        figure.clear()
        _disable_default_key_handler(figure)
        axis = figure.add_subplot(111)
        axis.imshow(np.log1p(np.clip(np.asarray(image, dtype=float), 0, None)), cmap="gray")
        axis.set_title(
            f"{title}\nLeft click: place | Right click/Backspace: undo | "
            "S: skip | Enter: finish | Esc: cancel",
            fontsize=11,
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")

        placed: list[tuple[str, int, int]] = []
        actions: list[tuple[str, object]] = []
        state = {"done": False, "accepted": False}
        height, width = np.asarray(image).shape

        def next_label():
            return labels[len(actions)] if len(actions) < len(labels) else None

        def redraw() -> None:
            for artist in list(axis.lines) + list(axis.texts):
                artist.remove()
            for roi_label, x, y in placed:
                axis.axvline(x, color="#ff3030", lw=0.8, alpha=0.7)
                axis.axhline(y, color="#ff3030", lw=0.8, alpha=0.7)
                axis.plot(x, y, "o", color="#ff3030", ms=4)
                axis.text(x + 5, y - 5, roi_label, color="#ff3030", fontsize=8)
            remaining = len(labels) - len(actions)
            upcoming = next_label()
            figure.suptitle(
                f"{title} | selected {len(placed)}, handled {len(actions)}/{len(labels)}"
                + (f" | next: {upcoming}" if remaining else " | all labels handled"),
                fontsize=12,
            )
            figure.canvas.draw_idle()

        def on_click(event) -> None:
            if event.inaxes is not axis or event.xdata is None:
                return
            if event.button == 3:
                if actions:
                    action, payload = actions.pop()
                    if action == "place":
                        placed.pop()
                    redraw()
                return
            if event.button != 1:
                return
            label = next_label()
            if label is None:
                print("  All labels have been handled; press Enter to finish.")
                return
            x = int(min(max(round(event.xdata), 0), width - 1))
            y = int(min(max(round(event.ydata), 0), height - 1))
            placed.append((str(label), x, y))
            actions.append(("place", (str(label), x, y)))
            redraw()

        def on_key(event) -> None:
            if event.key in {"backspace", "delete"} and actions:
                action, _payload = actions.pop()
                if action == "place":
                    placed.pop()
                redraw()
            elif event.key in {"s", "S"} and next_label() is not None:
                actions.append(("skip", str(next_label())))
                redraw()
            elif event.key == "enter":
                if placed:
                    state["done"] = True
                    state["accepted"] = True
                else:
                    print("  Select at least one center before finishing.")
            elif event.key == "escape":
                state["done"] = True
                state["accepted"] = False

        figure.canvas.mpl_connect("button_press_event", on_click)
        figure.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        _run_event_loop(figure, state)

        if not state["accepted"]:
            raise UiCancelled(f"{title} was cancelled")
        return {roi_label: (x, y) for roi_label, x, y in placed}

    def toggle_scans(self, series, excluded, title: str, figure=None, figsize=(12, 6)):
        """Implementation notes for toggle_scans."""
        self.require("scan exclusion")
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
                    label=f"scan {index}" + ("  [excluded]" if is_out else ""),
                )
            axis.set_yscale("log")
            axis.set_xlabel("energy")
            axis.set_ylabel("I0")
            axis.legend(fontsize=8, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5))
            axis.set_title(
                f"{title}\nClick a curve to toggle it; excluded: {sorted(dropped) or 'none'}",
                fontsize=11,
            )
            figure.canvas.draw_idle()

        def on_click(event) -> None:
            if event.inaxes is not axis or event.xdata is None or event.ydata is None:
                return
            if event.button != 1 or not prepared:
                return
# English note.
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

        done_button = Button(ax_ok, "Done", color="0.85", hovercolor="0.75")
        keep_button = Button(ax_keep, "Keep all", color="0.92", hovercolor="0.85")
        done_button.on_clicked(
            lambda _e: (state.update(done=True, accepted=True))
        )
        keep_button.on_clicked(
            lambda _e: (dropped.clear(), redraw())
        )
        _keep_widgets_alive(figure, done_button, keep_button)

        figure.canvas.mpl_connect("button_press_event", on_click)
        figure.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        _run_event_loop(figure, state)

        if not state["accepted"]:
            raise UiCancelled(f"{title} was cancelled")
        return sorted(dropped)

# English note.


def _keep_widgets_alive(figure, *widgets) -> None:
    """Retain Matplotlib widgets for as long as their figure exists."""
    retained = getattr(figure, "_xrs_widgets", None)
    if retained is None:
        retained = []
        figure._xrs_widgets = retained
    retained.extend(widgets)


def _disable_default_key_handler(figure) -> None:
    """Disable Matplotlib shortcuts that conflict with editor key bindings."""
    manager = getattr(figure.canvas, "manager", None)
    callback_id = getattr(manager, "key_press_handler_id", None)
    if callback_id is not None:
        figure.canvas.mpl_disconnect(callback_id)
        manager.key_press_handler_id = None


def _run_event_loop(figure, state: dict) -> None:
    """Implementation notes for _run_event_loop."""
    import matplotlib.pyplot as plt

    while not state["done"]:
        if not plt.fignum_exists(figure.number):
            state["done"] = True
            state["accepted"] = False
            return
        try:
            plt.pause(0.05)
        except Exception:
# English note.
            state["done"] = True
            state["accepted"] = False
            return
