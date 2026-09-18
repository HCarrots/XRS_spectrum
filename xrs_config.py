"""xrs_config implementation."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

# --------------------------------------------------------------------------
# English note.
# --------------------------------------------------------------------------

STAGE_ORDER = ("elastic", "xrs", "q", "sum", "save")

STAGE_TITLE = {
    "elastic": "elastic scans -> ROI -> elastic fits",
    "xrs": "XRS scans -> I0 processing -> ROI spectra",
    "q": "momentum-transfer Q calculation",
    "sum": "energy interpolation and combination",
    "save": "write result files",
}

# English note.
STAGE_KEYS = {
    "elastic": ("elastic",),
    "xrs": ("xrs",),
    "q": ("q",),
    "sum": ("sum",),
    "save": ("output",),
}

# English note.
COMMON_KEYS = ("data",)

DETECTORS = ("lambda", "minipix")
_MODULE_ORDER = ("VB", "VU", "VD", "HB", "HL", "HR")
_ROI_MODE_CHOICES = ("regular", "auto")
_OUTPUT_MODE_CHOICES = ("combined", "per_module", "per_crystal")
_ELASTIC_ENERGY_SOURCES = ("as_before", "nominal", "fitted")
_ON_ROI_FAILURE = ("warn", "error")


class ConfigError(Exception):
    """Implementation notes for ConfigError."""


def is_blank(value) -> bool:
    """Implementation notes for is_blank."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def _plain(value):
    """Implementation notes for _plain."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


class Config:
    """Implementation notes for Config."""

    def __init__(self, path: Path, data: CommentedMap, yaml: YAML):
        self.path = Path(path)
        self._data = data
        self._yaml = yaml
        self._original = self._dump()

# English note.

    @classmethod
    def load(cls, path) -> "Config":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Configuration file does not exist: {path}")
        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)
# English note.
        yaml.width = 100000
        with path.open(encoding="utf-8") as handle:
            data = yaml.load(handle)
        if data is None:
            data = CommentedMap()
        if not isinstance(data, CommentedMap):
            raise ConfigError(f"The top level of {path} must be a mapping")
        return cls(path, data, yaml)

# English note.

    def get(self, dotted: str, default=None):
        """Implementation notes for get."""
        node = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value) -> None:
        """Implementation notes for set."""
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, MutableMapping):
                child = CommentedMap()
                node[part] = child
            node = child

        key = parts[-1]
        if isinstance(value, (list, tuple)) and not isinstance(value, CommentedSeq):
# English note.
# English note.
# English note.
            sequence = CommentedSeq(value)
            sequence.fa.set_flow_style()
            value = sequence
        node[key] = value

    def __contains__(self, dotted: str) -> bool:
        return self.get(dotted, _MISSING) is not _MISSING

# English note.

    @property
    def data_root(self) -> Path:
        root = self.get("data.root")
        if is_blank(root):
            raise ConfigError("data.root is empty")
        return Path(os.path.expandvars(str(root))).expanduser()

    @property
    def raw_dir(self) -> Path:
        return self.data_root / str(self.get("data.raw_subdir", "raw"))

    @property
    def processed_dir(self) -> Path:
        return self.data_root / str(self.get("data.processed_subdir", "processed"))

    @property
    def state_dir(self) -> Path:
        """Implementation notes for state_dir."""
        return self.processed_dir / ".xrs_state"

    @property
    def figures_dir(self) -> Path:
        return self.state_dir / "figures"

    def resolve(self, value) -> Path:
        """Implementation notes for resolve."""
        path = Path(os.path.expandvars(str(value))).expanduser()
        if not path.is_absolute():
            path = self.path.parent / path
        return path

# English note.

    def _dump(self) -> str:
        from io import StringIO

        buffer = StringIO()
        self._yaml.dump(self._data, buffer)
        return buffer.getvalue()

    def save(self) -> bool:
        """Implementation notes for save."""
        text = self._dump()
        if text == self._original:
            return False
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, self.path)
        self._original = text
        return True

# English note.

    def problems(self, stage: str) -> list[str]:
        """Implementation notes for problems."""
        if stage not in STAGE_ORDER:
            raise ConfigError(f"Unknown stage: {stage!r}")
        checker = getattr(self, f"_check_{stage}")
        return checker()

    def _check_data_root(self) -> list[str]:
        if is_blank(self.get("data.root")):
            return ["data.root is empty (example: /hepsdatafs/ID33/202504/Data/GID33-250416-01/)"]
        return []

    def _check_scan_ids(self, prefix: str) -> list[str]:
        value = self.get(f"{prefix}.scan_ids")
        if is_blank(value):
            return [f"{prefix}.scan_ids is empty (at least one scan ID is required)"]
        if not isinstance(value, (list, tuple)):
            return [f"{prefix}.scan_ids must be a list, for example [14522]"]
        try:
            [int(item) for item in value]
        except (TypeError, ValueError):
            return [f"{prefix}.scan_ids must contain integers, got {value!r}"]
        return []

    def _check_elastic(self) -> list[str]:
        issues = self._check_data_root()
        issues += self._check_scan_ids("elastic")

        if not is_blank(self.get("elastic.divide_by_i0")):
            if is_blank(self.get("elastic.i0_pv")):
                issues.append("elastic.i0_pv is required when divide_by_i0 is true")

        mode = self.get("elastic.roi_mode")
        if is_blank(mode):
            issues.append("elastic.roi_mode is empty ('regular' or 'auto')")
        elif mode not in _ROI_MODE_CHOICES:
            issues.append(f"elastic.roi_mode = {mode!r} must be one of {_ROI_MODE_CHOICES}")
        elif mode == "regular":
            for detector in DETECTORS:
                if is_blank(self.get(f"elastic.regular_files.{detector}")):
                    issues.append(
                        f"elastic.regular_files.{detector} is required in regular mode"
                    )
        elif mode == "auto":
            for detector in DETECTORS:
                if (
                    is_blank(self.get(f"elastic.auto_files.{detector}"))
                    and is_blank(self.get(f"elastic.auto_centers.{detector}"))
                ):
                    issues.append(
                        f"elastic.auto_files.{detector} is empty "
                        f"(auto mode requires an HDF5 ROI file or a legacy center file)"
                    )

        filter_value = self.get("elastic.filter_value")
        if not is_blank(self.get("elastic.use_filter")) and filter_value is not None:
            try:
                if not 0.0 <= float(filter_value) <= 1.0:
                    issues.append(f"elastic.filter_value = {filter_value} must be between 0 and 1")
            except (TypeError, ValueError):
                issues.append(f"elastic.filter_value = {filter_value!r} is not numeric")

        low, high = self.get("elastic.fit.e_lowlim"), self.get("elastic.fit.e_highlim")
        try:
            if float(low) >= float(high):
                issues.append(
                    f"elastic.fit.e_lowlim ({low}) must be below e_highlim ({high})"
                )
        except (TypeError, ValueError):
            issues.append("elastic.fit.e_lowlim and e_highlim must be numeric")

        on_failure = self.get("elastic.auto_params.on_roi_failure", "warn")
        if on_failure not in _ON_ROI_FAILURE:
            issues.append(
                f"elastic.auto_params.on_roi_failure = {on_failure!r} must be one of {_ON_ROI_FAILURE}"
            )
        return issues

    def _check_xrs(self) -> list[str]:
        issues = self._check_data_root()
        issues += self._check_scan_ids("xrs")
        if not is_blank(self.get("xrs.divide_by_i0")):
            if is_blank(self.get("xrs.i0_pv")):
                issues.append("xrs.i0_pv is required when divide_by_i0 is true")
        excluded = self.get("xrs.exclude_scans", [])
        if not is_blank(excluded):
            if not isinstance(excluded, (list, tuple)):
                issues.append(f"xrs.exclude_scans must be a list, got {excluded!r}")
            else:
                try:
                    [int(item) for item in excluded]
                except (TypeError, ValueError):
                    issues.append(f"xrs.exclude_scans must contain integers, got {excluded!r}")
        return issues

    def _check_q(self) -> list[str]:
        issues = self._check_data_root()
        angles = self.get("q.module_angles_deg")
        if is_blank(angles):
            issues.append(
                "q.module_angles_deg requires six angles in order "
                "['VB','VU','VD','HB','HL','HR']"
            )
        elif not isinstance(angles, (list, tuple)) or len(angles) != len(_MODULE_ORDER):
            issues.append(
                f"q.module_angles_deg requires {len(_MODULE_ORDER)} angles in order "
                f"{list(_MODULE_ORDER)}, got {angles!r}"
            )
        source = self.get("q.elastic_energy_source", "as_before")
        if source not in _ELASTIC_ENERGY_SOURCES:
            issues.append(
                f"q.elastic_energy_source = {source!r} must be one of {_ELASTIC_ENERGY_SOURCES}"
            )
        return issues

    def _check_sum(self) -> list[str]:
        issues = self._check_data_root()
        modules = self.get("sum.modules")
        if is_blank(modules):
            issues.append("sum.modules is empty (example: ['VD'])")
        elif not isinstance(modules, (list, tuple)):
            issues.append(f"sum.modules must be a list, got {modules!r}")
        else:
            unknown = [m for m in modules if m not in _MODULE_ORDER]
            if unknown:
                issues.append(
                    f"sum.modules contains unknown modules {unknown}; valid: {list(_MODULE_ORDER)}"
                )

        q_range = self.get("sum.q_range")
        if not isinstance(q_range, (list, tuple)) or len(q_range) != 2:
            issues.append(f"sum.q_range must be [lower, upper], got {q_range!r}")
        else:
            try:
                if float(q_range[0]) > float(q_range[1]):
                    issues.append(f"sum.q_range lower bound {q_range[0]} exceeds {q_range[1]}")
            except (TypeError, ValueError):
                issues.append(f"sum.q_range values must be numeric, got {q_range!r}")

        step = self.get("sum.energy_step_ev")
        try:
            if float(step) <= 0:
                issues.append(f"sum.energy_step_ev = {step} must be positive")
        except (TypeError, ValueError):
            issues.append(f"sum.energy_step_ev = {step!r} is not numeric")

        excluded = self.get("sum.exclude_rois", [])
        if not is_blank(excluded) and not isinstance(excluded, (list, tuple)):
            issues.append(f"sum.exclude_rois must be a list, got {excluded!r}")
        return issues

    def _check_save(self) -> list[str]:
        issues = self._check_data_root()
        if is_blank(self.get("output.filename")):
            issues.append("output.filename is empty")
        mode = self.get("output.mode", "combined")
        if mode not in _OUTPUT_MODE_CHOICES:
            issues.append(
                f"output.mode = {mode!r} must be one of {_OUTPUT_MODE_CHOICES}"
            )
        return issues

# English note.

    def fingerprint(self, stage: str) -> str:
        """Implementation notes for fingerprint."""
        if stage not in STAGE_ORDER:
            raise ConfigError(f"Unknown stage: {stage!r}")
        upto = STAGE_ORDER[: STAGE_ORDER.index(stage) + 1]
        keys = list(COMMON_KEYS)
        for name in upto:
            keys.extend(STAGE_KEYS[name])

        payload = {}
        for key in sorted(set(keys)):
            payload[key] = _plain(self.get(key))
# English note.
        for dotted in _ROI_INPUT_KEYS:
            value = self.get(dotted)
            if not is_blank(value):
                payload[dotted] = _file_digest(self.resolve(value))

        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

# English note.

    def as_dict(self) -> dict:
        return _plain(self._data)


_MISSING = object()

_ROI_INPUT_KEYS = (
    "elastic.regular_files.lambda",
    "elastic.regular_files.minipix",
    "elastic.auto_centers.lambda",
    "elastic.auto_centers.minipix",
    "elastic.auto_files.lambda",
    "elastic.auto_files.minipix",
)


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError as exc:
        return f"<unreadable: {exc.__class__.__name__}>"


# --------------------------------------------------------------------------
# English note.
# --------------------------------------------------------------------------

_PIXI_TOML = "pixi.toml"
_ENVIRONMENT_YML = "environment.yml"


def environment_parity(base: Path | None = None) -> tuple[list[str], list[str]]:
    """Implementation notes for environment_parity."""
    base = Path(base) if base is not None else Path(__file__).resolve().parent
    pixi_path, env_path = base / _PIXI_TOML, base / _ENVIRONMENT_YML
    if not pixi_path.is_file() or not env_path.is_file():
        return [], []

    with pixi_path.open("rb") as handle:
        pixi_names = {str(name).lower() for name in tomllib.load(handle).get("dependencies", {})}

    yaml = YAML(typ="safe")
    with env_path.open(encoding="utf-8") as handle:
        env_spec = yaml.load(handle) or {}
    env_names = set()
    for entry in env_spec.get("dependencies", []):
        if isinstance(entry, str):
            env_names.add(entry.split()[0].split("=")[0].strip().lower())
        elif isinstance(entry, Mapping):
            env_names.add(str(next(iter(entry))).lower())

    return sorted(pixi_names - env_names), sorted(env_names - pixi_names)
