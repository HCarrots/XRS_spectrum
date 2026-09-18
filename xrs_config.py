"""YAML 配置的读取、回写、校验与阶段指纹。

这是"参数账本"的入口：notebook 里散落在各 cell 顶部的变量，在这里统一
变成 ``config.yaml`` 里的键。回写走 ruamel.yaml 的 round-trip 模式，
保证用户手写的中文注释和键顺序不会被程序抹掉。

设计要点
--------
* 参数可以为空。``problems(stage)`` 只报告"这个阶段还缺什么"，
  由 :mod:`xrs_pipeline` 决定是弹界面问你，还是直接报错。
* ``fingerprint(stage)`` 覆盖该阶段及其全部上游阶段的参数，
  这样改了 ``elastic.filter_value`` 之后 ``xrs`` 之后的阶段都会失效重算，
  而只改 ``sum.q_range`` 时前面的阶段会被跳过。
"""

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
# 阶段定义
# --------------------------------------------------------------------------

STAGE_ORDER = ("elastic", "xrs", "q", "sum", "save")

STAGE_TITLE = {
    "elastic": "弹性峰扫描 → ROI → 弹性峰拟合",
    "xrs": "XRS 扫描 → I0 处理 → 各 ROI 谱",
    "q": "动量转移 Q 计算",
    "sum": "能量内插与叠加",
    "save": "写出结果文件",
}

# 每个阶段自己拥有的 YAML 顶层键。上游阶段变化会让下游失效。
STAGE_KEYS = {
    "elastic": ("elastic",),
    "xrs": ("xrs",),
    "q": ("q",),
    "sum": ("sum",),
    "save": ("output",),
}

#: 所有阶段都依赖的公共键。
COMMON_KEYS = ("data",)

DETECTORS = ("lambda", "minipix")
_MODULE_ORDER = ("VB", "VU", "VD", "HB", "HL", "HR")
_ROI_MODE_CHOICES = ("regular", "auto")
_OUTPUT_MODE_CHOICES = ("combined", "per_module", "per_crystal")
_ELASTIC_ENERGY_SOURCES = ("as_before", "nominal", "fitted")
_ON_ROI_FAILURE = ("warn", "error")


class ConfigError(Exception):
    """配置缺失或取值非法。"""


def is_blank(value) -> bool:
    """``None`` / 空串 / 空列表 / 空字典 都算"还没填"。"""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def _plain(value):
    """把 ruamel 的 CommentedMap/CommentedSeq 递归还原成普通容器。"""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


class Config:
    """一个 ``config.yaml`` 的 round-trip 视图。"""

    def __init__(self, path: Path, data: CommentedMap, yaml: YAML):
        self.path = Path(path)
        self._data = data
        self._yaml = yaml
        self._original = self._dump()

    # -- 构造 ---------------------------------------------------------------

    @classmethod
    def load(cls, path) -> "Config":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"配置文件不存在：{path}")
        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)
        # 别把用户写的长注释/长列表折行
        yaml.width = 100000
        with path.open(encoding="utf-8") as handle:
            data = yaml.load(handle)
        if data is None:
            data = CommentedMap()
        if not isinstance(data, CommentedMap):
            raise ConfigError(f"{path} 的顶层必须是一个映射（key: value）")
        return cls(path, data, yaml)

    # -- 存取 ---------------------------------------------------------------

    def get(self, dotted: str, default=None):
        """按 ``"elastic.fit.e_lowlim"`` 这样的点分路径取值。"""
        node = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value) -> None:
        """按点分路径赋值；中间层不存在时补一个空映射。

        已存在的键直接原地替换，因此该键上挂的注释会被 ruamel 保留。
        列表统一写成 flow 风格（``[1, 2]``），与模板保持一致——
        否则 ruamel 会把 ``scan_ids: []`` 改写成占好几行的块状序列。
        """
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
            # 一律用 flow 风格。模板里所有列表都是 [a, b] 写法，
            # 而空序列的 flow_style() 返回 None（不是 False），
            # 靠它判断并不可靠，所以直接指定。
            sequence = CommentedSeq(value)
            sequence.fa.set_flow_style()
            value = sequence
        node[key] = value

    def __contains__(self, dotted: str) -> bool:
        return self.get(dotted, _MISSING) is not _MISSING

    # -- 路径 ---------------------------------------------------------------

    @property
    def data_root(self) -> Path:
        root = self.get("data.root")
        if is_blank(root):
            raise ConfigError("data.root 尚未填写（数据根目录）")
        return Path(os.path.expandvars(str(root))).expanduser()

    @property
    def raw_dir(self) -> Path:
        return self.data_root / str(self.get("data.raw_subdir", "raw"))

    @property
    def processed_dir(self) -> Path:
        return self.data_root / str(self.get("data.processed_subdir", "processed"))

    @property
    def state_dir(self) -> Path:
        """阶段中间结果 + QC 图的落脚点。

        刻意不放在 ``processed/<filename>/`` 里，因为跑到 elastic 阶段时
        ``output.filename`` 很可能还是空的。
        """
        return self.processed_dir / ".xrs_state"

    @property
    def figures_dir(self) -> Path:
        return self.state_dir / "figures"

    def resolve(self, value) -> Path:
        """把配置里的相对路径按"配置文件所在目录"解析。"""
        path = Path(os.path.expandvars(str(value))).expanduser()
        if not path.is_absolute():
            path = self.path.parent / path
        return path

    # -- 保存 ---------------------------------------------------------------

    def _dump(self) -> str:
        from io import StringIO

        buffer = StringIO()
        self._yaml.dump(self._data, buffer)
        return buffer.getvalue()

    def save(self) -> bool:
        """回写 YAML。内容没变就不动文件（免得白白刷新 mtime）。

        返回是否真的写了。
        """
        text = self._dump()
        if text == self._original:
            return False
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, self.path)
        self._original = text
        return True

    # -- 校验 ---------------------------------------------------------------

    def problems(self, stage: str) -> list[str]:
        """列出该阶段还缺哪些参数 / 哪些取值不合法。

        只做结构性检查，不碰磁盘。返回的是给人看的中文句子，
        每条前面带 YAML 路径，方便直接定位。
        """
        if stage not in STAGE_ORDER:
            raise ConfigError(f"未知阶段：{stage!r}")
        checker = getattr(self, f"_check_{stage}")
        return checker()

    def _check_data_root(self) -> list[str]:
        if is_blank(self.get("data.root")):
            return ["data.root 为空（数据根目录，例：/hepsdatafs/ID33/202504/Data/GID33-250416-01/）"]
        return []

    def _check_scan_ids(self, prefix: str) -> list[str]:
        value = self.get(f"{prefix}.scan_ids")
        if is_blank(value):
            return [f"{prefix}.scan_ids 为空（至少填一个扫描编号，例：[14522]）"]
        if not isinstance(value, (list, tuple)):
            return [f"{prefix}.scan_ids 必须是列表，例：[14522]"]
        try:
            [int(item) for item in value]
        except (TypeError, ValueError):
            return [f"{prefix}.scan_ids 只能包含整数，当前为 {value!r}"]
        return []

    def _check_elastic(self) -> list[str]:
        issues = self._check_data_root()
        issues += self._check_scan_ids("elastic")

        if not is_blank(self.get("elastic.divide_by_i0")):
            if is_blank(self.get("elastic.i0_pv")):
                issues.append("elastic.i0_pv 为空（divide_by_i0 为 true 时必须给出归一化 PV）")

        mode = self.get("elastic.roi_mode")
        if is_blank(mode):
            issues.append("elastic.roi_mode 为空（'regular' 或 'auto'）")
        elif mode not in _ROI_MODE_CHOICES:
            issues.append(f"elastic.roi_mode = {mode!r} 非法，只能是 {_ROI_MODE_CHOICES}")
        elif mode == "regular":
            for detector in DETECTORS:
                if is_blank(self.get(f"elastic.regular_files.{detector}")):
                    issues.append(
                        f"elastic.regular_files.{detector} 为空"
                        f"（roi_mode=regular 时必须给出矩形 ROI 文件）"
                    )
        elif mode == "auto":
            for detector in DETECTORS:
                if is_blank(self.get(f"elastic.auto_centers.{detector}")):
                    issues.append(
                        f"elastic.auto_centers.{detector} 为空"
                        f"（roi_mode=auto 时必须给出中心点文件）"
                    )

        filter_value = self.get("elastic.filter_value")
        if not is_blank(self.get("elastic.use_filter")) and filter_value is not None:
            try:
                if not 0.0 <= float(filter_value) <= 1.0:
                    issues.append(f"elastic.filter_value = {filter_value} 必须在 0~1 之间")
            except (TypeError, ValueError):
                issues.append(f"elastic.filter_value = {filter_value!r} 不是数字")

        low, high = self.get("elastic.fit.e_lowlim"), self.get("elastic.fit.e_highlim")
        try:
            if float(low) >= float(high):
                issues.append(
                    f"elastic.fit.e_lowlim ({low}) 必须小于 e_highlim ({high})"
                )
        except (TypeError, ValueError):
            issues.append("elastic.fit.e_lowlim / e_highlim 必须是数字")

        on_failure = self.get("elastic.auto_params.on_roi_failure", "warn")
        if on_failure not in _ON_ROI_FAILURE:
            issues.append(
                f"elastic.auto_params.on_roi_failure = {on_failure!r} 非法，只能是 {_ON_ROI_FAILURE}"
            )
        return issues

    def _check_xrs(self) -> list[str]:
        issues = self._check_data_root()
        issues += self._check_scan_ids("xrs")
        if not is_blank(self.get("xrs.divide_by_i0")):
            if is_blank(self.get("xrs.i0_pv")):
                issues.append("xrs.i0_pv 为空（divide_by_i0 为 true 时必须给出归一化 PV）")
        excluded = self.get("xrs.exclude_scans", [])
        if not is_blank(excluded):
            if not isinstance(excluded, (list, tuple)):
                issues.append(f"xrs.exclude_scans 必须是列表，当前为 {excluded!r}")
            else:
                try:
                    [int(item) for item in excluded]
                except (TypeError, ValueError):
                    issues.append(f"xrs.exclude_scans 只能包含整数，当前为 {excluded!r}")
        return issues

    def _check_q(self) -> list[str]:
        issues = self._check_data_root()
        angles = self.get("q.module_angles_deg")
        if is_blank(angles):
            issues.append(
                "q.module_angles_deg 为空（需要 6 个角度，顺序固定为 "
                "['VB','VU','VD','HB','HL','HR']）"
            )
        elif not isinstance(angles, (list, tuple)) or len(angles) != len(_MODULE_ORDER):
            issues.append(
                f"q.module_angles_deg 需要 {len(_MODULE_ORDER)} 个角度"
                f"（顺序 {list(_MODULE_ORDER)}），当前为 {angles!r}"
            )
        source = self.get("q.elastic_energy_source", "as_before")
        if source not in _ELASTIC_ENERGY_SOURCES:
            issues.append(
                f"q.elastic_energy_source = {source!r} 非法，只能是 {_ELASTIC_ENERGY_SOURCES}"
            )
        return issues

    def _check_sum(self) -> list[str]:
        issues = self._check_data_root()
        modules = self.get("sum.modules")
        if is_blank(modules):
            issues.append("sum.modules 为空（要叠加哪些模组，例：['VD']）")
        elif not isinstance(modules, (list, tuple)):
            issues.append(f"sum.modules 必须是列表，当前为 {modules!r}")
        else:
            unknown = [m for m in modules if m not in _MODULE_ORDER]
            if unknown:
                issues.append(
                    f"sum.modules 含未知模组 {unknown}，只能是 {list(_MODULE_ORDER)}"
                )

        q_range = self.get("sum.q_range")
        if not isinstance(q_range, (list, tuple)) or len(q_range) != 2:
            issues.append(f"sum.q_range 必须是两个数 [下限, 上限]，当前为 {q_range!r}")
        else:
            try:
                if float(q_range[0]) > float(q_range[1]):
                    issues.append(f"sum.q_range 下限 {q_range[0]} 大于上限 {q_range[1]}")
            except (TypeError, ValueError):
                issues.append(f"sum.q_range 必须都是数字，当前为 {q_range!r}")

        step = self.get("sum.energy_step_ev")
        try:
            if float(step) <= 0:
                issues.append(f"sum.energy_step_ev = {step} 必须大于 0")
        except (TypeError, ValueError):
            issues.append(f"sum.energy_step_ev = {step!r} 不是数字")

        excluded = self.get("sum.exclude_rois", [])
        if not is_blank(excluded) and not isinstance(excluded, (list, tuple)):
            issues.append(f"sum.exclude_rois 必须是列表，当前为 {excluded!r}")
        return issues

    def _check_save(self) -> list[str]:
        issues = self._check_data_root()
        if is_blank(self.get("output.filename")):
            issues.append("output.filename 为空（输出目录名）")
        mode = self.get("output.mode", "combined")
        if mode not in _OUTPUT_MODE_CHOICES:
            issues.append(
                f"output.mode = {mode!r} 非法，只能是 {_OUTPUT_MODE_CHOICES}"
            )
        return issues

    # -- 指纹 ---------------------------------------------------------------

    def fingerprint(self, stage: str) -> str:
        """该阶段 + 其全部上游阶段的参数摘要。

        用来判断"虽然产物还在，但参数已经变了，必须重算"。
        """
        if stage not in STAGE_ORDER:
            raise ConfigError(f"未知阶段：{stage!r}")
        upto = STAGE_ORDER[: STAGE_ORDER.index(stage) + 1]
        keys = list(COMMON_KEYS)
        for name in upto:
            keys.extend(STAGE_KEYS[name])

        payload = {}
        for key in sorted(set(keys)):
            payload[key] = _plain(self.get(key))
        # 中心点/矩形 ROI 文件的内容也属于几何输入，改了就重算
        for dotted in _ROI_INPUT_KEYS:
            value = self.get(dotted)
            if not is_blank(value):
                payload[dotted] = _file_digest(self.resolve(value))

        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # -- 杂项 ---------------------------------------------------------------

    def as_dict(self) -> dict:
        return _plain(self._data)


_MISSING = object()

_ROI_INPUT_KEYS = (
    "elastic.regular_files.lambda",
    "elastic.regular_files.minipix",
    "elastic.auto_centers.lambda",
    "elastic.auto_centers.minipix",
)


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError as exc:
        return f"<unreadable: {exc.__class__.__name__}>"


# --------------------------------------------------------------------------
# 两份环境清单的一致性
# --------------------------------------------------------------------------

_PIXI_TOML = "pixi.toml"
_ENVIRONMENT_YML = "environment.yml"


def environment_parity(base: Path | None = None) -> tuple[list[str], list[str]]:
    """比较 pixi.toml 的 [dependencies] 与 environment.yml 的 dependencies。

    返回 ``(只在 pixi 里的包, 只在 environment.yml 里的包)``。
    任何一边读不到就返回空——这不是致命错误，只是提醒。
    """
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
