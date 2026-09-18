#!/usr/bin/env python3
"""XRS 光谱处理命令行入口。

刻意只依赖标准库 + 项目模块，**不需要 pixi**：在束线站的 Linux 桌面上
用 conda 环境里的 ``python run_xrs.py config.yaml`` 即可。

用法::

    python run_xrs.py config.yaml                     # 从第一个未完成的阶段跑起
    python run_xrs.py run config.yaml --stage sum     # 只跑某个阶段
    python run_xrs.py run config.yaml --from xrs      # 从某个阶段往后跑
    python run_xrs.py run config.yaml --no-ui         # 不交互，只用 YAML 里已填好的值
    python run_xrs.py run config.yaml --force         # 忽略指纹，强制重算
    python run_xrs.py check config.yaml               # 只体检，不动数据
    python run_xrs.py pick-roi config.yaml --detector lambda
    python run_xrs.py propose-roi config.yaml --detector lambda [--write]
"""

from __future__ import annotations

import argparse
import sys

from xrs_config import (
    STAGE_ORDER,
    STAGE_TITLE,
    Config,
    ConfigError,
    environment_parity,
    is_blank,
)
from xrs_pipeline import StageStore, run_pick_roi, run_propose_roi, run_stage
from xrs_ui import UI, UiCancelled, UiRequired

KNOWN_COMMANDS = ("run", "check", "pick-roi", "propose-roi")
DETECTORS = ("lambda", "minipix")


def log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------
# 参数解析
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_xrs.py",
        description="XRS 光谱处理：YAML 驱动、可分阶段反复运行。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="运行一个或多个阶段")
    run.add_argument("config", nargs="?", default="config.yaml", help="YAML 配置文件")
    group = run.add_mutually_exclusive_group()
    group.add_argument("--stage", choices=STAGE_ORDER, help="只运行这一个阶段")
    group.add_argument("--from", dest="from_stage", choices=STAGE_ORDER,
                       help="从这个阶段开始往后运行")
    run.add_argument("--no-ui", action="store_true",
                     help="不做任何交互（不弹窗、不提问），缺参数直接报错")
    run.add_argument("--force", action="store_true",
                     help="忽略参数指纹，强制重算所选阶段")
    run.add_argument("--overwrite", action="store_true",
                     help="允许 save 阶段覆盖已存在的输出文件")

    check = subparsers.add_parser("check", help="检查参数完整性与各阶段状态")
    check.add_argument("config", nargs="?", default="config.yaml")

    pick = subparsers.add_parser("pick-roi", help="交互式点选 ROI 中心点")
    pick.add_argument("config", nargs="?", default="config.yaml")
    pick.add_argument("--detector", choices=DETECTORS,
                      help="只处理该探测器；默认两个都做")

    propose = subparsers.add_parser(
        "propose-roi", help="用峰值检测生成候选中心点（默认只写 *.proposed.txt）"
    )
    propose.add_argument("config", nargs="?", default="config.yaml")
    propose.add_argument("--detector", choices=DETECTORS,
                         help="只处理该探测器；默认两个都做")
    propose.add_argument("--write", action="store_true",
                         help="直接覆盖配置指定的中心点文件（默认只写候选文件）")

    return parser


def normalise_argv(argv: list[str]) -> list[str]:
    """允许省略 ``run``：``run_xrs.py config.yaml`` 等价于 ``run config.yaml``。"""
    if not argv:
        return ["run"]
    if argv[0] not in KNOWN_COMMANDS:
        return ["run"] + list(argv)
    return list(argv)


# --------------------------------------------------------------------------
# 子命令
# --------------------------------------------------------------------------


def select_stages(args) -> list[str]:
    if args.stage:
        return [args.stage]
    if args.from_stage:
        return list(STAGE_ORDER[STAGE_ORDER.index(args.from_stage):])
    return list(STAGE_ORDER)


def command_run(args) -> int:
    cfg = Config.load(args.config)
    ui = UI(interactive=not args.no_ui, dpi=int(cfg.get("ui.dpi", 150) or 150))
    stages = select_stages(args)

    if args.no_ui:
        # --no-ui 时先在门口把缺失参数报全，不要跑一半才炸
        blocked = []
        for stage in stages:
            problems = cfg.problems(stage)
            if problems:
                blocked.append((stage, problems))
        if blocked:
            log("--no-ui 模式下以下参数必须先在 config.yaml 里填好：")
            for stage, problems in blocked:
                log(f"  [{stage}]")
                for item in problems:
                    log(f"    - {item}")
            return 1

    for stage in stages:
        run_stage(cfg, stage, ui, log=log, force=args.force, overwrite=args.overwrite)
    log(f"完成：{', '.join(stages)}")
    return 0


def command_check(args) -> int:
    cfg = Config.load(args.config)
    log(f"配置文件：{cfg.path}")
    log("")

    root_ok = not is_blank(cfg.get("data.root"))
    store = StageStore(cfg.state_dir) if root_ok else None

    exit_code = 0
    for stage in STAGE_ORDER:
        problems = cfg.problems(stage)
        if problems:
            status = f"缺 {len(problems)} 项参数"
            exit_code = 1
        elif store is None:
            status = "参数齐全（data.root 未填，无法判断缓存）"
        elif store.is_current(stage, cfg.fingerprint(stage)):
            status = "参数齐全，缓存有效（会跳过）"
        elif store.stage_info(stage):
            status = "参数已变更，需要重跑"
        else:
            status = "参数齐全，尚未运行"
        log(f"[{stage}] {STAGE_TITLE[stage]}")
        log(f"    状态：{status}")
        for item in problems:
            log(f"    - {item}")

    log("")
    only_pixi, only_env = environment_parity(cfg.path.parent)
    if only_pixi or only_env:
        log("环境清单不一致（pixi.toml vs environment.yml）：")
        if only_pixi:
            log(f"    只在 pixi.toml 里：{only_pixi}")
        if only_env:
            log(f"    只在 environment.yml 里：{only_env}")
    else:
        log("环境清单一致（pixi.toml 与 environment.yml）。")
    return exit_code


def command_rebuild_centers(args) -> int:
    cfg = Config.load(args.config)
    ui = UI(interactive=True, dpi=int(cfg.get("ui.dpi", 150) or 150))
    targets = [args.detector] if args.detector else list(DETECTORS)
    for detector in targets:
        log(f"===== {detector} =====")
        if args.command == "pick-roi":
            run_pick_roi(cfg, detector, ui, write=True, log=log)
        else:
            run_propose_roi(cfg, detector, write=args.write, log=log)
    return 0


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = normalise_argv(sys.argv[1:] if argv is None else list(argv))
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return command_run(args)
        if args.command == "check":
            return command_check(args)
        if args.command == "pick-roi":
            return command_rebuild_centers(args)
        if args.command == "propose-roi":
            return command_rebuild_centers(args)
    except ConfigError as exc:
        log(f"\n[配置错误] {exc}")
        return 1
    except UiRequired as exc:
        log(f"\n[需要交互] {exc}")
        return 2
    except UiCancelled as exc:
        log(f"\n[已取消] {exc}")
        return 130
    except KeyboardInterrupt:
        log("\n[中断] 用户按了 Ctrl-C")
        return 130
    except FileNotFoundError as exc:
        log(f"\n[找不到文件] {exc}")
        return 1
    log(f"未处理的命令：{args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
