#!/usr/bin/env python3
"""Command-line entry point for the staged XRS processing pipeline.

Usage::

    python run_xrs.py config.yaml
    python run_xrs.py run config.yaml --stage sum
    python run_xrs.py run config.yaml --from xrs
    python run_xrs.py run config.yaml --no-ui
    python run_xrs.py run config.yaml --force
    python run_xrs.py check config.yaml
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
# Argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_xrs.py",
        description="YAML-driven staged XRS spectrum processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one or more stages")
    run.add_argument("config", nargs="?", default="config.yaml", help="YAML configuration")
    group = run.add_mutually_exclusive_group()
    group.add_argument("--stage", choices=STAGE_ORDER, help="Run only this stage")
    group.add_argument("--from", dest="from_stage", choices=STAGE_ORDER,
                       help="Run this stage and every downstream stage")
    run.add_argument("--no-ui", action="store_true",
                     help="Disable windows and prompts; report missing parameters")
    run.add_argument("--force", action="store_true",
                     help="Ignore fingerprints and recompute selected stages")
    run.add_argument("--overwrite", action="store_true",
                     help="Allow the save stage to overwrite output files")

    check = subparsers.add_parser("check", help="Check configuration and stage status")
    check.add_argument("config", nargs="?", default="config.yaml")

    pick = subparsers.add_parser("pick-roi", help="Interactively select and segment ROIs")
    pick.add_argument("config", nargs="?", default="config.yaml")
    pick.add_argument("--detector", choices=DETECTORS,
                      help="Process only this detector; default: both")

    propose = subparsers.add_parser(
        "propose-roi", help="Generate candidate HDF5 ROIs using peak detection"
    )
    propose.add_argument("config", nargs="?", default="config.yaml")
    propose.add_argument("--detector", choices=DETECTORS,
                         help="Process only this detector; default: both")
    propose.add_argument("--write", action="store_true",
                         help="Replace the configured HDF5 instead of writing a candidate")

    return parser


def normalise_argv(argv: list[str]) -> list[str]:
    """Allow omitting ``run`` before a configuration path."""
    if not argv:
        return ["run"]
    if argv[0] not in KNOWN_COMMANDS:
        return ["run"] + list(argv)
    return list(argv)


# --------------------------------------------------------------------------
# Commands
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
        # Report every missing parameter before starting a headless run.
        blocked = []
        for stage in stages:
            problems = cfg.problems(stage)
            if problems:
                blocked.append((stage, problems))
        if blocked:
            log("The following parameters are required for --no-ui:")
            for stage, problems in blocked:
                log(f"  [{stage}]")
                for item in problems:
                    log(f"    - {item}")
            return 1

    for stage in stages:
        run_stage(cfg, stage, ui, log=log, force=args.force, overwrite=args.overwrite)
    log(f"Completed: {', '.join(stages)}")
    return 0


def command_check(args) -> int:
    cfg = Config.load(args.config)
    log(f"Configuration: {cfg.path}")
    log("")

    root_ok = not is_blank(cfg.get("data.root"))
    store = StageStore(cfg.state_dir) if root_ok else None

    exit_code = 0
    for stage in STAGE_ORDER:
        problems = cfg.problems(stage)
        if problems:
            status = f"missing {len(problems)} parameter(s)"
            exit_code = 1
        elif store is None:
            status = "ready (cache unavailable because data.root is empty)"
        elif store.is_current(stage, cfg.fingerprint(stage)):
            status = "ready; cache is current"
        elif store.stage_info(stage):
            status = "parameters changed; rerun required"
        else:
            status = "ready; not yet run"
        log(f"[{stage}] {STAGE_TITLE[stage]}")
        log(f"    Status: {status}")
        for item in problems:
            log(f"    - {item}")

    log("")
    only_pixi, only_env = environment_parity(cfg.path.parent)
    if only_pixi or only_env:
        log("Environment manifests differ (pixi.toml vs environment.yml):")
        if only_pixi:
            log(f"    Only in pixi.toml: {only_pixi}")
        if only_env:
            log(f"    Only in environment.yml: {only_env}")
    else:
        log("Environment manifests match.")
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
# Entry point
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
        log(f"\n[CONFIG ERROR] {exc}")
        return 1
    except UiRequired as exc:
        log(f"\n[INTERACTION REQUIRED] {exc}")
        return 2
    except UiCancelled as exc:
        log(f"\n[CANCELLED] {exc}")
        return 130
    except KeyboardInterrupt:
        log("\n[INTERRUPTED] Ctrl-C")
        return 130
    except FileNotFoundError as exc:
        log(f"\n[FILE NOT FOUND] {exc}")
        return 1
    log(f"Unhandled command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
