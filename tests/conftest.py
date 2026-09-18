"""conftest implementation."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)  # Tests never open a GUI window.

import h5py
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xrs_config import Config  # noqa: E402

TEMPLATE = PROJECT_ROOT / "config.yaml"
WORK_ROOT = Path(__file__).resolve().parent / "_work"

IMAGE_SHAPE = (48, 60)
# English note.
BLOBS = ((16, 14), (41, 14))
REGULAR_BOXES = ((10, 22, 8, 20), (35, 47, 8, 20))  # (x1, x2, y1, y2)

ELASTIC_SCAN_ID = 14522
XRS_SCAN_ID = 14524
ENERGY_PV = "M_DCM_B5Energy_readback"
I0_PV = "D_SiC_I_A"

ELASTIC_CENTER_KEV = 9.68
ELASTIC_HWHM_KEV = 0.0006  # English note.


def _blob_image():
    """Implementation notes for _blob_image."""
    height, width = IMAGE_SHAPE
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros(IMAGE_SHAPE, dtype=float)
    for cx, cy in BLOBS:
        image += 100.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0**2))
    return image


def _lorentzian(energy, center, hwhm, peak, floor):
    return floor + (peak - floor) * hwhm**2 / ((energy - center) ** 2 + hwhm**2)


def build_case(root: Path, *, mode: str, glitch_index: int | None = None) -> Path:
    """Implementation notes for build_case."""
    raw = root / "raw"
    roi_dir = root / "roi"
    raw.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    blob = _blob_image()

# English note.
    elastic_energy = np.linspace(9.66, 9.70, 201)  # English note.
    elastic_profile = _lorentzian(
        elastic_energy, ELASTIC_CENTER_KEV, ELASTIC_HWHM_KEV, peak=10.0, floor=1.0
    )
    elastic_i0 = np.full(elastic_energy.size, 1000.0)
    elastic_lambda = blob[None, :, :] * elastic_profile[:, None, None] * elastic_i0[:, None, None]
    elastic_minipix = 0.5 * elastic_lambda

    _write_nexus(
        raw / f"{ELASTIC_SCAN_ID}_elastic.nxs",
        elastic_energy,
        elastic_lambda,
        elastic_minipix,
        elastic_i0,
    )

# English note.
    xrs_energy = np.linspace(9.90, 10.10, 101)
    xrs_profile = 1.0 + 0.5 * np.tanh((xrs_energy - 10.0) / 0.02)
    xrs_i0 = np.full(xrs_energy.size, 2000.0)
    if glitch_index is not None:
        xrs_i0[glitch_index] = xrs_i0[glitch_index - 1] * 0.1
    xrs_lambda = blob[None, :, :] * xrs_profile[:, None, None] * xrs_i0[:, None, None]
    xrs_minipix = 0.5 * xrs_lambda

    _write_nexus(
        raw / f"{XRS_SCAN_ID}_xrs.nxs",
        xrs_energy,
        xrs_lambda,
        xrs_minipix,
        xrs_i0,
    )

# English note.
    regular_lambda = roi_dir / "regular_lambda.txt"
    regular_minipix = roi_dir / "regular_minipix.txt"
    header = "roi_label\tx1\tx2\ty1\ty2\tx_shift\ty_shift\tx_expand\ty_expand\n"
    rows = "".join(
        f"HB-{'A1' if index == 0 else 'A2'}\t{x1}\t{x2}\t{y1}\t{y2}\t0\t0\t0\t0\n"
        for index, (x1, x2, y1, y2) in enumerate(REGULAR_BOXES)
    )
    regular_lambda.write_text(header + rows, encoding="utf-8")
    regular_minipix.write_text(header + rows, encoding="utf-8")

# English note.
    (roi_dir / "auto_lambda.txt").write_text(
        "roi_label\tx\ty\nVB-A1\t16\t14\nVB-A2\t41\t14\n", encoding="utf-8"
    )
    (roi_dir / "auto_minipix.txt").write_text(
        "roi_label\tx\ty\nHB-A1\t16\t14\nHB-A2\t41\t14\n", encoding="utf-8"
    )

# English note.
    config_path = root / "config.yaml"
    shutil.copy(TEMPLATE, config_path)
    cfg = Config.load(config_path)
    cfg.set("data.root", str(root))
    cfg.set("elastic.scan_ids", [ELASTIC_SCAN_ID])
    cfg.set("elastic.roi_mode", mode)
    cfg.set("elastic.regular_files.lambda", "roi/regular_lambda.txt")
    cfg.set("elastic.regular_files.minipix", "roi/regular_minipix.txt")
    cfg.set("elastic.auto_centers.lambda", "roi/auto_lambda.txt")
    cfg.set("elastic.auto_centers.minipix", "roi/auto_minipix.txt")
    cfg.set("xrs.scan_ids", [XRS_SCAN_ID])
    cfg.set("q.module_angles_deg", [145.6, 79.03, 25, 118.8, 58.85, 67.862])
# English note.
# English note.
    cfg.set("sum.modules", ["VB", "HB"] if mode == "auto" else ["HB"])
    cfg.set("sum.q_range", [0, 12])
    cfg.set("sum.energy_step_ev", 0.2)
    cfg.set("output.filename", "synthetic_run")
    cfg.set("ui.dpi", 80)
    cfg.save()
    return config_path


def _write_nexus(path: Path, energy, lambda_stack, minipix_stack, i0):
    with h5py.File(path, "w") as handle:
        instrument = handle.create_group("entry/instrument")
        instrument.create_dataset(ENERGY_PV, data=energy)
# English note.
        instrument.create_dataset("M_DCM_B5Link_Energy_readback", data=energy + 1000.0)
        data = handle.create_group("entry/data")
        data.create_dataset("D_LAMBDA", data=lambda_stack, compression="gzip")
        data.create_dataset("D_MINIPIX", data=minipix_stack, compression="gzip")
        data.create_dataset(I0_PV, data=i0)


@pytest.fixture
def work_dir(request):
    """Implementation notes for work_dir."""
    safe = request.node.name.replace("[", "_").replace("]", "_").replace("/", "_")
    path = WORK_ROOT / safe
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def regular_case(work_dir):
    return build_case(work_dir / "case", mode="regular")


@pytest.fixture
def auto_case(work_dir):
    return build_case(work_dir / "case", mode="auto")


@pytest.fixture
def glitch_case(work_dir):
    return build_case(work_dir / "case", mode="regular", glitch_index=50)


@pytest.fixture(autouse=True)
def _quiet_env(monkeypatch):
    """Implementation notes for _quiet_env."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    yield
