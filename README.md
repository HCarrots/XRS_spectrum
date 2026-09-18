# XRS spectrum processing

This project converts the original notebook workflow into a staged, YAML-driven command-line pipeline. Interactive runs use Matplotlib windows for ROI construction and quality control; `--no-ui` runs reproducibly from saved configuration and ROI files.

## Quick start

```bash
conda env create -f environment.yml
conda activate XRS_spectrum
python run_xrs.py config.yaml
```

For local development with Pixi:

```bash
pixi install
pixi run python run_xrs.py config.yaml
```

Useful commands:

```bash
python run_xrs.py run config.yaml --stage elastic
python run_xrs.py run config.yaml --from xrs
python run_xrs.py run config.yaml --no-ui
python run_xrs.py check config.yaml
python run_xrs.py pick-roi config.yaml --detector lambda
python run_xrs.py propose-roi config.yaml --detector minipix
```

## Elastic ROI workflow

The elastic stage now follows this order:

1. Read every ID in `elastic.scan_ids` and normalize each scan independently by I0 when enabled.
2. Concatenate the frames, sort them by energy, and show summed Lambda and Minipix images.
3. Choose `regular` or `auto` after inspecting the detector images.
4. Create new ROI geometry or review and reuse an existing file.
5. Build masks, integrate every ROI, and fit the elastic peaks.
6. Print `badFit` and the full `fitResult` table in the terminal.
7. Optionally enter a crystal name to inspect its fit and verify that all elastic curves are centered at zero energy transfer.
8. Continue to XRS scan loading after fit inspection is complete.

### Regular rectangular ROIs

The rectangle editor follows the canonical detector label order. Drag to draw a rectangle, press `S` to skip a label, `C`, `U`, or Backspace to undo, `R` to redraw the last ROI, `A` to start over, Enter to save the current detector, or Esc to cancel. Enter also accepts an empty detector.

Each detector uses an editable tab-separated file with these columns:

```text
roi_label  x1  x2  y1  y2  x_shift  y_shift  x_expand  y_expand
```

Coordinates use NumPy half-open bounds. Existing threshold filtering, automatic recentering, `roi_size`, and per-ROI shift/expand adjustments remain available.

### Automatic irregular ROIs

The center picker also follows canonical label order and supports skip/undo. Local Otsu segmentation creates irregular connected regions, and overlapping pixels are assigned to the nearest selected center.

Each detector stores final geometry in a versioned HDF5 file containing:

- `label_image`, `roi_labels`, `centers_xy`, and `bounding_boxes`;
- failed labels and failure reasons;
- detector, source scan IDs, image shape, and schema version;
- segmentation parameters and overlap statistics.

Downstream processing derives masks directly from `label_image`; it does not rerun segmentation. Legacy `roi_label x y` center files are accepted as migration inputs and converted to HDF5 on the next elastic run.

`propose-roi` writes `*.proposed.h5` by default. Add `--write` to replace the configured HDF5 target; existing targets are backed up as `.bak`.

## Pipeline stages

| Stage | Purpose |
| --- | --- |
| `elastic` | Load all elastic scans, establish ROI geometry, integrate, and fit elastic peaks |
| `xrs` | Load XRS scans, repair optional I0 glitches, and integrate saved ROI masks |
| `q` | Calculate momentum transfer |
| `sum` | Interpolate and combine selected ROI spectra |
| `save` | Write data, ROI metadata, and processing information |

Intermediate state is stored under `<data.root>/processed/.xrs_state/`. ROI file contents participate in stage fingerprints, so geometry changes invalidate elastic and downstream caches.

## Testing

```bash
pixi run python -m pytest tests -q
```

The tests use synthetic NeXus data and cover regular and automatic ROI modes, HDF5 validation, multiple elastic scans, interactive selectors, cache invalidation, and end-to-end processing.
