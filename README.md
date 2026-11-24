# Flower Detection Project

Single-class flower object detection, four-member group project: six YOLO families trained at epoch 15 and 30, one IEEE conference paper.

## Current state

- Tooling is finished and tested: dataset validator/splitter, two cloud-GPU notebooks, 12-run results-to-paper builder, IEEE paper scaffold, local LaTeX toolchain.
- `dataset/Original/` and `dataset/Annotated/source/` are still empty. All 12 rows of the results table stay `--` until the group's own ≥400 photos are annotated and the runs happen on Colab/Kaggle.

## Quick start

1. Read [Project plan](PROJECT_PLAN.md) and [annotation protocol](docs/annotation_protocol.md).
2. Put original photographs under `dataset/Original/`.
3. Annotate with LabelImg into `dataset/Annotated/source/images/` + `labels/`, and fill `metadata.csv`.
4. Validate and split:

       python scripts/prepare_dataset.py validate
       python scripts/prepare_dataset.py split

5. Train on a cloud GPU with `notebooks/flower_ultralytics.ipynb` and `notebooks/flower_yolov7.ipynb` (see [cloud training](docs/cloud_training.md)).
6. Build the paper table and run log from the real results, then compile:

       python scripts/build_submission.py --dry-run
       python scripts/build_submission.py
       .tools\tectonic.exe -X compile paper\main.tex

## Layout

- `dataset/Original/` — unchanged photos as collected.
- `dataset/Annotated/source/` — labeled images, YOLO labels, capture metadata.
- `dataset/Annotated/split/` — deterministic train/val/test copies, `data.yaml`, `split_manifest.csv`.
- `scripts/` — `prepare_dataset.py` (validate/split), `train_ultralytics.py`, `train_yolov7.py`, `build_submission.py`.
- `notebooks/` — Colab/Kaggle drivers for both training stacks.
- `supplementary/` — collection log, team sheet, run log, combined `benchmark_results.csv`, `runs/`.
- `paper/` — `main.tex`, `references.bib`, compiled `main.pdf`.
- `docs/` — annotation protocol, cloud training, paper outline.

Every run records exact model name, package versions, device, seed and metrics under `supplementary/runs/<model>/`, and `benchmark_results.csv` combines all 12 evaluations. YOLOv7 uses the upstream WongKinYiu repository; the other five families use Ultralytics 8.4.218.

## Local environment (optional)

`.venv` holds CPU torch + Ultralytics for validation and smoke runs only — it cannot train on a GPU. `.tools\tectonic.exe` compiles the paper offline. Both are gitignored.

## Important

This workspace does not contain original photos, annotations, group names, a course sheet, or a Drive folder. The scripts cannot create authentic collected data or fill a remote sheet without those inputs.
