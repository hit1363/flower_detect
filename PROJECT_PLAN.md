# Project Plan

## Goal and deliverables

Create a self-collected dataset for one-class flower detection, compare YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11 and YOLO26 at the 15- and 30-epoch checkpoints, and submit a six-page IEEE conference paper as a hard copy by 9 December 2026.

## Team of four

| Member | Lead responsibility | Shared work |
|---|---|---|
| Member 1 | Coordination, course sheet, literature source list | Collect and label a share; review paper |
| Member 2 | Image collection and collection log | Collect and label a share; review paper |
| Member 3 | Annotation rules, label QA and dataset split | Collect and label a share; review paper |
| Member 4 | Cloud runs, run log and metrics table | Collect and label a share; review paper |

Replace member placeholders with the actual four names before submitting the course sheet. Aim for roughly 110–115 usable photos per member before duplicate/quality removal.

## Dataset protocol

- Dataset name: Flower Detection Dataset.
- Task: object detection, one class named flower.
- Target: collect about 450 original images and retain at least 400 usable, annotated images.
- Each box covers one clearly identifiable flower head; do not box stems or leaves. Record consistent handling of occluded blooms and dense clusters in the annotation protocol.
- Split the frozen data by capture group into 70% train, 20% validation and 10% test. Keep images from the same burst/location capture group together.
- Resize to 640 pixels for training. Apply model augmentation only on training data.

## Model protocol

- Use the smallest official pretrained detection checkpoint in each family: YOLOv7-tiny, YOLOv8n, YOLOv9t, YOLOv10n, YOLO11n and YOLO26n.
- Train each model once for 30 epochs. Save and evaluate the state at epochs 15 and 30, for 12 table rows.
- Use Colab as the primary runtime, with Kaggle GPU as a fallback. Save run outputs to persistent project storage after each model.
- Keep the data split, 640 image size and evaluation split identical. Record model-specific optimizer/default differences rather than claiming a pure architecture-only comparison.
- Report one class, trainable parameter count, precision, recall, F1, mAP@0.5 and mAP@0.5:0.95. F1 is calculated from the reported precision and recall. Use the validator’s standard confidence handling for each framework and record exact framework versions.

## Work schedule

| Dates | Milestone |
|---|---|
| 22–23 Nov | Confirm team names, create Drive folder, fill known sheet fields, agree capture/annotation rules, check both training stacks |
| 24–27 Nov | Collect about 450 images and remove duplicates as they arrive |
| 26–29 Nov | Annotate, cross-review, and retain at least 400 usable images |
| 30 Nov | Freeze data, make the 70/20/10 split, validate labels, run a short end-to-end check |
| 1–3 Dec | Run six models to epoch 30 and preserve epoch-15/30 checkpoints |
| 4–5 Dec | Evaluate all 12 checkpoints and complete the results table |
| 5–7 Dec | Draft, cite and review the paper in the IEEE six-page template |
| 8 Dec | Verify links, finish the sheet, print and check the hard copy |
| 9 Dec | Submit |

## Acceptance checklist

- Four member names and a working shared Drive link are entered in the course sheet.
- The Drive folder contains unchanged originals, annotated/split data, labels, config, checkpoints and run logs.
- The frozen dataset has at least 400 usable images, one class, no duplicate files and no capture group crossing splits.
- Six model families each have epoch-15 and epoch-30 test metrics.
- The paper has all required sections, is no more than six pages in the course template, and cites at least 12 real sources.

## Current blockers

Everything that can be prepared without the data is prepared: validation/split, the two cloud notebooks, the results-to-paper builder, and the paper scaffold. Three inputs are still human-owned and none can be fabricated:

1. **The photographs.** `dataset/Original/` and `dataset/Annotated/source/` are empty. Collect ~450, annotate at least 400 usable images following `docs/annotation_protocol.md`.
2. **Identifiers.** Member names, student IDs, group number and the shared Drive URL. Fill them in `supplementary/team_sheet_template.csv` once known.
3. **The runs.** All 12 metric rows stay `--` until the notebooks finish on a GPU. `scripts/build_submission.py` refuses to touch the paper table until every row exists.

## Submission package (Drive layout)

Create one shared Drive folder with these sub-folders and fill `dataset_link` / `drive_link` in the sheet with its URL:

- `01_original/` — the untouched photographs, exactly as collected.
- `02_annotated/` — `images/`, `labels/`, `metadata.csv`, plus the generated `split/` folder (`split_manifest.csv`, `data.yaml`, split images/labels).
- `03_supplementary/` — `benchmark_results.csv`, `run_log.csv`, `collection_log.csv`, and `runs/` with every checkpoint and per-run `results.csv`.
- `04_paper/` — `main.pdf`, `main.tex`, `references.bib`.
- `05_sheet/` — the filled team sheet.

## Handover commands

    python scripts/prepare_dataset.py validate     # after annotating
    python scripts/prepare_dataset.py split
    # notebooks/flower_ultralytics.ipynb and notebooks/flower_yolov7.ipynb on Colab/Kaggle GPU
    python scripts/build_submission.py             # results CSV -> paper table + run_log.csv
    python scripts/build_submission.py --dry-run   # preview before it edits main.tex

