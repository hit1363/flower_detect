# IEEE Paper Drafting Guide

Working title: **Comparative Evaluation of YOLO Detectors for Single-Class Flower Detection on a Self-Collected Dataset**

## Required sections

1. **Title**
2. **Abstract** — write after evaluation; state task, dataset, model/epoch comparison, key measured result and limitation.
3. **Introduction** — application motivation, object detection versus flower classification, project objective and contributions.
4. **Literature Review** — flower/plant datasets; annotation and scene challenges; one-stage detection; YOLOv7, v8, v9, v10, YOLO11 and YOLO26; prior flower detection studies; evaluation metrics; research gap.
5. **Methodology** — collection, one-class definition, annotation rules, quality review, group-aware split, preprocessing, model variants, training settings, software/hardware and evaluation definitions.
6. **Results and Discussion** — 12-row table of the actual epoch-15/30 results; compare accuracy and parameter count; discuss errors using examples from the test set.
7. **Limitations and Future Work** — small single-class dataset, limited collection locations/seasons, possible annotation subjectivity, model-recipe differences, and future multi-species/generalization work.
8. **Conclusion** — summarize measured findings without claiming results not supported by the test set.
9. **References** — IEEE numeric citations; at least 12 sources.

## Results table columns

| YOLO family | Variant | Epoch | Classes | Trainable params | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fill from experiment log |  |  | 1 |  |  |  |  |  |  |

Keep the final paper within the course's six-page IEEE template, including references. Replace all bracketed placeholders in main.tex only after collecting real project details and results.
