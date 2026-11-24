# Annotation Protocol

## Object definition

Use one object class: flower. Draw one tight rectangular box around each clearly identifiable flower head or bloom. Exclude stems and leaves.

## Include and exclude

- Include a bloom when its visible petals or flower structure are sufficient to identify it as a flower.
- Include a partly occluded bloom if its visible portion is still identifiable.
- In dense groups, draw separate boxes around blooms that can be visually separated. If individual blooms cannot be separated reliably, record the image for team review instead of switching between a cluster box and individual boxes.
- Exclude buds that have not opened, unless the team explicitly changes the target definition before annotation begins.
- Do not include an image with no identifiable flower in the positive dataset.

## Quality review

- Use the same annotation tool and rules for all four members.
- Peer-review at least 20% of images, sampled across all four annotators; resolve disagreements and update the rules before continuing.
- Before splitting, check every image-label pair, class ID, box bounds, duplicate image hash and capture-group metadata using scripts/prepare_dataset.py.
- Do not modify the original photographs. Store labels and reviewed copies separately.

## Annotation tool

Use LabelImg, which is free, offline and writes YOLO `.txt` files directly. Install it once per member:

    pip install labelImg
    labelImg

Then, in LabelImg: **Open Dir** the images folder, **Change Save Dir** to the labels folder, and tick the **YOLO** format button on the toolbar (it shows `.txt` when active). In `data/predefined_classes.txt` write the single line `flower` so every box is class 0. Copy `dataset/Annotated/source/metadata.csv` next to the images and fill one row per image before annotation is called finished.

## YOLO label format

Each image has a matching .txt file. Each object is one row:

    0 x_center y_center width height

Coordinates are normalized to the image width and height and must lie between 0 and 1. Class ID 0 maps to flower.

## Collection metadata

Fill one row per annotated source image in metadata.csv. Use capture_group for a burst or set of photos taken from the same spot during the same short session; this prevents near-duplicate leakage across train, validation and test splits.
