# Cloud Training

Two ready-made notebooks in `notebooks/` drive everything below; use them unless you need the manual steps.

- `notebooks/flower_ultralytics.ipynb` — YOLOv8/9/10/11/26, one family per process.
- `notebooks/flower_yolov7.ipynb` — YOLOv7-tiny on a pinned torch 2.5.1 runtime.

Both write run outputs under `supplementary/runs/` and pack `flower_supplementary.zip` back to Drive (Colab) or `/kaggle/working` (Kaggle).

## Google Colab (primary)

1. Copy this project folder and the shared dataset into the team's Google Drive.
2. Open a GPU runtime and mount Drive.
3. Install the current PyTorch build supported by the selected runtime, then install Ultralytics for the five current-family runs: `pip install -q ultralytics==8.4.218 pyyaml`. PyYAML must be explicit — the training script imports `yaml` directly and Ultralytics does not always pull it in.
4. Run the dataset validation/split commands after copying the project into the runtime's persistent storage. The generated data.yaml contains absolute paths for both training stacks.
5. Clone the upstream YOLOv7 repository. Download its official tiny checkpoint into that checkout (or pass its full path with `--weights`):

       wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt -O yolov7-tiny.pt

6. Run each stack's preflight before training:

       python scripts/train_ultralytics.py --preflight
       python scripts/train_yolov7.py --repo /path/to/yolov7 --preflight

7. Run the Ultralytics and YOLOv7 scripts separately. Keep their package versions and run directories separate; use a fresh runtime for the older YOLOv7 dependencies if needed.

   **YOLOv7 pins torch 2.5.1** (`torch==2.5.1`, `torchvision==0.20.1`, `numpy==1.26.4`, `tensorboard`, `thop`). Reason: upstream `train.py:87` and `models/experimental.py:252` call `torch.load()` on pickled checkpoints, which torch 2.6 and newer reject by default (`weights_only=True`). Pinning keeps the clone unpatched apart from the epoch-15/30 save branch.

   If a session dies mid-way, rerun for a single family rather than the whole batch:

       python scripts/train_ultralytics.py --models yolov8
8. Confirm the model checkpoints and results CSV are visible in Drive before ending a session.

## Kaggle GPU (fallback)

Upload or attach the frozen dataset as a Kaggle input. Run outputs under /kaggle/working and download or copy them back to the shared Drive after each model. Kaggle inputs are read-only, so do not direct the training scripts' output directory into the input mount.

## Training entry points

- Ultralytics families: python scripts/train_ultralytics.py --data-yaml dataset/Annotated/split/data.yaml --runs-dir supplementary/runs
- YOLOv7: clone the upstream WongKinYiu/yolov7 repository in the cloud runtime, then run python scripts/train_yolov7.py --repo /path/to/yolov7 --data-yaml /path/to/data.yaml --runs-dir /path/to/persistent/runs

Run one model stack at a time. The YOLOv7 helper makes a narrowly scoped change to its local clone so it saves the exact 15th and 30th epoch checkpoints; it does not change this project’s local sources.
