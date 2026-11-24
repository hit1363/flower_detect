#!/usr/bin/env python3
"""Train upstream YOLOv7-tiny and evaluate its exact 15/30 epoch checkpoints."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "supplementary" / "benchmark_results.csv"
OLD_SAVE_BRANCH = "elif ((epoch+1) % 25) == 0:"
NEW_SAVE_BRANCH = "elif ((epoch+1) in (15, 30)) or ((epoch+1) % 25) == 0:"
RESULT_FIELDS = [
    "model_family", "variant", "epoch", "classes", "total_parameters",
    "trainable_parameters", "precision", "recall", "f1", "map50",
    "map50_95", "image_size", "seed", "batch_size", "framework_version",
    "torch_version", "gpu", "run_directory",
]


def patch_epoch_saves(train_file: Path) -> None:
    source = train_file.read_text(encoding="utf-8")
    if NEW_SAVE_BRANCH in source:
        return
    if source.count(OLD_SAVE_BRANCH) != 1:
        raise RuntimeError(
            "The YOLOv7 upstream checkpoint-saving code changed. Review train.py before applying the epoch-save patch."
        )
    # Upstream saves epoch 0, every 25 epochs, and the final five; add the required 15th epoch.
    train_file.write_text(source.replace(OLD_SAVE_BRANCH, NEW_SAVE_BRANCH, 1), encoding="utf-8")


def append_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def parameter_counts(checkpoint: Path, repo: Path, torch) -> tuple[int, int]:
    sys.path.insert(0, str(repo))
    load_args = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_args["weights_only"] = False
    state = torch.load(checkpoint, **load_args)
    model = state.get("ema") or state.get("model")
    if model is None:
        raise RuntimeError(f"No model object in checkpoint: {checkpoint}")
    model.requires_grad_(True)  # checkpoint pickles carry requires_grad=False; training re-enables it
    parameters = list(model.parameters())
    return sum(p.numel() for p in parameters), sum(p.numel() for p in parameters if p.requires_grad)


def evaluate(checkpoint: Path, epoch: int, args, repo: Path, torch, gpu: str) -> None:
    sys.path.insert(0, str(repo))
    import test as yolov7_test  # imported from the upstream YOLOv7 checkout

    eval_dir = args.runs_dir.resolve() / "yolov7" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    yolov7_test.opt = SimpleNamespace(
        device=args.device,
        project=str(eval_dir),
        name=f"epoch{epoch}",
        exist_ok=True,
        task="test",
        rect=True,
        cache_images=False,
        image_weights=False,
        quad=False,
        workers=args.workers,
        global_rank=-1,
        local_rank=-1,
        world_size=1,
        single_cls=False,
        max_det=300,
        batch_size=args.batch,
        img_size=640,
        conf_thres=0.001,
        iou_thres=0.65,
        save_json=False,
        augment=False,
        verbose=False,
        save_txt=False,
        save_hybrid=False,
        save_conf=False,
        no_trace=True,
        v5_metric=False,
    )
    metrics, _, _ = yolov7_test.test(
        str(args.data_yaml.resolve()),
        weights=str(checkpoint.resolve()),
        batch_size=args.batch,
        imgsz=640,
        conf_thres=0.001,
        iou_thres=0.65,
        save_json=False,
        single_cls=False,
        plots=False,
        trace=False,
        v5_metric=False,
    )
    precision, recall, map50, map50_95 = map(float, metrics[:4])
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total, trainable = parameter_counts(checkpoint, repo, torch)
    result = {
        "model_family": "YOLOv7",
        "variant": args.weights.name,
        "epoch": epoch,
        "classes": 1,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "image_size": 640,
        "seed": 2,
        "batch_size": args.batch,
        "framework_version": "upstream WongKinYiu/yolov7",
        "torch_version": torch.__version__,
        "gpu": gpu,
        "run_directory": str(args.runs_dir.resolve() / "yolov7"),
    }
    append_result(args.runs_dir.resolve() / "yolov7" / "results.csv", result)
    append_result(args.results_csv, result)
    print(f"Recorded YOLOv7 epoch {epoch} test metrics.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="path to the cloned WongKinYiu/yolov7 repository")
    parser.add_argument("--data-yaml", type=Path, default=ROOT / "dataset" / "Annotated" / "split" / "data.yaml")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "supplementary" / "runs")
    parser.add_argument("--results-csv", type=Path, default=RESULTS)
    parser.add_argument("--weights", type=Path, default=Path("yolov7-tiny.pt"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preflight", action="store_true", help="check the dataset, upstream imports, checkpoint and GPU without training")
    args = parser.parse_args()

    repo = args.repo.resolve()
    train_file = repo / "train.py"
    if not train_file.is_file() or not (repo / "test.py").is_file():
        parser.error(f"--repo must point to the upstream YOLOv7 checkout: {repo}")
    if not args.data_yaml.is_file():
        parser.error(f"Dataset YAML not found: {args.data_yaml}")
    run_dir = args.runs_dir.resolve() / "yolov7"
    if run_dir.exists():
        parser.error(f"Run folder already exists; choose a new --runs-dir to avoid overwriting: {run_dir}")
    try:
        import torch
    except ImportError as error:
        print(f"Install the YOLOv7 dependencies in the cloud runtime first: {error}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("A CUDA GPU is required for this cloud benchmark. Select a Colab or Kaggle GPU runtime.", file=sys.stderr)
        return 2
    gpu = torch.cuda.get_device_name(0)
    weights = args.weights
    if not weights.is_absolute():
        weights = repo / weights
    try:
        import yaml
        with args.data_yaml.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if int(data.get("nc", len(data.get("names", [])))) != 1:
            raise ValueError("YOLOv7 preflight expects exactly one class.")
        missing = [split for split in ("train", "val", "test") if not data.get(split) or not Path(data[split]).exists()]
        if missing:
            raise ValueError(f"Missing absolute dataset paths for: {', '.join(missing)}")
        sys.path.insert(0, str(repo))
        from models.experimental import attempt_load
        model = attempt_load(str(weights), map_location="cuda:0")
        import torch.nn as nn
        with torch.no_grad():
            model(torch.zeros(1, 3, 640, 640, device="cuda:0"))
    except Exception as error:
        print(f"YOLOv7 compatibility check failed: {error}", file=sys.stderr)
        return 2
    if args.preflight:
        print(f"YOLOv7 preflight passed: GPU={gpu}, weights={weights}, data={args.data_yaml.resolve()}. No training was started.")
        return 0

    patch_epoch_saves(train_file)

    command = [
        sys.executable, str(train_file),
        "--workers", str(args.workers),
        "--device", args.device,
        "--batch-size", str(args.batch),
        "--data", str(args.data_yaml.resolve()),
        "--img-size", "640", "640",
        "--weights", str(weights),
        "--hyp", "data/hyp.scratch.custom.yaml",
        "--epochs", "30",
        "--project", str(args.runs_dir.resolve()),
        "--name", "yolov7",
    ]
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    config = {
        "model_family": "YOLOv7",
        "variant": weights.name,
        "epochs": 30,
        "saved_epochs": [15, 30],
        "image_size": 640,
        "batch_size": args.batch,
        "workers": args.workers,
        "seed": 2,
        "dataset_yaml": str(args.data_yaml.resolve()),
        "upstream_revision": revision,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "gpu": gpu,
    }
    (args.runs_dir.resolve() / "yolov7_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    subprocess.run(command, cwd=repo, check=True)

    checkpoints = run_dir / "weights"
    normalized = run_dir / "checkpoints"
    normalized.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for epoch, source_name in ((15, "epoch_014.pt"), (30, "epoch_029.pt")):
        source = checkpoints / source_name
        if not source.is_file():
            raise RuntimeError(f"Training finished without the epoch-{epoch} checkpoint: {source}")
        destination = normalized / f"checkpoint_epoch{epoch}.pt"
        shutil.copy2(source, destination)
        evaluate(destination, epoch, args, repo, torch, gpu)
    print(f"YOLOv7 outputs saved under {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
