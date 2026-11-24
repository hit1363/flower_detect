#!/usr/bin/env python3
"""Train YOLOv8/9/10/11/26 and evaluate exact 15- and 30-epoch checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_FIELDS = [
    "model_family", "variant", "epoch", "classes", "total_parameters",
    "trainable_parameters", "precision", "recall", "f1", "map50",
    "map50_95", "image_size", "seed", "batch_size", "framework_version",
    "torch_version", "gpu", "run_directory",
]
MODELS = {
    "yolov8": "yolov8n.pt",
    "yolov9": "yolov9t.pt",
    "yolov10": "yolov10n.pt",
    "yolo11": "yolo11n.pt",
    "yolo26": "yolo26n.pt",
}


def append_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def count_parameters(model) -> tuple[int, int]:
    # Checkpoint loading leaves requires_grad False; training re-enables it, so mirror that before counting.
    model.model.requires_grad_(True)
    parameters = list(model.model.parameters())
    return sum(parameter.numel() for parameter in parameters), sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, default=ROOT / "dataset" / "Annotated" / "split" / "data.yaml")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "supplementary" / "runs")
    parser.add_argument("--results-csv", type=Path, default=ROOT / "supplementary" / "benchmark_results.csv")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0", help="Ultralytics device: '0' for the first GPU, 'cpu' for a smoke run")
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS),
                        help="Train these families only, so a lost cloud session costs one run instead of all five")
    parser.add_argument("--preflight", action="store_true", help="load each official checkpoint and check the GPU/data paths without training")
    args = parser.parse_args()

    if not args.data_yaml.is_file():
        parser.error(f"Dataset YAML not found: {args.data_yaml}")
    try:
        import torch
        import ultralytics
        import yaml
        from ultralytics import YOLO
    except ImportError as error:
        print(f"Install Ultralytics and its supported PyTorch build in the cloud runtime first: {error}", file=sys.stderr)
        return 2

    on_gpu = str(args.device) != "cpu"
    if on_gpu and not torch.cuda.is_available():
        print("A CUDA GPU is required for this cloud benchmark. Select a Colab or Kaggle GPU runtime.", file=sys.stderr)
        return 2

    with args.data_yaml.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if int(data.get("nc", len(data.get("names", [])))) != 1:
        print("This project expects exactly one class.", file=sys.stderr)
        return 2
    root = Path(data.get("path", "."))
    if not root.is_absolute():
        root = (args.data_yaml.parent / root).resolve()
    missing_splits = [split for split in ("train", "val", "test") if not (root / data[split]).is_dir()]
    if missing_splits:
        print(f"Missing dataset split folders: {', '.join(missing_splits)}. Run prepare_dataset.py split first.", file=sys.stderr)
        return 2

    gpu = torch.cuda.get_device_name(0) if on_gpu else "cpu"
    print(f"Ultralytics {ultralytics.__version__}; PyTorch {torch.__version__}; GPU: {gpu}")
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        for family in args.models:
            checkpoint = MODELS[family]
            model = YOLO(checkpoint)
            total, trainable = count_parameters(model)
            print(f"{family}: {checkpoint}, task={model.task}, parameters={total}, trainable={trainable}")
            if model.task != "detect":
                raise RuntimeError(f"{checkpoint} is not a detection checkpoint.")
        print("Preflight complete; no training was started.")
        return 0

    for family in args.models:
        checkpoint = MODELS[family]
        run_dir = args.runs_dir.resolve() / family
        if run_dir.exists():
            print(f"Run folder already exists; choose a new --runs-dir to avoid overwriting: {run_dir}", file=sys.stderr)
            return 2
        run_dir.mkdir(parents=True)
        run_config = {
            "model_family": family,
            "checkpoint": checkpoint,
            "epochs": 30,
            "saved_epochs": [15, 30],
            "image_size": 640,
            "batch_size": args.batch,
            "workers": args.workers,
            "seed": args.seed,
            "device": str(args.device),
            "dataset_yaml": str(args.data_yaml.resolve()),
            "framework_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "gpu": gpu,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        model = YOLO(checkpoint)

        def save_requested_epoch(trainer) -> None:
            trained_epochs = int(trainer.epoch) + 1
            if trained_epochs in (15, 30):
                destination = run_dir / "checkpoints" / f"checkpoint_epoch{trained_epochs}.pt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(Path(trainer.last).read_bytes())
                print(f"Saved epoch {trained_epochs} checkpoint: {destination}")

        model.add_callback("on_model_save", save_requested_epoch)
        model.train(
            data=str(args.data_yaml.resolve()),
            epochs=30,
            imgsz=640,
            batch=args.batch,
            workers=args.workers,
            seed=args.seed,
            device=str(args.device),
            deterministic=True,
            pretrained=True,
            save=True,
            save_period=-1,
            plots=True,
            project=str(args.runs_dir.resolve()),
            name=family,
            # The run folder is pre-created for run_config.json; the guard above
            # ensures this is a new run, so keep Ultralytics on the same path.
            exist_ok=True,
        )

        for epoch in (15, 30):
            checkpoint_path = run_dir / "checkpoints" / f"checkpoint_epoch{epoch}.pt"
            if not checkpoint_path.is_file():
                raise RuntimeError(f"Training finished without the requested checkpoint: {checkpoint_path}")
            evaluated = YOLO(str(checkpoint_path))
            metrics = evaluated.val(
                data=str(args.data_yaml.resolve()),
                split="test",
                imgsz=640,
                batch=args.batch,
                workers=args.workers,
                device=str(args.device),
                conf=0.001,
                iou=0.65,
                plots=False,
                project=str(run_dir / "evaluation"),
                name=f"epoch{epoch}",
                exist_ok=False,
            )
            precision = float(metrics.box.mp)
            recall = float(metrics.box.mr)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            total, trainable = count_parameters(evaluated)
            result = {
                "model_family": family,
                "variant": checkpoint,
                "epoch": epoch,
                "classes": 1,
                "total_parameters": total,
                "trainable_parameters": trainable,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
                "image_size": 640,
                "seed": args.seed,
                "batch_size": args.batch,
                "framework_version": ultralytics.__version__,
                "torch_version": torch.__version__,
                "gpu": gpu,
                "run_directory": str(run_dir),
            }
            append_result(run_dir / "results.csv", result)
            append_result(args.results_csv, result)
            print(f"Recorded {family} epoch {epoch} test metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
