#!/usr/bin/env python3
"""Validate YOLO flower labels and create a deterministic capture-group split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset" / "Annotated" / "source"
DEFAULT_OUTPUT = ROOT / "dataset" / "Annotated" / "split"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}


def fail(message: str) -> None:
    raise ValueError(message)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(source: Path, images: list[Path]) -> dict[str, dict[str, str]]:
    metadata_path = source / "metadata.csv"
    if not metadata_path.is_file():
        fail(f"Missing metadata file: {metadata_path}")
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        if not {"image", "capture_group"}.issubset(columns):
            fail("metadata.csv must contain image and capture_group columns.")
        rows = {}
        for row in reader:
            name = (row.get("image") or "").strip()
            group = (row.get("capture_group") or "").strip()
            if not name or not group:
                fail("Every metadata row needs a filename and capture_group.")
            if Path(name).name != name:
                fail(f"Metadata image values must be filenames, not paths: {name}")
            if name in rows:
                fail(f"Duplicate metadata row for {name}")
            rows[name] = {key: (value or "").strip() for key, value in row.items() if key}
    image_names = {image.name for image in images}
    missing = sorted(image_names - rows.keys())
    extra = sorted(rows.keys() - image_names)
    if missing or extra:
        details = []
        if missing:
            details.append(f"no metadata for {len(missing)} images (first: {missing[0]})")
        if extra:
            details.append(f"{len(extra)} metadata rows do not match an image (first: {extra[0]})")
        fail("metadata.csv mismatch: " + "; ".join(details))
    return rows


def validate_source(source: Path, minimum_images: int) -> tuple[list[Path], dict[str, dict[str, str]]]:
    images_dir = source / "images"
    labels_dir = source / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        fail(f"Expected image and label folders under {source}")
    images = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(images) < minimum_images:
        fail(f"Found {len(images)} images; at least {minimum_images} are required.")
    stems: set[str] = set()
    hashes: dict[str, str] = {}
    for image in images:
        stem = image.stem.lower()
        if stem in stems:
            fail(f"Image basenames must be unique across formats: {image.name}")
        stems.add(stem)
        digest = file_hash(image)
        if digest in hashes:
            fail(f"Duplicate image content: {hashes[digest]} and {image.name}")
        hashes[digest] = image.name
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            fail(f"Missing label file for {image.name}: {label.name}")
        lines = [line.strip() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not lines:
            fail(f"{image.name} has no flower box; this project expects positive images.")
        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 5:
                fail(f"{label.name}:{line_number} must contain class, x_center, y_center, width, height.")
            if fields[0] != "0":
                fail(f"{label.name}:{line_number} must use class ID 0.")
            try:
                cx, cy, width, height = map(float, fields[1:])
            except ValueError:
                fail(f"{label.name}:{line_number} contains a non-numeric coordinate.")
            if not all(math.isfinite(value) for value in (cx, cy, width, height)):
                fail(f"{label.name}:{line_number} contains a non-finite coordinate.")
            if width <= 0 or height <= 0 or not (0 <= cx <= 1 and 0 <= cy <= 1 and width <= 1 and height <= 1):
                fail(f"{label.name}:{line_number} coordinates must be normalized and box size positive.")
            if cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6 or cy + height / 2 > 1 + 1e-6:
                fail(f"{label.name}:{line_number} box extends outside the image.")
    labels = {path.name for path in labels_dir.glob("*.txt")}
    expected_labels = {f"{image.stem}.txt" for image in images}
    if labels != expected_labels:
        fail("Label folder contains unmatched .txt files or non-image labels.")
    metadata = load_metadata(source, images)
    groups = {row["capture_group"] for row in metadata.values()}
    if len(groups) < 3:
        fail("At least three distinct capture_group values are required for a group-aware split.")
    return images, metadata


def choose_group_splits(images: list[Path], metadata: dict[str, dict[str, str]], seed: int) -> dict[str, str]:
    by_group: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        by_group[metadata[image.name]["capture_group"]].append(image)
    groups = list(by_group.items())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda item: -len(item[1]))
    counts = {split: 0 for split in SPLITS}
    targets = {split: len(images) * RATIOS[split] for split in SPLITS}
    assignment: dict[str, str] = {}

    # Seed each split with a group so the test and validation sets cannot be empty.
    initial = {"train": groups[0], "val": groups[1], "test": groups[-1]}
    seeded = {group for group, _ in initial.values()}
    for split, (group, members) in initial.items():
        assignment[group] = split
        counts[split] += len(members)

    remaining = [item for item in groups if item[0] not in seeded]
    for group, members in remaining:
        candidates = list(SPLITS)
        rng.shuffle(candidates)

        def score(candidate: str) -> float:
            return sum(
                ((counts[split] + (len(members) if split == candidate else 0) - targets[split]) / max(targets[split], 1)) ** 2
                for split in SPLITS
            )

        selected = min(candidates, key=score)
        assignment[group] = selected
        counts[selected] += len(members)
    return {image.name: assignment[metadata[image.name]["capture_group"]] for image in images}


def split_source(source: Path, output: Path, minimum_images: int, seed: int) -> None:
    images, metadata = validate_source(source, minimum_images)
    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if output_resolved == source_resolved or source_resolved in output_resolved.parents:
        fail("Split output must be outside the source annotation folder.")
    if output.exists() and any(output.iterdir()):
        fail(f"Split output already contains files; choose a new folder: {output}")

    assignments = choose_group_splits(images, metadata, seed)
    counts = {split: 0 for split in SPLITS}
    for image in images:
        split = assignments[image.name]
        counts[split] += 1
        for kind, filename in (("images", image.name), ("labels", f"{image.stem}.txt")):
            target = output / kind / split / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2((source / kind / filename), target)
    if any(counts[split] == 0 for split in SPLITS):
        fail("A split is empty. Add capture groups or adjust the collection grouping.")

    with (output / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "capture_group", "collector", "split"])
        writer.writeheader()
        for image in images:
            row = metadata[image.name]
            writer.writerow({
                "image": image.name,
                "capture_group": row["capture_group"],
                "collector": row.get("collector", ""),
                "split": assignments[image.name],
            })
    (output / "data.yaml").write_text(
        "path: " + json.dumps(output.resolve().as_posix()) + "\n"
        + "train: " + json.dumps((output / "images" / "train").resolve().as_posix()) + "\n"
        + "val: " + json.dumps((output / "images" / "val").resolve().as_posix()) + "\n"
        + "test: " + json.dumps((output / "images" / "test").resolve().as_posix()) + "\n"
        + "nc: 1\nnames: [flower]\n",
        encoding="utf-8",
    )
    print(f"Split {len(images)} images: " + ", ".join(f"{key}={counts[key]}" for key in SPLITS))
    print(f"Created {output / 'data.yaml'} with seed {seed}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="check source images, labels, duplicates and metadata")
    validate.add_argument("--source", type=Path, default=SOURCE)
    validate.add_argument("--minimum-images", type=int, default=400)
    split = subparsers.add_parser("split", help="copy source annotations into a deterministic group-aware split")
    split.add_argument("--source", type=Path, default=SOURCE)
    split.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    split.add_argument("--minimum-images", type=int, default=400)
    split.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            images, metadata = validate_source(args.source, args.minimum_images)
            print(f"Valid: {len(images)} images, {len(metadata)} metadata rows, one class (flower).")
        else:
            split_source(args.source, args.output, args.minimum_images, args.seed)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
