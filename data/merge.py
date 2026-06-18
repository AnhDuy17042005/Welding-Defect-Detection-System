from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from config import (
    CLASS_NAMES,
    PROCESSED_DIR,
    RAW_DIR,
    SPLITS,
)
from validate import (
    find_image_paths,
    find_label_paths,
    find_orphan_labels,
    sanitize_yolo_seg_label,
)


@dataclass
class SplitStats:
    images: int = 0
    labels: int = 0
    missing_labels: int = 0
    bad_labels: int = 0
    dropped_lines: int = 0
    orphan_labels: int = 0


@dataclass
class MergeStats:
    datasets: int = 0
    skipped_datasets: list[str] = field(default_factory=list)
    splits: dict[str, SplitStats] = field(
        default_factory=lambda: {split: SplitStats() for split in SPLITS}
    )
    warnings: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing train/valid/test folders in output before merging.",
    )
    return parser.parse_args()


def safe_prefix(dataset_dir: Path) -> str:
    text = dataset_dir.name.lower()
    chars = []
    for char in text:
        chars.append(char if char.isalnum() else "_")
    return "_".join("".join(chars).split("_"))


def find_dataset_dirs(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {raw_dir}")

    dataset_dirs = []
    for item in sorted(raw_dir.iterdir()):
        if not item.is_dir():
            continue
        has_split = any((item / split / "images").exists() for split in SPLITS)
        if has_split:
            dataset_dirs.append(item)
    return dataset_dirs


def prepare_output_dirs(output_dir: Path, clean: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        for split in SPLITS:
            split_dir = output_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)

    for split in SPLITS:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_pair(
    image_path: Path,
    label_path: Path,
    output_dir: Path,
    split: str,
    prefix: str,
    stats: MergeStats,
) -> None:
    output_stem = f"{prefix}__{image_path.stem}"
    output_image = output_dir / split / "images" / f"{output_stem}{image_path.suffix.lower()}"
    output_label = output_dir / split / "labels" / f"{output_stem}.txt"

    shutil.copy2(image_path, output_image)
    stats.splits[split].images += 1

    if label_path.exists():
        valid_lines, issues = sanitize_yolo_seg_label(label_path)
        if issues:
            stats.splits[split].bad_labels += 1
            stats.splits[split].dropped_lines += len(issues)
            stats.warnings.append(f"{label_path}: {'; '.join(issues)}")
        output_label.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""), encoding="utf-8")
        stats.splits[split].labels += 1
    else:
        output_label.write_text("", encoding="utf-8")
        stats.splits[split].missing_labels += 1
        stats.warnings.append(f"Missing label for image: {image_path}")


def count_orphan_labels(dataset_dir: Path, split: str, stats: MergeStats) -> None:
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    if not labels_dir.exists():
        return

    images = find_image_paths(images_dir, missing_ok=True)
    labels = find_label_paths(labels_dir, missing_ok=True)
    for label_path in find_orphan_labels(images, labels):
        stats.splits[split].orphan_labels += 1
        stats.warnings.append(f"Label has no matching image: {label_path}")


def write_data_yaml(output_dir: Path) -> None:
    lines = [
        "path: .",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    lines.extend(f"  {class_id}: {class_name}" for class_id, class_name in CLASS_NAMES.items())
    lines.append("")

    yaml_text = "\n".join(lines)
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def merge_datasets(raw_dir: Path, output_dir: Path, clean: bool = False) -> MergeStats:
    dataset_dirs = find_dataset_dirs(raw_dir)
    if not dataset_dirs:
        raise RuntimeError(f"No YOLO datasets found in: {raw_dir}")

    prepare_output_dirs(output_dir, clean=clean)
    stats = MergeStats(datasets=len(dataset_dirs))

    for dataset_dir in dataset_dirs:
        prefix = safe_prefix(dataset_dir)
        for split in SPLITS:
            images_dir = dataset_dir / split / "images"
            labels_dir = dataset_dir / split / "labels"

            paths = find_image_paths(images_dir, missing_ok=True)
            if not paths:
                continue

            for image_path in paths:
                label_path = labels_dir / f"{image_path.stem}.txt"
                copy_pair(
                    image_path=image_path,
                    label_path=label_path,
                    output_dir=output_dir,
                    split=split,
                    prefix=prefix,
                    stats=stats,
                )

            count_orphan_labels(dataset_dir, split, stats)

    write_data_yaml(output_dir)
    return stats


def print_summary(stats: MergeStats, output_dir: Path) -> None:
    print(f"Merged datasets: {stats.datasets}")
    print(f"Output folder: {output_dir}")
    print("")

    for split in SPLITS:
        split_stats = stats.splits[split]
        print(
            f"{split}: "
            f"{split_stats.images} images, "
            f"{split_stats.labels} labels, "
            f"{split_stats.missing_labels} missing labels, "
            f"{split_stats.bad_labels} labels with issues, "
            f"{split_stats.dropped_lines} dropped lines, "
            f"{split_stats.orphan_labels} orphan labels"
        )

    print("")
    print(f"data.yaml: {output_dir / 'data.yaml'}")

    if stats.warnings:
        print("")
        print(f"Warnings: {len(stats.warnings)}")
        for warning in stats.warnings[:20]:
            print(f"- {warning}")
        if len(stats.warnings) > 20:
            print(f"... {len(stats.warnings) - 20} more warnings hidden")


def main() -> None:
    args = parse_args()
    stats = merge_datasets(
        raw_dir=args.raw_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        clean=args.clean,
    )
    print_summary(stats, args.output_dir.resolve())


if __name__ == "__main__":
    main()
