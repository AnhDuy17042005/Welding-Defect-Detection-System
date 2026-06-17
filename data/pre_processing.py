from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR       = BASE_DIR / "dataset" / "raw"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CLASS_NAMES = {
    0: "crack",
    1: "porosity",
    2: "spatter",
    3: "welding_line",
}


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


def image_paths(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sanitize_yolo_seg_label(label_path: Path) -> tuple[list[str], list[str]]:
    issues = []
    valid_lines = []
    lines = label_path.read_text(encoding="utf-8").splitlines()

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) < 7:
            issues.append(f"line {line_number}: segmentation label needs class + at least 3 points")
            continue

        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            issues.append(f"line {line_number}: non-numeric value")
            continue

        if class_id not in CLASS_NAMES:
            issues.append(f"line {line_number}: unsupported class id {class_id}")
            continue

        if len(coords) % 2 != 0:
            issues.append(f"line {line_number}: odd number of polygon coordinates")
            continue

        if len(coords) < 6:
            issues.append(f"line {line_number}: polygon has fewer than 3 points")
            continue

        out_of_range = [value for value in coords if value < 0.0 or value > 1.0]
        if out_of_range:
            issues.append(f"line {line_number}: coordinates outside [0, 1]")
            continue

        valid_lines.append(stripped)

    return valid_lines, issues


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

    image_stems = {path.stem for path in image_paths(images_dir)}
    for label_path in sorted(labels_dir.glob("*.txt")):
        if label_path.stem not in image_stems:
            stats.splits[split].orphan_labels += 1
            stats.warnings.append(f"Label has no matching image: {label_path}")


def write_data_yaml(output_dir: Path) -> None:
    yaml_text = "\n".join(
        [
            "path: .",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "",
            "nc: 4",
            "names:",
            "  0: crack",
            "  1: porosity",
            "  2: spatter",
            "  3: welding_line",
            "",
        ]
    )
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

            paths = image_paths(images_dir)
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