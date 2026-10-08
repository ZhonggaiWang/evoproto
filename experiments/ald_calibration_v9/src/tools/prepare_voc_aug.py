"""Prepare an independent VOC trainaug mask directory without editing source data.

The archive is the SegmentationClassAug.zip linked by the SEAM authors and this
repository's README. Only regular PNG members under SegmentationClassAug/ are
accepted; macOS metadata is skipped and archive paths are never extracted.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

import numpy as np
from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_output(path, project):
    path = path.absolute()
    if not path.resolve().is_relative_to(project):
        raise ValueError(f"Output escapes project: {path}")
    for parent in (path, *path.parents):
        if parent == project.parent:
            break
        if parent.is_symlink():
            raise ValueError(f"Output uses a symbolic link: {parent}")
    return path


def write_new_or_equal(path, content, project):
    path = checked_output(path, project)
    if path.exists():
        if not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError(f"Unsafe existing output: {path}")
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite different existing output: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
    return True


def validate_mask_bytes(content, filename):
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        array = np.asarray(image)
    if array.ndim != 2 or array.dtype != np.uint8:
        raise ValueError(f"Not a uint8 class-index mask: {filename}, {array.shape}, {array.dtype}")
    invalid = np.setdiff1d(np.unique(array), np.r_[np.arange(21), 255])
    if invalid.size:
        raise ValueError(f"Invalid VOC labels: {filename}: {invalid.tolist()}")


def prepare(args):
    project = Path(__file__).resolve().parents[1]
    destination = checked_output(Path(args.output), project)
    report_path = checked_output(Path(args.report), project)
    source = Path(args.voc_source).resolve(strict=True)
    archive = Path(args.archive).resolve(strict=True)
    seen = set()
    added = 0
    with zipfile.ZipFile(archive) as masks:
        for member in masks.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Archive contains symlink: {member.filename}")
            if member.is_dir() or path.parts[0] == "__MACOSX":
                continue
            if len(path.parts) != 2 or path.parts[0] != "SegmentationClassAug":
                raise ValueError(f"Unexpected archive member: {member.filename}")
            if not re.fullmatch(r"\d{4}_\d{6}\.png", path.name):
                raise ValueError(f"Unexpected mask filename: {path.name}")
            if path.name in seen:
                raise ValueError(f"Duplicate archive mask: {path.name}")
            seen.add(path.name)
            content = masks.read(member)  # Reading verifies the member CRC.
            validate_mask_bytes(content, member.filename)
            added += write_new_or_equal(destination / path.name, content, project)
    original_added = 0
    for split in (project / "datasets/voc/incremental_split").glob("*10-5*.txt"):
        for name in split.read_text().split():
            mask = destination / (name + ".png")
            if not mask.exists():
                content = (source / "SegmentationClass" / mask.name).read_bytes()
                validate_mask_bytes(content, mask.name)
                original_added += write_new_or_equal(mask, content, project)
    split_reports = {}
    errors = []
    verified = set()
    mask_hashes = {}
    for split in sorted((project / "datasets/voc/incremental_split").glob("*10-5*.txt")):
        names = split.read_text().split()
        missing_images, missing_masks, wrong_dimensions = [], [], []
        for name in names:
            image_path = source / "JPEGImages" / (name + ".jpg")
            mask_path = destination / (name + ".png")
            if not image_path.is_file():
                missing_images.append(name)
            if not mask_path.is_file():
                missing_masks.append(name)
            if name in verified or not image_path.is_file() or not mask_path.is_file():
                continue
            validate_mask_bytes(mask_path.read_bytes(), mask_path.name)
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    wrong_dimensions.append(name)
            mask_hashes[name] = sha256(mask_path)
            verified.add(name)
        split_reports[split.name] = {
            "count": len(names), "missing_images": missing_images,
            "missing_masks": missing_masks, "wrong_dimensions": wrong_dimensions,
            "split_sha256": sha256(split),
        }
        if missing_images or missing_masks or wrong_dimensions:
            errors.append(split.name)
    manifest = "\n".join(f"{name} {digest}" for name, digest in sorted(mask_hashes.items()))
    report = {
        "archive": str(archive), "archive_sha256": sha256(archive),
        "archive_source": args.source_url, "archive_masks": len(seen),
        "new_archive_masks": added, "new_original_masks": original_added,
        "voc_source": str(source), "output": str(destination),
        "verified_unique_images": len(verified),
        "mask_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "splits": split_reports, "ok": not errors,
    }
    # Reports are new files as well, to avoid silently replacing earlier evidence.
    write_new_or_equal(report_path, (json.dumps(report, indent=2) + "\n").encode(), project)
    print(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError(f"Incomplete VOC splits: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--voc-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source-url", required=True)
    prepare(parser.parse_args())
