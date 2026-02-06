"""
Image loader for orthomap generation.
Loads photos from a directory, supporting common image formats.
"""

import cv2
from pathlib import Path
from typing import List, Tuple


def load_images(
    source: str | Path,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"),
) -> List[Tuple[Path, "cv2.Mat"]]:
    """
    Load images from a directory.

    Args:
        source: Path to directory containing images or to a single image file.
        extensions: Supported image file extensions.

    Returns:
        List of (path, image) tuples, sorted by filename for consistent ordering.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Path does not exist: {source}")

    images: List[Tuple[Path, cv2.Mat]] = []

    if source.is_file():
        if source.suffix.lower() in extensions:
            img = cv2.imread(str(source))
            if img is not None:
                images.append((source, img))
        return images

    # Load from directory
    paths = sorted(
        p for p in source.iterdir() if p.suffix.lower() in extensions and p.is_file()
    )

    for path in paths:
        img = cv2.imread(str(path))
        if img is not None:
            images.append((path, img))

    return images
