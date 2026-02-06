"""
Orthomosaic stitcher using OpenCV.
Stitches overlapping aerial/drone photos into a single orthomap.
"""

import cv2
from pathlib import Path
from typing import List, Tuple

import numpy as np


class OrthomapStitcher:
    """
    Creates orthomosaics from overlapping images using OpenCV's stitcher.

    Works best with:
    - Overlapping photos (30-70% overlap recommended)
    - Similar exposure/lighting
    - Ordered or grid-based flight patterns
    """

    def __init__(self, mode: str = "panorama"):
        """
        Args:
            mode: "panorama" for homography-based (default), "scans" for affine-based.
                  Use "scans" for overhead drone imagery with minimal perspective change.
        """
        if mode == "scans":
            self.stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
        else:
            self.stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

    def stitch(self, images: List["cv2.Mat"]) -> Tuple[int, "np.ndarray | None"]:
        """
        Stitch a list of images into an orthomosaic.

        Args:
            images: List of OpenCV images (BGR format).

        Returns:
            Tuple of (status_code, result_image).
            status_code: cv2.Stitcher status (OK=0 on success).
            result_image: Stitched orthomosaic or None on failure.
        """
        if len(images) < 2:
            return cv2.Stitcher_ERR_NEED_MORE_IMGS, None

        status, pano = self.stitcher.stitch(images)
        return status, pano

    def stitch_from_paths(
        self, image_paths: List[Path]
    ) -> Tuple[int, "np.ndarray | None"]:
        """
        Load and stitch images from file paths.

        Args:
            image_paths: List of paths to image files.

        Returns:
            Tuple of (status_code, result_image).
        """
        images = []
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is not None:
                images.append(img)
        return self.stitch(images)


def create_orthomap(
    images: List["cv2.Mat"],
    output_path: str | Path | None = None,
    mode: str = "panorama",
) -> Tuple[bool, "np.ndarray | None"]:
    """
    Convenience function to create an orthomap from images.

    Args:
        images: List of OpenCV images.
        output_path: Optional path to save the result (PNG or JPG).
        mode: "panorama" or "scans".

    Returns:
        Tuple of (success, result_image).
    """
    stitcher = OrthomapStitcher(mode=mode)
    status, result = stitcher.stitch(images)

    if status == cv2.Stitcher_OK and result is not None:
        if output_path:
            cv2.imwrite(str(output_path), result)
        return True, result

    return False, result
