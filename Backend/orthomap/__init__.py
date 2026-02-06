"""
Orthomap - Generate orthomosaics from overlapping drone/aerial photos.
"""

from .loader import load_images
from .stitcher import OrthomapStitcher

__all__ = ["load_images", "OrthomapStitcher"]
