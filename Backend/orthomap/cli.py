"""CLI for orthomap generation."""

import argparse
import sys
from pathlib import Path

import cv2

from . import load_images, OrthomapStitcher


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an orthomap from overlapping aerial/drone photos."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to folder containing images, or path to a single image",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output path for the orthomap (default: orthomap.png in input folder)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["panorama", "scans"],
        default="panorama",
        help="Stitching mode: panorama (default) or scans (better for overhead drone shots)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print status messages",
    )

    args = parser.parse_args()

    if args.verbose:
        print(f"Loading images from {args.input}...")

    try:
        loaded = load_images(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not loaded:
        print("Error: No images found.", file=sys.stderr)
        return 1

    images = [img for _, img in loaded]
    if args.verbose:
        print(f"Loaded {len(images)} images.")

    if len(images) < 2:
        print("Error: Need at least 2 images to stitch.", file=sys.stderr)
        return 1

    stitcher = OrthomapStitcher(mode=args.mode)
    status, result = stitcher.stitch(images)

    if status != 0:
        codes = {
            -1: "ERR_NEED_MORE_IMGS",
            -3: "ERR_HOMOGRAPHY_EST_FAIL",
            -4: "ERR_CAMERA_PARAMS_ADJUST_FAIL",
        }
        msg = codes.get(status, f"Unknown error (code {status})")
        print(f"Error: Stitching failed - {msg}", file=sys.stderr)
        return 1

    output_path = args.output or (Path(args.input) / "orthomap.png")
    if args.input.is_file():
        output_path = args.output or args.input.parent / "orthomap.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), result)
    if args.verbose:
        print(f"Orthomap saved to {output_path}")

    print(str(output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
