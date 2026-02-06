#!/usr/bin/env python3
"""
Generate an orthomap from overlapping photos.

Usage:
    python run_orthomap.py <input_folder> [output_path]
    python run_orthomap.py ./photos ./output/orthomap.png
    python run_orthomap.py ./photos -m scans -v

Or run as module (from Backend directory):
    python -m orthomap ./photos
"""

from orthomap.cli import main

if __name__ == "__main__":
    exit(main())
