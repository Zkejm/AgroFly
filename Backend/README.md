# AgroFly Backend - Orthomap Generation

Create orthomosaics from overlapping aerial/drone photos using OpenCV.

## Setup

```bash
cd Backend
pip install -r requirements.txt
```

## Usage

### From folder of photos

```bash
python run_orthomap.py ./photos
```

### With custom output path

```bash
python run_orthomap.py ./photos ./output/orthomap.png
```

### Options

- `-m scans` — Use affine-based stitching (better for overhead drone shots)
- `-v` — Verbose output

```bash
python run_orthomap.py ./photos -m scans -v
```

### As Python module

```bash
python -m orthomap ./photos
```

### From Python code

```python
from orthomap import load_images, OrthomapStitcher

# Load images
loaded = load_images("./photos")
images = [img for _, img in loaded]

# Stitch
stitcher = OrthomapStitcher(mode="panorama")
status, result = stitcher.stitch(images)

if status == 0:
    import cv2
    cv2.imwrite("orthomap.png", result)
```

## Tips for best results

- Use overlapping photos (30–70% overlap recommended)
- Keep similar lighting/exposure across images
- For drone imagery, try `-m scans` mode
