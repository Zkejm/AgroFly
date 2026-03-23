import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

TASK_QUEUED = 10
TASK_RUNNING = 20
TASK_FAILED = 30
TASK_COMPLETED = 40
TASK_CANCELED = 50

STITCH_WORKING_SIZE = 6000  # kept for backward compat, no longer used internally

# ── Resolution constants for the high-quality pipeline ───────────────────────
# Features are detected at _FEATURE_DETECT_SIZE (fast, accurate geometry).
# Warping/compositing is done at _COMPOSE_SIZE per image (high quality, fits RAM).
# Both scales are corrected in camera intrinsics so geometry is always consistent.
_FEATURE_DETECT_SIZE = 2000   # max long-side px for feature detection
_COMPOSE_SIZE = 1500          # max long-side px per image during warp/blend


def _stitch_hq(images: list, mode: str = "scans", verbose: bool = True):
    """
    High-quality stitching via the OpenCV detail pipeline:
      SIFT features (@_FEATURE_DETECT_SIZE) → BestOf2Nearest matching →
      HomographyBased estimator → BundleAdjusterRay →
      plane/spherical warper (@_COMPOSE_SIZE per image) →
      GAIN_BLOCKS exposure compensation → GraphCut seam finding →
      Multi-band blending.

    Returns the composited BGR image (uint8) or None on failure.
    """
    cv2.ocl.setUseOpenCL(False)

    # ── Feature detection (at reduced scale for speed) ───────────────────────
    try:
        finder = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02, edgeThreshold=15)
        det_name = "SIFT"
    except AttributeError:
        finder = cv2.ORB_create(nfeatures=8000)
        det_name = "ORB"

    if verbose:
        print(f"  Detecting {det_name} features...")

    h0, w0 = images[0].shape[:2]
    feat_scale = min(1.0, _FEATURE_DETECT_SIZE / max(h0, w0))

    features = []
    for img in images:
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * feat_scale), int(h * feat_scale)), interpolation=cv2.INTER_AREA) if feat_scale < 1.0 else img
        feat = cv2.detail.computeImageFeatures2(finder, small)
        features.append(feat)

    if verbose:
        print(f"  Found {sum(len(f.keypoints) for f in features)} keypoints across {len(features)} images")

    # ── Matching ─────────────────────────────────────────────────────────────
    if verbose:
        print("  Matching features...")
    matcher = cv2.detail.BestOf2NearestMatcher(False, 0.65)
    pairwise_matches = matcher.apply2(features)
    matcher.collectGarbage()

    # ── Drop images with insufficient overlap ────────────────────────────────
    orig_indices = cv2.detail.leaveBiggestComponent(features, pairwise_matches, 0.3)
    if len(orig_indices) < 2:
        if verbose:
            print("  Not enough matching images for HQ pipeline.", file=sys.stderr)
        return None

    selected = [images[int(i)] for i in orig_indices]
    if len(orig_indices) < len(images) and verbose:
        print(f"  Using {len(orig_indices)}/{len(images)} images (others had insufficient overlap)")

    # ── Camera estimation ────────────────────────────────────────────────────
    estimator = cv2.detail_HomographyBasedEstimator()
    ok, cameras = estimator.apply(features, pairwise_matches, None)
    if not ok or not cameras:
        if verbose:
            print("  Camera estimation failed.", file=sys.stderr)
        return None

    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    # ── Bundle adjustment (global refinement) ────────────────────────────────
    adjuster = cv2.detail_BundleAdjusterRay()
    adjuster.setConfThresh(0.5)
    adjuster.apply(features, pairwise_matches, cameras)

    # ── Wave correction ──────────────────────────────────────────────────────
    rmats = [np.copy(cam.R) for cam in cameras]
    try:
        cv2.detail.waveCorrect(rmats, cv2.detail.WAVE_CORRECT_HORIZ)
        for cam, R in zip(cameras, rmats):
            cam.R = R
    except cv2.error:
        pass

    # ── Scale camera intrinsics to compose-scale pixels ──────────────────────
    # Features were detected at feat_scale; we warp at compose_scale.
    # Net multiplier brings intrinsics from feature-scale → compose-scale pixels.
    compose_scale = min(1.0, _COMPOSE_SIZE / max(h0, w0))
    net_scale = compose_scale / feat_scale
    for cam in cameras:
        cam.focal *= net_scale
        cam.ppx *= net_scale
        cam.ppy *= net_scale

    focals = sorted(cam.focal for cam in cameras)
    warped_image_scale = max(1.0, focals[len(focals) // 2])

    # ── Warp at compose scale ─────────────────────────────────────────────────
    # "plane" matches the flat-ground geometry of nadir drone shots;
    # "spherical" is used for classic panoramas.
    warper_type = "plane" if mode == "scans" else "spherical"
    warper = cv2.PyRotationWarper(warper_type, warped_image_scale)

    if verbose:
        pct = int(compose_scale * 100)
        print(f"  Warping images at {pct}% of input resolution ({_COMPOSE_SIZE}px cap)...")

    corners, sizes, images_warped, masks_warped = [], [], [], []
    for img, cam in zip(selected, cameras):
        h, w = img.shape[:2]
        if compose_scale < 1.0:
            compose_img = cv2.resize(img, (int(w * compose_scale), int(h * compose_scale)), interpolation=cv2.INTER_AREA)
        else:
            compose_img = img
        ch, cw = compose_img.shape[:2]
        K = cam.K().astype(np.float32)
        R = cam.R.astype(np.float32)
        try:
            corner, warped = warper.warp(compose_img, K, R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        except cv2.error as exc:
            if verbose:
                print(f"  Warp error: {exc}", file=sys.stderr)
            return None
        corners.append(corner)
        sizes.append((warped.shape[1], warped.shape[0]))
        images_warped.append(warped)
        mask = np.ones((ch, cw), dtype=np.uint8) * 255
        _, mask_w = warper.warp(mask, K, R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        masks_warped.append(mask_w)

    # ── Exposure compensation (per-image scalar gain) ────────────────────────
    # ExposureCompensator_GAIN uses per-image statistics only — no full-canvas
    # buffer — so it stays memory-safe regardless of how large the mosaic is.
    if verbose:
        print("  Exposure compensation...")
    try:
        compensator = cv2.detail.ExposureCompensator_createDefault(
            cv2.detail.ExposureCompensator_GAIN
        )
        compensator.feed(corners, images_warped, masks_warped)
        for i in range(len(images_warped)):
            compensator.apply(i, corners[i], images_warped[i], masks_warped[i])
    except cv2.error:
        if verbose:
            print("  Exposure compensation skipped (insufficient memory).")

    # ── Seam finding (GraphCut minimises colour discontinuity) ───────────────
    if verbose:
        print("  Finding seams (GraphCut)...")
    images_f = [img.astype(np.float32) for img in images_warped]
    masks_seam = [m.copy() for m in masks_warped]
    try:
        seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
        seam_finder.find(images_f, corners, masks_seam)
    except Exception:
        masks_seam = masks_warped  # fall back to raw masks if seam finding fails

    # ── Multi-band blending ──────────────────────────────────────────────────
    if verbose:
        print("  Multi-band blending...")
    dst_roi = cv2.detail.resultRoi(corners, sizes)
    blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_MULTI_BAND)
    blender.prepare(dst_roi)

    for i in range(len(images_warped)):
        blender.feed(images_warped[i].astype(np.int16), masks_seam[i], corners[i])

    try:
        result, _ = blender.blend(None, None)
        return cv2.convertScaleAbs(result)
    except cv2.error as exc:
        if verbose:
            print(f"  Blending error: {exc}", file=sys.stderr)
        return None


def _stitch_fallback(images: list, mode: str = "scans", verbose: bool = True):
    """
    Fallback to OpenCV's built-in Stitcher (less quality but more tolerant of
    difficult image sets). Tries both scans and panorama modes.
    """
    cv2.ocl.setUseOpenCL(False)
    modes = (
        [cv2.Stitcher_SCANS, cv2.Stitcher_PANORAMA]
        if mode == "scans"
        else [cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS]
    )
    for sm in modes:
        stitcher = cv2.Stitcher.create(sm)
        try:
            status, result = stitcher.stitch(images)
            if status == cv2.Stitcher_OK:
                return result
        except cv2.error:
            pass
    return None


def create_orthomap_local(
    input_folder: str | Path,
    output_path: str | Path | None = None,
    mode: str = "scans",
    max_size: int = 0,
    max_images: int | None = 15,
    direction: str = "vertical",
    sequential: bool = True,   # kept for CLI backward compatibility
    working_size: int = 0,     # kept for CLI backward compatibility
    verbose: bool = True,
) -> Path | None:
    input_folder = Path(input_folder)
    output_path = Path(output_path or input_folder / "orthomap.png")

    exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
    image_paths = []
    for ext in exts:
        image_paths.extend(input_folder.glob(ext))
    output_path_resolved = output_path.resolve()
    image_paths = sorted(set(p for p in image_paths if p.resolve() != output_path_resolved))

    if len(image_paths) < 2:
        if verbose:
            print("Error: Need at least 2 images.", file=sys.stderr)
        return None

    if verbose:
        print(f"Found {len(image_paths)} images in {input_folder}")

    images = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            h, w = img.shape[:2]
            if max_size > 0 and max(h, w) > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            images.append(img)

    if len(images) < 2:
        if verbose:
            print("Error: Could not load at least 2 images.", file=sys.stderr)
        return None

    if max_images and max_images >= 2 and len(images) > max_images:
        images = images[:max_images]
        if verbose:
            print(f"Using first {len(images)} images")

    if direction == "vertical":
        images = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in images]

    if verbose:
        print(f"Stitching {len(images)} images...")

    result = _stitch_hq(images, mode=mode, verbose=verbose)
    if result is None:
        if verbose:
            print("  High-quality pipeline did not succeed, trying fallback Stitcher...")
        result = _stitch_fallback(images, mode=mode, verbose=verbose)

    if result is None:
        if verbose:
            print("Error: Stitching failed with all methods.", file=sys.stderr)
        return None

    if direction == "vertical":
        result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    if verbose:
        print(f"Saved {output_path}")
    return output_path


def create_orthomap(
    input_folder: str | Path,
    output_path: str | Path | None = None,
    base_url: str | None = None,
    username: str | None = None,
    password: str | None = None,
    orthophoto_resolution: float = 5,
    poll_interval: int = 3,
    cleanup_project: bool = True,
    verbose: bool = True,
) -> Path | None:
    base_url = (base_url or os.environ.get("WEBODM_URL") or "http://localhost:8000").rstrip("/")
    username = username or os.environ.get("WEBODM_USER") or "admin"
    password = password or os.environ.get("WEBODM_PASSWORD") or "admin"
    input_folder = Path(input_folder)
    output_path = Path(output_path or "orthophoto.tif")

    # Collect images
    exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
    images = []
    for ext in exts:
        images.extend(input_folder.glob(ext))
    images = sorted(set(images))

    if len(images) < 2:
        if verbose:
            print("Error: Need at least 2 images.", file=sys.stderr)
        return None

    if verbose:
        print(f"Found {len(images)} images in {input_folder}")

    try:
        r = requests.post(
            f"{base_url}/api/token-auth/",
            data={"username": username, "password": password},
            timeout=10,
        )
        r.raise_for_status()
        auth = r.json()
    except requests.RequestException as e:
        if verbose:
            print(f"Error: Cannot reach WebODM at {base_url}: {e}", file=sys.stderr)
        return None
    if "token" not in auth:
        if verbose:
            print(f"Error: Auth failed: {auth}", file=sys.stderr)
        return None
    headers = {"Authorization": f"JWT {auth['token']}"}

    try:
        r = requests.post(
            f"{base_url}/api/projects/",
            headers=headers,
            data={"name": f"Orthomap_{input_folder.name}"},
            timeout=10,
        )
        r.raise_for_status()
        project = r.json()
    except requests.RequestException as e:
        if verbose:
            print(f"Error: Cannot create project: {e}", file=sys.stderr)
        return None

    project_id = project["id"]
    if verbose:
        print(f"Created project {project_id}")

    files = [
        ("images", (p.name, open(p, "rb"), "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"))
        for p in images
    ]
    options = json.dumps([{"name": "orthophoto-resolution", "value": orthophoto_resolution}])

    try:
        r = requests.post(
            f"{base_url}/api/projects/{project_id}/tasks/",
            headers=headers,
            files=files,
            data={"options": options},
            timeout=300,
        )
        for _, fdata in files:
            fdata[1].close()
    except requests.RequestException as e:
        for _, fdata in files:
            fdata[1].close()
        if verbose:
            print(f"Error: Cannot create task: {e}", file=sys.stderr)
        if cleanup_project:
            requests.delete(f"{base_url}/api/projects/{project_id}/", headers=headers)
        return None

    if r.status_code != 200 and r.status_code != 201:
        if verbose:
            print(f"Error: Task creation failed: {r.status_code} {r.text}", file=sys.stderr)
        if cleanup_project:
            requests.delete(f"{base_url}/api/projects/{project_id}/", headers=headers)
        return None

    task = r.json()
    task_id = task["id"]
    if verbose:
        print(f"Created task {task_id}, processing...")

    while True:
        time.sleep(poll_interval)
        try:
            r = requests.get(
                f"{base_url}/api/projects/{project_id}/tasks/{task_id}/",
                headers=headers,
                timeout=10,
            )
            r.raise_for_status()
            task = r.json()
        except requests.RequestException as e:
            if verbose:
                print(f"Error polling task: {e}", file=sys.stderr)
            continue

        status = task.get("status")
        if status == TASK_COMPLETED:
            if verbose:
                print("Task completed!")
            break
        if status == TASK_FAILED:
            err = task.get("last_error", "Unknown error")
            if verbose:
                print(f"Task failed: {err}", file=sys.stderr)
            if cleanup_project:
                requests.delete(f"{base_url}/api/projects/{project_id}/", headers=headers)
            return None
        if status == TASK_CANCELED:
            if verbose:
                print("Task was canceled.", file=sys.stderr)
            if cleanup_project:
                requests.delete(f"{base_url}/api/projects/{project_id}/", headers=headers)
            return None

        elapsed = task.get("processing_time", 0) / 1000
        if elapsed < 0:
            elapsed = 0
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        if verbose:
            sys.stdout.write(f"\rProcessing... [{h:02d}:{m:02d}:{s:02d}]")
            sys.stdout.flush()

    download_url = f"{base_url}/api/projects/{project_id}/tasks/{task_id}/download/orthophoto.tif"
    try:
        r = requests.get(download_url, headers=headers, stream=True, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        if verbose:
            print(f"\nError downloading orthophoto: {e}", file=sys.stderr)
        if cleanup_project:
            requests.delete(f"{base_url}/api/projects/{project_id}/", headers=headers)
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    if cleanup_project:
        requests.delete(f"{base_url}/api/projects/{project_id}/", headers=headers)

    if verbose:
        print(f"\nSaved {output_path}")
    return output_path


def main() -> int:
    _default_data = Path(__file__).resolve().parent.parent / "data"

    parser = argparse.ArgumentParser(
        description="Create an orthomap from photos using WebODM API."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=_default_data,
        help=f"Folder containing images (default: {_default_data})",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output path for orthophoto (default: <input>/orthophoto.tif)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="WebODM base URL (default: env WEBODM_URL or http://localhost:8000)",
    )
    parser.add_argument("--user", default=None, help="WebODM username")
    parser.add_argument("--password", default=None, help="WebODM password")
    parser.add_argument(
        "-r",
        "--resolution",
        type=float,
        default=5,
        help="Orthophoto resolution in cm/pixel (default: 5)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep project in WebODM (WebODM mode)")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use OpenCV Stitcher (offline, no WebODM needed). Default.",
    )
    parser.add_argument(
        "--webodm",
        action="store_true",
        help="Use WebODM API instead of local OpenCV.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["panorama", "scans"],
        default="scans",
        help="Stitching mode: scans (default, fewer skips for drone/aerial) or panorama",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=0,
        metavar="N",
        help="Max pixel dimension for input images; 0 = keep full resolution (default). Set e.g. 2000 to reduce memory/speed.",
    )
    parser.add_argument(
        "-d",
        "--direction",
        choices=["horizontal", "vertical"],
        default="vertical",
        help="Output layout: vertical (tall) or horizontal (wide), default: vertical",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=15,
        metavar="N",
        help="Max images to use (samples evenly when exceeded, helps with large sets, default: 15). Use 0 for no limit.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Stitch all images at once (default: stitch one by one, sequential).",
    )
    parser.add_argument(
        "--working-size",
        type=int,
        default=0,
        metavar="N",
        help="Stitching resolution (longest side in px). 0 = use default (6000, high detail). Lower = fewer crashes, less detail.",
    )

    args = parser.parse_args()
    use_local = args.local or not args.webodm

    if use_local:
        output = args.output if args.output is not None else args.input / "orthomap.png"
        max_imgs = args.max_images if args.max_images > 0 else None
        result = create_orthomap_local(
            input_folder=args.input,
            output_path=output,
            mode=args.mode,
            max_size=args.max_size,
            max_images=max_imgs,
            direction=args.direction,
            sequential=not args.batch,
            working_size=args.working_size or 0,
            verbose=not args.quiet,
        )
    else:
        output = args.output if args.output is not None else args.input / "orthophoto.tif"
        result = create_orthomap(
            input_folder=args.input,
            output_path=output,
            base_url=args.url,
            username=args.user,
            password=args.password,
            orthophoto_resolution=args.resolution,
            cleanup_project=not args.no_cleanup,
            verbose=not args.quiet,
        )
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
