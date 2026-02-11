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

# Stitcher works reliably only when both inputs are under ~6k; we always scale to this before each stitch
STITCH_WORKING_SIZE = 6000

# Min inliers for homography fallback
MIN_INLIERS_HOMOGRAPHY = 10

# Scale down large inputs so homography can find matches
HOMOGRAPHY_WORKING_SIZE = 2500


def _stitch_with_homography(result: "cv2.Mat", img: "cv2.Mat") -> "cv2.Mat | None":
    """Fallback: stitch two images using ORB + findHomography + warp. Returns stitched image or None."""
    res_h, res_w = result.shape[:2]
    img_h, img_w = img.shape[:2]
    scale_r = HOMOGRAPHY_WORKING_SIZE / max(res_h, res_w) if max(res_h, res_w) > HOMOGRAPHY_WORKING_SIZE else 1.0
    scale_i = HOMOGRAPHY_WORKING_SIZE / max(img_h, img_w) if max(img_h, img_w) > HOMOGRAPHY_WORKING_SIZE else 1.0
    if scale_r < 1.0:
        result = cv2.resize(result, (int(res_w * scale_r), int(res_h * scale_r)), interpolation=cv2.INTER_AREA)
        res_h, res_w = result.shape[:2]
    if scale_i < 1.0:
        img = cv2.resize(img, (int(img_w * scale_i), int(img_h * scale_i)), interpolation=cv2.INTER_AREA)
        img_h, img_w = img.shape[:2]
    orb = cv2.ORB_create(nfeatures=8000)
    # For long panoramas, only use bottom and top bands for keypoints (where overlap with next image is)
    crop_ratio = 0.5
    for y_start, y_end in [(0, int(res_h * crop_ratio)), (int(res_h * (1 - crop_ratio)), res_h)]:
        if y_end <= y_start + 50:
            continue
        band = result[y_start:y_end, :]
        kp1, d1 = orb.detectAndCompute(band, None)
        kp2, d2 = orb.detectAndCompute(img, None)
        if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
            continue
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(d1, d2)
        if len(matches) < MIN_INLIERS_HOMOGRAPHY:
            continue
        pts1 = np.array([(kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1] + y_start) for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 10.0)
        if H is not None and mask is not None and mask.sum() >= MIN_INLIERS_HOMOGRAPHY:
            break
    else:
        return None
    if H is None or mask is None or mask.sum() < MIN_INLIERS_HOMOGRAPHY:
        return None
    h1, w1 = result.shape[:2]
    h2, w2 = img.shape[:2]
    corners = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    x_min = min(0, warped_corners[:, 0, 0].min())
    x_max = max(w1, warped_corners[:, 0, 0].max())
    y_min = min(0, warped_corners[:, 0, 1].min())
    y_max = max(h1, warped_corners[:, 0, 1].max())
    tx, ty = int(round(-x_min)), int(round(-y_min))
    out_w = int(round(x_max - x_min))
    out_h = int(round(y_max - y_min))
    if out_w <= 0 or out_h <= 0 or out_w > 32000 or out_h > 32000:
        return None
    H_translate = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    H_full = H_translate @ H
    warped = cv2.warpPerspective(img, H_full, (out_w, out_h))
    canvas = np.zeros((out_h, out_w, 3), dtype=result.dtype)
    canvas[ty : ty + h1, tx : tx + w1] = result
    mask_w = (warped.sum(axis=2) > 0)
    for c in range(3):
        canvas[:, :, c] = np.where(mask_w, warped[:, :, c], canvas[:, :, c])
    return canvas


def create_orthomap_local(
    input_folder: str | Path,
    output_path: str | Path | None = None,
    mode: str = "panorama",
    max_size: int = 0,  # 0 = keep full resolution
    max_images: int | None = 15,  # None = no limit
    direction: str = "vertical",
    sequential: bool = True,
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
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            images.append(img)
    if len(images) < 2:
        if verbose:
            print("Error: Could not load at least 2 images.", file=sys.stderr)
        return None

    if max_images and max_images >= 2 and len(images) > max_images:
        if sequential:
            images = images[:max_images]
            if verbose:
                print(f"Using first {len(images)} images (consecutive order for sequential stitching)")
        else:
            step = (len(images) - 1) / (max_images - 1)
            indices = [int(round(i * step)) for i in range(max_images)]
            images = [images[i] for i in indices]
            if verbose:
                print(f"Using {len(images)} images (sampled from original set)")

    if verbose:
        print("Stitching with OpenCV...")

    if direction == "vertical":
        images = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in images]

    cv2.ocl.setUseOpenCL(False)
    stitcher_mode = cv2.Stitcher_SCANS if mode == "scans" else cv2.Stitcher_PANORAMA
    stitcher = cv2.Stitcher.create(stitcher_mode)
    err_map = {-1: "ERR_NEED_MORE_IMGS", -3: "ERR_HOMOGRAPHY_EST_FAIL", -4: "ERR_CAMERA_PARAMS_ADJUST_FAIL"}

    if sequential:
        result = images[0].copy()
        skipped = 0
        for i in range(1, len(images)):
            if verbose:
                print(f"  Stitching image {i + 1}/{len(images)}...")
            res_h, res_w = result.shape[:2]
            img = images[i]
            img_h, img_w = img.shape[:2]
            # Always scale both to STITCH_WORKING_SIZE so stitcher gets same-size inputs (fewer skips)
            res_scale = STITCH_WORKING_SIZE / max(res_h, res_w) if max(res_h, res_w) > STITCH_WORKING_SIZE else 1.0
            img_scale = STITCH_WORKING_SIZE / max(img_h, img_w) if max(img_h, img_w) > STITCH_WORKING_SIZE else 1.0
            if res_scale < 1.0:
                result = cv2.resize(result, (int(res_w * res_scale), int(res_h * res_scale)), interpolation=cv2.INTER_AREA)
            if img_scale < 1.0:
                img = cv2.resize(img, (int(img_w * img_scale), int(img_h * img_scale)), interpolation=cv2.INTER_AREA)
            status, stitched = stitcher.stitch([result, img])
            if status != cv2.Stitcher_OK:
                status, stitched = stitcher.stitch([img, result])
            if status != cv2.Stitcher_OK:
                other_stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS if stitcher_mode == cv2.Stitcher_PANORAMA else cv2.Stitcher_PANORAMA)
                status, stitched = other_stitcher.stitch([result, img])
                if status != cv2.Stitcher_OK:
                    status, stitched = other_stitcher.stitch([img, result])
            if status == cv2.Stitcher_OK:
                result = stitched
            else:
                stitched = _stitch_with_homography(result, img)
                if stitched is None:
                    stitched = _stitch_with_homography(img, result)
                if stitched is not None:
                    result = stitched
                    if verbose:
                        print(f"    (added with homography fallback)")
                else:
                    skipped += 1
                    if verbose:
                        msg = err_map.get(status, f"Error {status}")
                        print(f"    Skipping image {i + 1} ({msg}), continuing...", file=sys.stderr)
                    continue
        if verbose and skipped:
            print(f"  Skipped {skipped} image(s).")
    else:
        other_mode = "scans" if mode == "panorama" else "panorama"
        result = None
        for try_mode in [mode, other_mode]:
            stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS if try_mode == "scans" else cv2.Stitcher_PANORAMA)
            status, result = stitcher.stitch(images)
            if status == cv2.Stitcher_OK:
                if verbose and try_mode != mode:
                    print(f"  (succeeded with {try_mode} mode)")
                break
            if verbose:
                print(f"  {try_mode} failed ({err_map.get(status, status)}), trying next...")
        if result is None:
            if verbose:
                print(f"Error: Stitching failed - {err_map.get(status, status)}", file=sys.stderr)
            return None

    if direction == "vertical":
        result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)  # undo input rotation

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
        default="panorama",
        help="Stitching mode for OpenCV: panorama or scans (better for overhead drone shots)",
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
