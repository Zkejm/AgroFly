import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import requests

TASK_QUEUED = 10
TASK_RUNNING = 20
TASK_FAILED = 30
TASK_COMPLETED = 40
TASK_CANCELED = 50

# Working resolution for stitching; output detail matches this (6000 = high detail). We scale after each round to stay under OpenCV limit.
STITCH_WORKING_SIZE = 6000


def create_orthomap_local(
    input_folder: str | Path,
    output_path: str | Path | None = None,
    mode: str = "scans",
    max_size: int = 0,  # 0 = keep full resolution
    max_images: int | None = 15,  # None = no limit
    direction: str = "vertical",
    sequential: bool = True,
    working_size: int = 0,  # 0 = use STITCH_WORKING_SIZE constant
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
    # OpenCV returns 0=OK, 1/-1=need more imgs (not enough overlap/features), 2/-3=homography fail, 3/-4=camera params
    err_map = {
        1: "not enough overlap/features (try more overlap, or --working-size)",
        -1: "not enough overlap/features",
        2: "homography failed",
        -3: "homography failed",
        3: "camera params failed",
        -4: "camera params failed",
    }

    if sequential:
        wsize = working_size or STITCH_WORKING_SIZE
        def scale_to_working(img):
            h, w = img.shape[:2]
            if max(h, w) <= wsize:
                return img
            s = wsize / max(h, w)
            return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

        def try_stitch_two(a, b):
            a_s, b_s = scale_to_working(a), scale_to_working(b)
            status, out = -1, None
            for x, y in [(a_s, b_s), (b_s, a_s)]:
                try:
                    s, o = stitcher.stitch([x, y])
                    status, out = s, o
                    if s == cv2.Stitcher_OK:
                        return s, o
                except cv2.error:
                    pass
            other = cv2.Stitcher.create(cv2.Stitcher_SCANS if stitcher_mode == cv2.Stitcher_PANORAMA else cv2.Stitcher_PANORAMA)
            for x, y in [(a_s, b_s), (b_s, a_s)]:
                try:
                    s, o = other.stitch([x, y])
                    status, out = s, o
                    if s == cv2.Stitcher_OK:
                        return s, o
                except cv2.error:
                    pass
            return status, out

        layer = [img.copy() for img in images]
        round_num = 0
        no_progress_rounds = 0
        while len(layer) > 1:
            round_num += 1
            if verbose:
                print(f"  Round {round_num}: pairing {len(layer)} images...")
            next_layer = []
            for i in range(0, len(layer), 2):
                if i + 1 >= len(layer):
                    next_layer.append(scale_to_working(layer[i]))
                    continue
                status, stitched = try_stitch_two(layer[i], layer[i + 1])
                if status == cv2.Stitcher_OK:
                    next_layer.append(scale_to_working(stitched))
                else:
                    if verbose:
                        print(f"    Pair ({i + 1},{i + 2}) failed, keeping both.")
                    next_layer.append(scale_to_working(layer[i]))
                    next_layer.append(scale_to_working(layer[i + 1]))
            if len(next_layer) >= len(layer):
                no_progress_rounds += 1
                if no_progress_rounds >= 3:
                    if verbose:
                        print(f"  3 rounds with no progress, stitching all {len(layer)} images together...")
                    try:
                        scaled = [scale_to_working(img) for img in layer]
                        status, batch_result = stitcher.stitch(scaled)
                        if status != cv2.Stitcher_OK:
                            other = cv2.Stitcher.create(cv2.Stitcher_SCANS if stitcher_mode == cv2.Stitcher_PANORAMA else cv2.Stitcher_PANORAMA)
                            status, batch_result = other.stitch(scaled)
                        if status == cv2.Stitcher_OK:
                            next_layer = [scale_to_working(batch_result)]
                            if verbose:
                                print("    Batch stitch succeeded.")
                    except cv2.error:
                        if verbose:
                            print("    Batch stitch failed, continuing with current layer.")
                    no_progress_rounds = 0
            else:
                no_progress_rounds = 0
            layer = next_layer
        result = layer[0]
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
