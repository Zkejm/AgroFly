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


def create_orthomap_local(
    input_folder: str | Path,
    output_path: str | Path | None = None,
    mode: str = "panorama",
    verbose: bool = True,
) -> Path | None:
    input_folder = Path(input_folder)
    output_path = Path(output_path or input_folder / "orthomap.png")

    exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
    image_paths = []
    for ext in exts:
        image_paths.extend(input_folder.glob(ext))
    image_paths = sorted(set(image_paths))

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
            images.append(img)
    if len(images) < 2:
        if verbose:
            print("Error: Could not load at least 2 images.", file=sys.stderr)
        return None

    if verbose:
        print("Stitching with OpenCV...")

    stitcher_mode = cv2.Stitcher_SCANS if mode == "scans" else cv2.Stitcher_PANORAMA
    stitcher = cv2.Stitcher.create(stitcher_mode)
    status, result = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        err = {-1: "ERR_NEED_MORE_IMGS", -3: "ERR_HOMOGRAPHY_EST_FAIL", -4: "ERR_CAMERA_PARAMS_ADJUST_FAIL"}
        msg = err.get(status, f"Error {status}")
        if verbose:
            print(f"Error: Stitching failed - {msg}", file=sys.stderr)
        return None

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

    args = parser.parse_args()
    use_local = args.local or not args.webodm

    if use_local:
        output = args.output if args.output is not None else args.input / "orthomap.png"
        result = create_orthomap_local(
            input_folder=args.input,
            output_path=output,
            mode=args.mode,
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
