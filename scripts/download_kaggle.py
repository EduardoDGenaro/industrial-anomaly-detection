"""Download a single MVTec AD category from the Kaggle mirror.

The Kaggle mirror ``ipythonx/mvtec-ad`` stores files individually, so we can
fetch only one category (e.g. ``bottle`` ~150 MB) instead of the full 5.3 GB.

Authentication uses the official Kaggle client, which reads the token from
``~/.kaggle/access_token`` or the ``KAGGLE_API_TOKEN`` environment variable.

License: MVTec AD is CC BY-NC-SA 4.0 (non-commercial).

Usage:
    .venv\\Scripts\\python.exe scripts\\download_kaggle.py --category bottle --root data\\mvtec
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.mvtec import MVTEC_CATEGORIES, validate_category  # noqa: E402

DEFAULT_DATASET = "ipythonx/mvtec-ad"
HF_DATASET = "foersben/mvtec-ad"


def list_hf_category_files(category: str) -> List[str]:
    """List all file paths under ``category/`` from the HuggingFace MVTec AD mirror."""
    url = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main/{category}?recursive=true&limit=1000"
    req = urllib.request.Request(url, headers={"User-Agent": "mvtec-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        items = json.loads(resp.read().decode("utf-8"))
    return [item["path"] for item in items if item.get("type") == "file"]


def download_from_hf(category: str, root: Path, workers: int = 16) -> bool:
    """Fast parallel download of a single MVTec AD category from HuggingFace CDN."""
    try:
        files = list_hf_category_files(category)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] HF mirror listing failed ({exc}), falling back to Kaggle API...")
        return False

    if not files:
        return False

    print(f"[INFO] {len(files)} files found for '{category}'. Downloading to {root / category} ...")

    def _dl(rel_path: str) -> str:
        dest = root / rel_path
        if dest.exists() and dest.stat().st_size > 0:
            return rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{rel_path}"
        delay = 1.0
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "mvtec-downloader/1.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                dest.write_bytes(data)
                return rel_path
            except Exception:  # noqa: BLE001
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 15.0)
        return rel_path

    done = 0
    with ThreadPoolExecutor(max_workers=max(workers, 16)) as pool:
        futures = {pool.submit(_dl, f): f for f in files}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 50 == 0 or done == len(files):
                print(f"  {done}/{len(files)}")
    return True


def list_category_files(api, dataset: str, category: str) -> List[str]:
    """List all file paths of a dataset that belong to ``category/``."""
    prefix = f"{category}/"
    names: List[str] = []
    token = None
    while True:
        result = api.dataset_list_files(dataset, page_token=token, page_size=200)
        files = getattr(result, "files", None) or []
        names.extend(f.name for f in files if f.name.startswith(prefix))
        token = getattr(result, "next_page_token", None) or getattr(result, "nextPageToken", None)
        if not token:
            break
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True, choices=MVTEC_CATEGORIES)
    parser.add_argument("--root", default="data/mvtec", help="Destination root (category folder is created inside).")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset ref.")
    parser.add_argument("--workers", type=int, default=2, help="Parallel downloads (Kaggle rate-limits >2-3).")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if download_from_hf(args.category, root, workers=args.workers):
        report = validate_category(root, args.category)
        print(report.summary())
        return 0 if report.ok else 2

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"[INFO] Listing '{args.category}/' files in Kaggle dataset '{args.dataset}'...")
    names = list_category_files(api, args.dataset, args.category)
    if not names:
        print(f"[ERROR] No files found for category '{args.category}'.")
        return 1
    print(f"[INFO] {len(names)} files found. Downloading to {root / args.category} ...")

    def _fetch(name: str) -> str:
        # NOTE: the Kaggle client appends the file's relative path to ``path``,
        # so we pass the dataset root (not the file's parent folder).
        if (root / name).exists():
            return name
        delay = 2.0
        for attempt in range(8):
            try:
                api.dataset_download_file(args.dataset, name, path=str(root), quiet=True, unzip=True)
                return name
            except Exception as exc:  # noqa: BLE001
                if "429" in str(exc) and attempt < 7:
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                raise
        return name

    failed: List[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch, n): n for n in names}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{futures[fut]}: {exc}")
            done += 1
            if done % 50 == 0 or done == len(names):
                print(f"  {done}/{len(names)}")

    if failed:
        print(f"[ERROR] {len(failed)} downloads failed (re-run to resume):")
        for f in failed[:10]:
            print("   ", f)
        return 1

    report = validate_category(root, args.category)
    print(report.summary())
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
