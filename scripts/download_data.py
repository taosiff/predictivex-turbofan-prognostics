"""
PredictiveX Dataset Acquisition Script for NASA C-MAPSS Turbofan Degradation Dataset.
Downloads and verifies all subsets: FD001, FD002, FD003, FD004 (train, test, and RUL).
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path

# Base directories
RAW_DATA_DIR = Path("data/raw/cmapss")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Primary and fallback mirrors
MIRRORS = [
    "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/",
    "https://raw.githubusercontent.com/Azure/lstms_for_predictive_maintenance/master/Dataset/",
]

FILES = [
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
    "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
    "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
]

def download_file(filename: str, target_path: Path):
    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"[OK] {filename} already exists ({target_path.stat().st_size:,} bytes).")
        return True

    for mirror in MIRRORS:
        url = mirror + filename
        print(f"Attempting download: {url} -> {target_path}")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(target_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            print(f"[SUCCESS] Downloaded {filename} ({len(data):,} bytes).")
            return True
        except Exception as e:
            print(f"[FAILED] from {url}: {e}")
            if target_path.exists():
                target_path.unlink()
    return False

def verify_files():
    print("\n--- Verifying Dataset Files ---")
    all_ok = True
    summary = []
    for fname in FILES:
        fpath = RAW_DATA_DIR / fname
        if not fpath.exists():
            print(f"[ERROR] Missing {fname}")
            all_ok = False
            continue
        size = fpath.stat().st_size
        with open(fpath, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            lines = sum(1 for _ in f)
        summary.append({
            "file": fname,
            "size_bytes": size,
            "lines": lines,
            "md5": md5
        })
        print(f"[VERIFIED] {fname:16s} | Lines: {lines:7d} | Size: {size:10,d} bytes | MD5: {md5}")
    return all_ok, summary

if __name__ == "__main__":
    print("=" * 60)
    print("PredictiveX: NASA C-MAPSS Dataset Acquisition")
    print("=" * 60)
    for fname in FILES:
        dest = RAW_DATA_DIR / fname
        download_file(fname, dest)

    ok, summary = verify_files()
    if ok:
        print("\nAll NASA C-MAPSS files (FD001 - FD004) acquired and verified successfully.")
    else:
        print("\nSome files failed verification. Please check the network or mirrors.")
        sys.exit(1)
