#!/usr/bin/env python3
"""
Setup YCHC clean directory, link/copy available files, compute SHA-256,
and document file inventory per patient.
"""

import os
import hashlib
import json
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

EXPECTED_FILES = ["P0.nii", "P1.nii", "P2.nii", "P3.nii", "P4.nii", "P5.nii", "GT.nii", "Breast_mask.nii"]


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(16384 * 1024):
            h.update(chunk)
    return h.hexdigest()


def hash_worker(args):
    rel_path, abs_path = args
    csum = compute_sha256(abs_path)
    return f"{csum}  {rel_path}\n"


def main():
    parser = argparse.ArgumentParser(description="Setup and verify YCHC clean dataset")
    parser.add_argument("--src-dir", type=str, default=os.environ.get("YCHC_RAW", "./data/sample_yunnan_hospital"))
    parser.add_argument("--dst-dir", type=str, default="./data/ychc_clean")
    parser.add_argument("--workers", type=int, default=min(16, cpu_count()))
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] Setting up links from {src_dir} to {dst_dir}...")
    case_records = []
    files_to_hash = []

    for c in range(1, 101):
        case_str = str(c)
        src_case_dir = src_dir / case_str
        dst_case_dir = dst_dir / case_str

        if c == 100:
            case_records.append({
                "case_id": c,
                "status": "Zenodo download failed; excluded",
                "present_files": [],
                "missing_files": EXPECTED_FILES,
                "total_present": 0
            })
            continue

        if not src_case_dir.exists():
            case_records.append({
                "case_id": c,
                "status": "Directory missing",
                "present_files": [],
                "missing_files": EXPECTED_FILES,
                "total_present": 0
            })
            continue

        dst_case_dir.mkdir(parents=True, exist_ok=True)
        present = []
        missing = []

        for fname in EXPECTED_FILES:
            s_file = src_case_dir / fname
            d_file = dst_case_dir / fname

            if s_file.exists():
                if not d_file.exists():
                    try:
                        os.link(s_file, d_file)
                    except Exception:
                        import shutil
                        shutil.copy2(s_file, d_file)
                present.append(fname)
                files_to_hash.append((f"{case_str}/{fname}", d_file))
            else:
                missing.append(fname)

        case_records.append({
            "case_id": c,
            "status": "Available (partial)" if missing else "Complete",
            "present_files": present,
            "missing_files": missing,
            "total_present": len(present)
        })

    print(f"[setup] Found {len(files_to_hash)} files. Computing SHA-256 checksums in parallel...")
    with Pool(args.workers) as pool:
        checksum_lines = pool.map(hash_worker, files_to_hash)

    csum_file = dst_dir / "checksums_sha256.txt"
    with open(csum_file, "w") as fp:
        fp.writelines(checksum_lines)

    inv_json = dst_dir / "case_inventory.json"
    with open(inv_json, "w") as fp:
        json.dump(case_records, fp, indent=2)

    print(f"[setup] Done! Wrote {csum_file} and {inv_json}")


if __name__ == "__main__":
    main()
