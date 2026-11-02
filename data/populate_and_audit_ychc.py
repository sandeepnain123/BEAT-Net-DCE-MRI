#!/usr/bin/env python3
"""
Populate and audit YCHC clean directory from raw cohort directory.
Computes SHA-256 checksums, generates case inventory, analyzes spacing and dimensions
across all phases, and inspects NIfTI headers for phase ordering.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pandas as pd
import nibabel as nib
import numpy as np

EXPECTED_FILES = ["P0.nii", "P1.nii", "P2.nii", "P3.nii", "P4.nii", "P5.nii", "Breast_mask.nii", "GT.nii"]


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
    parser = argparse.ArgumentParser(description="Populate and audit YCHC cohort")
    parser.add_argument("--src-dir", type=str, default=os.environ.get("YCHC_RAW", "./data/sample_yunnan_hospital"))
    parser.add_argument("--dst-dir", type=str, default="./data/ychc_clean")
    parser.add_argument("--workers", type=int, default=min(16, cpu_count()))
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"[1] Hardlinking/copying files from {src_dir} to {dst_dir}...")
    print("=" * 80)

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
                    except FileExistsError:
                        pass
                    except Exception:
                        import shutil
                        shutil.copy2(s_file, d_file)
                elif d_file.stat().st_size != s_file.stat().st_size:
                    try:
                        d_file.unlink()
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
            "status": "Complete" if len(present) == 8 else f"Partial ({len(present)}/8)",
            "present_files": present,
            "missing_files": missing,
            "total_present": len(present)
        })

    print(f"Total files to hash across all cases: {len(files_to_hash)}")

    print("\n" + "=" * 80)
    print("[2] Computing SHA-256 Checksums in parallel...")
    print("=" * 80)
    with Pool(args.workers) as pool:
        checksum_lines = pool.map(hash_worker, files_to_hash)

    csum_file = dst_dir / "checksums_sha256.txt"
    with open(csum_file, "w") as fp:
        fp.writelines(checksum_lines)
    print(f"Saved {len(checksum_lines)} checksums to {csum_file}")

    inv_json = dst_dir / "case_inventory.json"
    with open(inv_json, "w") as fp:
        json.dump(case_records, fp, indent=2)

    inv_csv = dst_dir / "ychc_case_inventory.csv"
    df_inv = pd.DataFrame([{
        "case_id": r["case_id"],
        "status": r["status"],
        "present_files": ";".join(r["present_files"]),
        "missing_files": ";".join(r["missing_files"]),
        "total_present": r["total_present"]
    } for r in case_records])
    df_inv.to_csv(inv_csv, index=False)
    print(f"Saved case inventory to {inv_csv}")

    print("\n" + "=" * 80)
    print("[3] Inspecting Image Shape, Spacing, and Headers across Phases...")
    print("=" * 80)

    shapes = {p: [] for p in ["P0", "P1", "P2", "P3", "P4", "P5", "Breast_mask", "GT"]}
    spacings = {p: [] for p in ["P0", "P1", "P2", "P3", "P4", "P5", "Breast_mask", "GT"]}
    descriptions = []

    for c in range(1, 100):
        case_str = str(c)
        case_dir = dst_dir / case_str
        if not case_dir.exists():
            continue

        for phase in ["P0", "P1", "P2", "P3", "P4", "P5", "Breast_mask", "GT"]:
            fpath = case_dir / f"{phase}.nii"
            if fpath.exists():
                img = nib.load(str(fpath))
                hdr = img.header
                shapes[phase].append(img.shape)
                spacings[phase].append(tuple(np.round(hdr.get_zooms()[:3], 4)))
                if phase in ["P0", "P1", "P2", "P5"] and c in [1, 2, 10, 50, 99]:
                    desc = hdr.get('descrip', b'').tobytes().decode('latin1', errors='ignore').strip('\x00 ')
                    aux = hdr.get('aux_file', b'').tobytes().decode('latin1', errors='ignore').strip('\x00 ')
                    descriptions.append({
                        "case": c, "phase": phase, "descrip": desc, "aux_file": aux,
                        "dim": img.shape, "pixdim": tuple(np.round(hdr.get_zooms()[:3], 4)),
                        "datatype": int(hdr['datatype']), "qform_code": int(hdr['qform_code']), "sform_code": int(hdr['sform_code'])
                    })

    for p in shapes:
        unique_shapes = list(set(shapes[p]))
        unique_spacings = list(set(spacings[p]))
        print(f"Phase {p:12s}: {len(shapes[p])} cases | Unique Spacings: {unique_spacings} | Sample Shapes: {unique_shapes[:3]}")

    with open(dst_dir / "yunan_geometry_summary.json", "w") as fp:
        json.dump({
            "sample_headers": descriptions
        }, fp, indent=2)

    print("\n" + "=" * 80)
    print("[Done] Yunnan data audit complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
