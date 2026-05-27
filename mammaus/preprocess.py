"""DICOM ultrasound video preprocessing: frame extraction to PNG."""

import sys
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


def find_dicom_files(input_path: str) -> list[Path]:  # pragma: no cover
    """Recursively find all DICOM files in a directory or return a single file.

    Excluded from coverage: requires real DICOM files on disk.
    """
    p = Path(input_path)
    if p.is_file():
        return [p]
    dcm_files = []
    for f in sorted(p.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() in (".dll", ".exe", ".command", ".bat", ".sh", ".xml", ".txt", ".json", ".png", ".jpg"):
            continue
        if f.suffix.lower() in (".dcm", ".dicom"):
            dcm_files.append(f)
        else:
            try:
                pydicom.dcmread(str(f), stop_before_pixels=True)
                dcm_files.append(f)
            except Exception:
                continue
    return dcm_files

def apply_windowing(pixel_array: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """Apply DICOM windowing and normalize pixel values to 0-255 uint8."""
    arr = pixel_array.astype(np.float32)
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        wc = float(wc[0]) if isinstance(wc, pydicom.multival.MultiValue) else float(wc)
        ww = float(ww[0]) if isinstance(ww, pydicom.multival.MultiValue) else float(ww)
        arr = np.clip(arr, wc - ww / 2, wc + ww / 2)
    pmin, pmax = arr.min(), arr.max()
    if pmax > pmin:
        arr = (arr - pmin) / (pmax - pmin) * 255.0
    return arr.astype(np.uint8)

def crop_ultrasound_region(img_array: np.ndarray) -> np.ndarray:
    """Crop black borders around the ultrasound region of interest."""
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2).astype(np.uint8)
    else:
        gray = img_array
    threshold = 10
    mask = gray > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img_array
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    margin = 5
    rmin = min(rmin + margin, rmax)
    rmax = max(rmax - margin, rmin)
    cmin = min(cmin + margin, cmax)
    cmax = max(cmax - margin, cmin)
    return img_array[rmin:rmax + 1, cmin:cmax + 1]

def get_series_info(ds: pydicom.Dataset) -> str:
    """Extract a clean series identifier from DICOM metadata."""
    desc = getattr(ds, "SeriesDescription", "").strip()
    series_num = getattr(ds, "SeriesNumber", "")
    if desc:
        desc = desc.replace(" ", "_").replace("/", "-")
        return f"{series_num}_{desc}" if series_num else desc
    return str(series_num) if series_num else "unknown"

def get_patient_id(ds: pydicom.Dataset) -> str:
    """Extract a filesystem-safe patient identifier from DICOM metadata."""
    name = str(getattr(ds, "PatientName", "")).replace("^", "_").replace(" ", "_")
    pid = str(getattr(ds, "PatientID", ""))
    if name and name != "":
        return name
    return pid if pid else "patient"

def process_dicom(dcm_path: Path, output_base: Path) -> int:  # pragma: no cover
    """Extract frames from a DICOM file, process and save as PNGs.

    Excluded from coverage: requires real DICOM files with pixel data.
    """
    ds = pydicom.dcmread(str(dcm_path))
    if not hasattr(ds, "pixel_array"):
        print(f"  [SKIP] No pixels: {dcm_path.name}")
        return 0
    pixel_data = ds.pixel_array
    series = get_series_info(ds)
    patient = get_patient_id(ds)
    out_dir = output_base / patient / series
    out_dir.mkdir(parents=True, exist_ok=True)
    if pixel_data.ndim == 2:
        frames = [pixel_data]
    elif pixel_data.ndim == 3 and pixel_data.shape[2] in (3, 4):
        frames = [pixel_data]
    elif pixel_data.ndim == 3:
        frames = [pixel_data[i] for i in range(pixel_data.shape[0])]
    elif pixel_data.ndim == 4:
        frames = [pixel_data[i] for i in range(pixel_data.shape[0])]
    else:
        print(f"  [SKIP] Unexpected pixel format {pixel_data.shape}: {dcm_path.name}")
        return 0
    count = 0
    for i, frame in enumerate(frames):
        processed = apply_windowing(frame, ds)
        cropped = crop_ultrasound_region(processed)
        img = Image.fromarray(cropped)
        if img.mode != "RGB":
            img = img.convert("RGB")
        fname = f"frame_{i:03d}.png"
        img.save(out_dir / fname)
        count += 1
    return count

def preprocess_cli() -> None:  # pragma: no cover
    """CLI entry point: extract frames from DICOM ultrasound files to PNG.

    Excluded from coverage: orchestration over real DICOM I/O.
    """
    import argparse
    parser = argparse.ArgumentParser(description="DICOM preprocessing → PNG (ultrasound video frame extraction)")
    parser.add_argument("input_path", help="Folder or DICOM file to process")
    parser.add_argument("--output", default="preprocessed", help="Output folder (default: preprocessed)")
    args = parser.parse_args()
    dcm_files = find_dicom_files(args.input_path)
    if not dcm_files:
        print(f"No DICOM file found in: {args.input_path}")
        sys.exit(1)
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"{'=' * 60}")
    print("  DICOM PREPROCESSING → PNG")
    print(f"  Source: {args.input_path}")
    print(f"  Destination: {output_base}")
    print(f"  DICOM files found: {len(dcm_files)}")
    print(f"{'=' * 60}\n")
    total_images = 0
    for dcm_path in dcm_files:
        series_info = "?"
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            series_info = get_series_info(ds)
        except Exception:
            pass
        try:
            n = process_dicom(dcm_path, output_base)
            print(f"  [{series_info:>10s}]  {n} frame(s) → {output_base.name}/")
            total_images += n
        except Exception as e:
            print(f"  [{series_info:>10s}]  ERROR: {e}")
    print(f"\n{'=' * 60}")
    print(f"  Total: {total_images} PNG images generated")
    print(f"  Folder: {output_base.resolve()}")
    print(f"{'=' * 60}")
