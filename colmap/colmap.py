#!/usr/bin/env python3
"""
Full COLMAP 3D reconstruction pipeline with comparison against reference.

Setup:  python3 -m venv .venv && source .venv/bin/activate && pip install pycolmap numpy
Run:    source .venv/bin/activate && python colmap.py
"""

import shutil
import time
from pathlib import Path

import numpy as np
import pycolmap

ROOT     = Path(__file__).parent                          # colmap/
DATASET  = ROOT.parent / "datasets" / "gerrard-hall"
IMAGES   = DATASET / "images"
REF      = DATASET / "sparse"         # reference sparse reconstruction (text)
OUTPUT   = ROOT                        # results land directly in colmap/
DB       = OUTPUT / "database.db"
SPARSE   = OUTPUT / "sparse"          # new sparse reconstruction
DENSE    = OUTPUT / "dense"           # dense workspace + fused.ply


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(n: int, total: int, msg: str) -> float:
    print(f"\n[{n}/{total}] {msg}", flush=True)
    return time.time()


def _done(t0: float) -> None:
    print(f"    done in {time.time() - t0:.1f}s", flush=True)


def _sim3_align(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, float]:
    """Umeyama Sim(3): align src onto tgt. Returns (aligned_src, scale)."""
    n = len(src)
    mu_s, mu_t = src.mean(0), tgt.mean(0)
    sc, tc = src - mu_s, tgt - mu_t
    var_s = (sc ** 2).sum() / n
    U, D, Vt = np.linalg.svd((tc.T @ sc) / n)
    det_sign = float(np.linalg.det(U @ Vt))
    S = np.diag([1.0, 1.0, det_sign])
    R = U @ S @ Vt
    scale = float((D * np.diag(S)).sum() / var_s)
    return scale * (R @ sc.T).T + mu_t, scale


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> pycolmap.Reconstruction:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()

    # 1. Feature extraction ─ GPU SIFT
    t0 = _step(1, 5, "Feature extraction  (GPU SIFT)")
    pycolmap.extract_features(
        database_path=DB,
        image_path=IMAGES,
        device=pycolmap.Device.cuda,
    )
    _done(t0)

    # 2. Exhaustive feature matching ─ GPU
    t0 = _step(2, 5, "Exhaustive feature matching  (GPU)")
    pycolmap.match_exhaustive(
        database_path=DB,
        matching_options=pycolmap.FeatureMatchingOptions(use_gpu=True, gpu_index=-1),
    )
    _done(t0)

    # 3. Incremental SfM
    t0 = _step(3, 5, "Incremental SfM mapping")
    if SPARSE.exists():
        shutil.rmtree(SPARSE)
    SPARSE.mkdir()
    maps = pycolmap.incremental_mapping(
        database_path=DB,
        image_path=IMAGES,
        output_path=SPARSE,
    )
    if not maps:
        raise RuntimeError("Incremental mapping produced no reconstruction.")
    recon = maps[0]
    _done(t0)
    print(f"    {recon.num_reg_images()} images registered, {recon.num_points3D()} 3D points")

    # 4. Undistort images for dense workspace
    t0 = _step(4, 5, "Undistorting images for dense reconstruction")
    if DENSE.exists():
        shutil.rmtree(DENSE)
    DENSE.mkdir()
    pycolmap.undistort_images(
        output_path=DENSE,
        input_path=SPARSE / "0",
        image_path=IMAGES,
    )
    _done(t0)

    # 5. Patch match stereo + stereo fusion ─ GPU
    t0 = _step(5, 5, "Dense reconstruction  (patch match stereo + fusion, GPU)")
    pycolmap.patch_match_stereo(DENSE)
    fused = DENSE / "fused.ply"
    pycolmap.stereo_fusion(str(fused), str(DENSE))
    _done(t0)
    print(f"    Dense point cloud → {fused.relative_to(ROOT.parent)}")

    return recon


# ---------------------------------------------------------------------------
# Comparison against reference
# ---------------------------------------------------------------------------

def compare(new_recon: pycolmap.Reconstruction) -> None:
    ref = pycolmap.Reconstruction()
    ref.read_text(str(REF))

    W = 62
    print("\n" + "=" * W)
    print("  RECONSTRUCTION COMPARISON")
    print("=" * W)

    # --- Scalar stats ---
    def stat_row(label, r_val, n_val, thresh_pct, fmt="{:.2f}"):
        diff = abs(r_val - n_val) / max(abs(r_val), 1e-9) * 100
        status = "OK" if diff < thresh_pct else f"~{diff:.0f}%"
        print(f"  {label:<26} {fmt.format(r_val):>11} {fmt.format(n_val):>11}  {status}")

    print(f"  {'Metric':<26} {'Reference':>11} {'New':>11}  Status")
    print("  " + "-" * (W - 2))
    stat_row("Registered images",   ref.num_reg_images(),                new_recon.num_reg_images(),                5,  "{:.0f}")
    stat_row("3D points",           ref.num_points3D(),                  new_recon.num_points3D(),                  15, "{:.0f}")
    stat_row("Mean reproj. error",  ref.compute_mean_reprojection_error(), new_recon.compute_mean_reprojection_error(), 20)
    stat_row("Mean track length",   ref.compute_mean_track_length(),     new_recon.compute_mean_track_length(),     15)

    # --- Camera pose alignment via Sim(3) ---
    ref_by_name = {img.name: img for img in ref.images.values()}
    new_by_name = {img.name: img for img in new_recon.images.values()}
    common = sorted(set(ref_by_name) & set(new_by_name))

    print(f"\n  Camera pose alignment  (Sim3, {len(common)} common images)")
    print("  " + "-" * (W - 2))

    pose_ok = False
    rel_mean = float("nan")

    if len(common) < 3:
        print("  Not enough common images to align.")
    else:
        ref_c = np.array([ref_by_name[n].projection_center() for n in common])
        new_c = np.array([new_by_name[n].projection_center() for n in common])
        aligned, scale = _sim3_align(new_c, ref_c)
        errs = np.linalg.norm(aligned - ref_c, axis=1)
        scene_scale = np.linalg.norm(ref_c - ref_c.mean(0), axis=1).mean()
        rel = errs / scene_scale * 100
        rel_mean = rel.mean()

        scale_ok = 0.9 < scale < 1.1
        pose_ok  = scale_ok and rel_mean < 2.0

        print(f"  {'Scale factor (new/ref):':<28} {scale:>9.4f}  {'OK' if scale_ok else 'WARNING'}")
        print(f"  {'Mean pose error:':<28} {rel_mean:>8.2f}%")
        print(f"  {'Median pose error:':<28} {np.median(rel):>8.2f}%")
        print(f"  {'Max pose error:':<28} {rel.max():>8.2f}%")

    # --- Verdict ---
    imgs_ok = abs(new_recon.num_reg_images() - ref.num_reg_images()) / max(ref.num_reg_images(), 1) < 0.05
    pts_ok  = abs(new_recon.num_points3D()   - ref.num_points3D())   / max(ref.num_points3D(),  1) < 0.15

    print("\n  " + "-" * (W - 2))
    if imgs_ok and pts_ok and pose_ok:
        print("  RESULT:  PASS  ✓  New reconstruction matches the reference.")
    else:
        issues = []
        if not imgs_ok: issues.append("image count")
        if not pts_ok:  issues.append("point count")
        if not pose_ok: issues.append("camera poses")
        print(f"  RESULT:  REVIEW — differences in: {', '.join(issues)}")
    print("=" * W)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_total = time.time()
    print("=" * 62)
    print("  COLMAP Full Reconstruction Pipeline")
    print(f"  Dataset : {DATASET.relative_to(ROOT.parent)}")
    print(f"  Output  : {OUTPUT.relative_to(ROOT.parent)}")
    print("=" * 62)

    new_recon = run_pipeline()
    compare(new_recon)

    print(f"\nTotal time: {(time.time() - t_total) / 60:.1f} min\n")
