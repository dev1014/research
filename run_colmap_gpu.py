#!/usr/bin/env python3
"""GPU COLMAP on one AQUALOC harbor sequence (native CUDA binary + pycolmap scoring).

Usage: run_colmap_gpu.py <seq_number e.g. 07> [stride]
Extracts stride-N frames, runs feature_extractor/sequential_matcher/mapper on GPU,
then scores the largest model: registered images, points, reproj error, track
length, #submodels, and ATE/RPE vs ground-truth (Sim3-aligned).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap

ROOT = Path(__file__).parent
DS = ROOT / "datasets" / "aqualoc"

SEQ = sys.argv[1] if len(sys.argv) > 1 else "07"
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 5

CAM_PARAMS = ("413.32595366566017,413.70198739483686,305.9507483284928,259.4439948946375,"
              "-0.06125568297136998,-0.003796743395135256,0.027326634771204592,-0.030296403142887066")


def sim3_align(src, tgt):
    n = len(src)
    mu_s, mu_t = src.mean(0), tgt.mean(0)
    sc, tc = src - mu_s, tgt - mu_t
    var_s = (sc ** 2).sum() / n
    U, D, Vt = np.linalg.svd((tc.T @ sc) / n)
    S = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    R = U @ S @ Vt
    scale = float((D * np.diag(S)).sum() / var_s)
    return scale * (R @ sc.T).T + mu_t, scale


def load_gt(path):
    poses = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 8:
            poses[f"frame{int(float(p[0])):06d}.png"] = np.array([float(p[1]), float(p[2]), float(p[3])])
    return poses


def ensure_images():
    tar = DS / f"harbor_seq{SEQ}_raw.tar.gz"
    raw_dir = DS / "raw_data" / f"harbor_images_sequence_{SEQ}"
    if not raw_dir.exists():
        subprocess.run(["tar", "-xzf", str(tar)], cwd=DS, check=True)
    out = DS / f"seq{SEQ}_stride{STRIDE}"
    if out.exists() and len(list(out.glob("*.png"))) > 0:
        return out
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted(raw_dir.glob("frame*.png"))
    for i, f in enumerate(frames):
        if i % STRIDE == 0:
            (out / f.name).write_bytes(f.read_bytes())
    return out


def main():
    images = ensure_images()
    work = ROOT / "results" / f"aqualoc_seq{SEQ}_gpu"
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    sparse = work / "sparse"
    if db.exists():
        db.unlink()
    sparse.mkdir(exist_ok=True)
    n_in = len(list(images.glob("*.png")))
    log = {"sequence": SEQ, "stride": STRIDE, "num_input_images": n_in}

    t0 = time.time()
    subprocess.run(["colmap", "feature_extractor", "--database_path", str(db),
                    "--image_path", str(images), "--ImageReader.camera_model", "OPENCV_FISHEYE",
                    "--ImageReader.single_camera", "1", "--ImageReader.camera_params", CAM_PARAMS,
                    "--SiftExtraction.use_gpu", "1"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log["t_extract_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    subprocess.run(["colmap", "sequential_matcher", "--database_path", str(db),
                    "--SiftMatching.use_gpu", "1", "--SequentialMatching.overlap", "10"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log["t_match_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    subprocess.run(["colmap", "mapper", "--database_path", str(db), "--image_path", str(images),
                    "--output_path", str(sparse)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log["t_mapping_s"] = round(time.time() - t0, 1)

    submodels = sorted([d for d in sparse.iterdir() if d.is_dir()])
    log["num_submodels"] = len(submodels)
    if not submodels:
        log["error"] = "no reconstruction"
        (work / "metrics.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(log, indent=2)); return

    recs = [pycolmap.Reconstruction(str(d)) for d in submodels]
    rec = max(recs, key=lambda r: r.num_reg_images())
    log["total_registered_all_models"] = sum(r.num_reg_images() for r in recs)
    log["registered_images"] = rec.num_reg_images()
    log["pct_registered_largest"] = round(100 * rec.num_reg_images() / n_in, 1)
    log["points3D"] = rec.num_points3D()
    log["mean_reproj_error_px"] = round(rec.compute_mean_reprojection_error(), 4)
    log["mean_track_length"] = round(rec.compute_mean_track_length(), 3)

    gt = load_gt(DS / f"gt_traj_seq{SEQ}.txt")
    est = {img.name: img.projection_center() for img in rec.images.values() if img.has_pose}
    common = sorted(set(gt) & set(est))
    log["gt_common_poses"] = len(common)
    if len(common) >= 3:
        E = np.array([est[n] for n in common]); G = np.array([gt[n] for n in common])
        aligned, scale = sim3_align(E, G)
        ate = np.linalg.norm(aligned - G, axis=1)
        log["sim3_scale"] = round(scale, 4)
        log["ate_rmse_m"] = round(float(np.sqrt((ate ** 2).mean())), 4)
        log["ate_std_m"] = round(float(ate.std()), 4)
        rpe = [abs(np.linalg.norm(aligned[i + 1] - aligned[i]) - np.linalg.norm(G[i + 1] - G[i]))
               for i in range(len(common) - 1)]
        log["rpe_mean_m"] = round(float(np.mean(rpe)), 4)

    (work / "metrics.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
