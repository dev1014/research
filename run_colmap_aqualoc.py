#!/usr/bin/env python3
"""COLMAP reconstruction on one AQUALOC harbor sequence (stride-5 frames).

Produces real metrics: registered images, 3D points, mean reprojection error,
mean track length, and ATE/RPE of camera centers vs the dataset ground-truth
COLMAP trajectory (Sim(3)-aligned).
"""
import json
import time
from pathlib import Path

import numpy as np
import pycolmap

ROOT = Path(__file__).parent
DS = ROOT / "datasets" / "aqualoc"
IMAGES = DS / "seq07_stride5"
GT_TRAJ = DS / "gt_traj_seq07.txt"
WORK = ROOT / "results" / "aqualoc_seq07"
DB = WORK / "database.db"
SPARSE = WORK / "sparse"

# AQUALOC harbor camera (from harbor_camera_calib.yaml): pinhole + equidistant
FX, FY, CX, CY = 413.32595366566017, 413.70198739483686, 305.9507483284928, 259.4439948946375
K1, K2, K3, K4 = -0.06125568297136998, -0.003796743395135256, 0.027326634771204592, -0.030296403142887066
W, H = 640, 512


def sim3_align(src, tgt):
    """Umeyama Sim(3): align src->tgt. Returns aligned_src, scale, R, t."""
    n = len(src)
    mu_s, mu_t = src.mean(0), tgt.mean(0)
    sc, tc = src - mu_s, tgt - mu_t
    var_s = (sc ** 2).sum() / n
    U, D, Vt = np.linalg.svd((tc.T @ sc) / n)
    S = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    R = U @ S @ Vt
    scale = float((D * np.diag(S)).sum() / var_s)
    aligned = scale * (R @ sc.T).T + mu_t
    return aligned, scale


def load_gt(path):
    poses = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) < 8:
            continue
        idx = int(float(p[0]))
        name = f"frame{idx:06d}.png"
        poses[name] = np.array([float(p[1]), float(p[2]), float(p[3])])
    return poses


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        DB.unlink()
    log = {"images_dir": str(IMAGES), "num_input_images": len(list(IMAGES.glob("*.png")))}

    t0 = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV_FISHEYE"
    reader.camera_params = f"{FX},{FY},{CX},{CY},{K1},{K2},{K3},{K4}"
    print("[1/3] Feature extraction (CPU SIFT, OPENCV_FISHEYE)", flush=True)
    pycolmap.extract_features(
        database_path=DB, image_path=IMAGES,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        device=pycolmap.Device.cpu,
    )
    log["t_extract_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    print("[2/3] Sequential matching (CPU)", flush=True)
    pycolmap.match_sequential(
        database_path=DB,
        pairing_options=pycolmap.SequentialPairingOptions(overlap=10),
        device=pycolmap.Device.cpu,
    )
    log["t_match_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    print("[3/3] Incremental mapping", flush=True)
    if SPARSE.exists():
        import shutil
        shutil.rmtree(SPARSE)
    SPARSE.mkdir()
    maps = pycolmap.incremental_mapping(database_path=DB, image_path=IMAGES, output_path=SPARSE)
    log["t_mapping_s"] = round(time.time() - t0, 1)
    if not maps:
        log["error"] = "no reconstruction"
        print(json.dumps(log, indent=2))
        return
    rec = maps[max(maps, key=lambda k: maps[k].num_reg_images())]

    log["registered_images"] = rec.num_reg_images()
    log["pct_registered"] = round(100 * rec.num_reg_images() / log["num_input_images"], 1)
    log["points3D"] = rec.num_points3D()
    log["mean_reproj_error_px"] = round(rec.compute_mean_reprojection_error(), 4)
    log["mean_track_length"] = round(rec.compute_mean_track_length(), 3)

    # ATE / RPE vs ground truth (Sim3 aligned camera centers)
    gt = load_gt(GT_TRAJ)
    est = {img.name: img.projection_center() for img in rec.images.values() if img.has_pose}
    common = sorted(set(gt) & set(est))
    log["gt_common_poses"] = len(common)
    if len(common) >= 3:
        E = np.array([est[n] for n in common])
        G = np.array([gt[n] for n in common])
        aligned, scale = sim3_align(E, G)
        ate = np.linalg.norm(aligned - G, axis=1)
        log["sim3_scale"] = round(scale, 4)
        log["ate_rmse_m"] = round(float(np.sqrt((ate ** 2).mean())), 4)
        log["ate_mean_m"] = round(float(ate.mean()), 4)
        log["ate_median_m"] = round(float(np.median(ate)), 4)
        log["ate_std_m"] = round(float(ate.std()), 4)
        # RPE: consecutive relative translation error
        rpe = []
        for i in range(len(common) - 1):
            d_est = np.linalg.norm(aligned[i + 1] - aligned[i])
            d_gt = np.linalg.norm(G[i + 1] - G[i])
            rpe.append(abs(d_est - d_gt))
        log["rpe_mean_m"] = round(float(np.mean(rpe)), 4)
        log["rpe_std_m"] = round(float(np.std(rpe)), 4)

    (WORK / "colmap_metrics.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
