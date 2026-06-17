#!/usr/bin/env python3
"""DUSt3R reconstruction on an AQUALOC harbor seq07 window (paper defaults).

Reports: registered (aligned) images, dense points after confidence filtering
(thr 3.0), ATE/RPE of recovered camera centers vs ground-truth (Sim3-aligned),
runtime, peak VRAM.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "dust3r"))

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.image import load_images

DS = ROOT / "datasets" / "aqualoc"
IMAGES = DS / "seq07_stride5"
GT_TRAJ = DS / "gt_traj_seq07.txt"
CKPT = ROOT / "dust3r" / "checkpoints" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
WORK = ROOT / "results" / "aqualoc_seq07"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
CONF_THR = 3.0
DEVICE = "cuda"


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
            poses[f"frame{int(float(p[0])):06d}.png"] = np.array(
                [float(p[1]), float(p[2]), float(p[3])])
    return poses


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    files = sorted(IMAGES.glob("*.png"))[:N]
    names = [f.name for f in files]
    log = {"num_images": len(files), "conf_thr": CONF_THR}

    torch.cuda.reset_peak_memory_stats()
    model = AsymmetricCroCo3DStereo.from_pretrained(str(CKPT)).to(DEVICE)

    t0 = time.time()
    imgs = load_images([str(f) for f in files], size=512)
    pairs = make_pairs(imgs, scene_graph="swin", prefilter=None, symmetrize=True)
    log["num_pairs"] = len(pairs)
    output = inference(pairs, model, DEVICE, batch_size=1, verbose=False)
    model = model.cpu()
    del model
    torch.cuda.empty_cache()

    scene = global_aligner(output, device=DEVICE, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)
    log["t_infer_align_s"] = round(time.time() - t0, 1)
    log["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    poses = scene.get_im_poses().detach().cpu().numpy()      # cam->world 4x4
    pts3d = scene.get_pts3d()
    confs = scene.get_conf()
    all_conf = np.concatenate([c.detach().cpu().numpy().ravel() for c in confs])
    n_pts = int((all_conf > CONF_THR).sum())
    log["dense_points_conf_filtered"] = n_pts
    log["dense_points_total"] = int(all_conf.size)
    log["conf_min"] = round(float(all_conf.min()), 3)
    log["conf_mean"] = round(float(all_conf.mean()), 3)
    log["conf_max"] = round(float(all_conf.max()), 3)
    log["conf_p90"] = round(float(np.percentile(all_conf, 90)), 3)
    log["frac_conf_gt_1_1"] = round(float((all_conf > 1.1).mean()), 3)

    centers = poses[:, :3, 3]
    gt = load_gt(GT_TRAJ)
    common_idx = [i for i, nm in enumerate(names) if nm in gt]
    log["gt_common_poses"] = len(common_idx)
    if len(common_idx) >= 3:
        E = centers[common_idx]
        G = np.array([gt[names[i]] for i in common_idx])
        aligned, scale = sim3_align(E, G)
        ate = np.linalg.norm(aligned - G, axis=1)
        log["sim3_scale"] = round(scale, 4)
        log["ate_rmse_m"] = round(float(np.sqrt((ate ** 2).mean())), 4)
        log["ate_mean_m"] = round(float(ate.mean()), 4)
        log["ate_median_m"] = round(float(np.median(ate)), 4)
        log["ate_std_m"] = round(float(ate.std()), 4)
        rpe = [abs(np.linalg.norm(aligned[i + 1] - aligned[i]) - np.linalg.norm(G[i + 1] - G[i]))
               for i in range(len(common_idx) - 1)]
        log["rpe_mean_m"] = round(float(np.mean(rpe)), 4)
        log["rpe_std_m"] = round(float(np.std(rpe)), 4)

    (WORK / "dust3r_metrics.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
