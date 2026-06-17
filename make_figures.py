#!/usr/bin/env python3
"""Generate real result figures from the saved metrics JSONs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
DPI = 300

seqs = ["01", "02", "03", "04", "05", "06", "07"]
data = {s: json.loads((RES / f"aqualoc_seq{s}_gpu" / "metrics.json").read_text()) for s in seqs}
dust_seqs = ["01", "02", "07"]
dust = {s: json.loads((RES / f"aqualoc_seq{s}_gpu" / "dust3r_metrics.json").read_text()) for s in dust_seqs}

# Figure 1: registration % (largest model) and reprojection error per sequence
fig, ax1 = plt.subplots(figsize=(8, 4.5))
x = range(len(seqs))
pct = [data[s]["pct_registered_largest"] for s in seqs]
rep = [data[s]["mean_reproj_error_px"] for s in seqs]
ax1.bar(x, pct, color="#4C72B0", alpha=0.85, label="Largest-model registration (%)")
ax1.set_ylabel("Largest-model registration (%)", color="#4C72B0")
ax1.set_ylim(0, 110)
ax1.set_xticks(list(x))
ax1.set_xticklabels([f"H{s}" for s in seqs])
ax1.set_xlabel("AQUALOC harbor sequence")
ax2 = ax1.twinx()
ax2.plot(x, rep, "o-", color="#C44E52", label="Mean reprojection error (px)")
ax2.set_ylabel("Mean reprojection error (px)", color="#C44E52")
ax2.set_ylim(0, max(rep) * 1.25)
l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="upper center", fontsize=8, framealpha=0.9)
ax1.set_title("COLMAP per-sequence registration and reprojection error (AQUALOC harbor)")
fig.tight_layout()
fig.savefig(FIG / "figure1_colmap_per_sequence.png", dpi=DPI)
plt.close(fig)

# Figure 2: DUSt3R confidence-survival curves for 3 sequences
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = {"01": "#55A868", "02": "#4C72B0", "07": "#8172B3"}
thr_keys = list(next(iter(dust.values()))["conf_sweep"].keys())
thr = [float(t) for t in thr_keys]
for s in dust_seqs:
    sweep = dust[s]["conf_sweep"]
    frac = [sweep[t]["frac"] * 100 for t in thr_keys]
    ax.plot(thr, frac, "o-", color=colors[s], linewidth=1.8,
            label=f"H{s} (max conf {dust[s]['conf_max']:.2f})")
ax.axvline(3.0, color="#C44E52", linestyle="--", label="DUSt3R default threshold (3.0)")
ax.set_xlabel("Confidence threshold")
ax.set_ylabel("Dense points surviving (%)")
ax.set_title("DUSt3R confidence collapse on underwater imagery (3 harbor windows)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "figure2_dust3r_confidence.png", dpi=DPI)
plt.close(fig)

# Figure 3: COLMAP ATE per sequence (log) with DUSt3R overlaid for shared sequences
fig, ax = plt.subplots(figsize=(8, 4.5))
ate = [data[s]["ate_rmse_m"] * 100 for s in seqs]
ax.bar([f"H{s}" for s in seqs], ate, color="#4C72B0", alpha=0.85, label="COLMAP (full sequence)")
dx = [seqs.index(s) for s in dust_seqs]
dy = [dust[s]["ate_rmse_m"] * 100 for s in dust_seqs]
ax.plot(dx, dy, "D", color="#C44E52", markersize=8, label="DUSt3R (20-frame window)")
ax.set_yscale("log")
ax.set_ylabel("ATE RMSE (cm, log scale)")
ax.set_xlabel("AQUALOC harbor sequence")
ax.set_title("Trajectory error vs ground truth (Sim(3)-aligned): COLMAP vs DUSt3R")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "figure3_ate_per_sequence.png", dpi=DPI)
plt.close(fig)

print("wrote:", [p.name for p in sorted(FIG.glob('*.png'))])
