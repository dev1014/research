# Comparative Evaluation of COLMAP and DUSt3R for Underwater Pose and Sparse Reconstruction: A Reproducible Benchmark on the AQUALOC Harbor Sequences

**Author:** Dev Narang

**Affiliation:** Georgetown Day School, 12309 Potomac Hunt Rd, Gaithersburg, MD 20878, USA

**Corresponding author:** Dev Narang, devnarang083@gmail.com, 301-792-0099

**Manuscript type:** Technical note

## Abstract

**Background/Objective:** Underwater pose and sparse 3D reconstruction are degraded by light absorption, backscatter, turbidity, and refraction through camera housings. This technical note compares a classical feature-based pipeline (COLMAP) and a learning-based dense method (DUSt3R) on real underwater imagery, reporting exactly what was run so the pose, registration, sparse-point, and confidence-filtering results can be reproduced. **Methods:** All experiments ran on a single RTX 4070 Super (12 GB) workstation. COLMAP was run on all seven AQUALOC harbor sequences using the dataset's fisheye calibration, and its recovered camera centers were compared against the dataset's SfM-derived ground-truth trajectories after Sim(3) alignment. DUSt3R was run with its official 512 px model and default settings on deterministic 20-frame windows from three sequences; COLMAP was additionally run on those same windows for a matched comparison, and a confidence-threshold sweep was performed. **Results:** COLMAP was accurate where it converged (median reprojection error 0.35 px; median trajectory error 3.8 cm), but its registration frequently fragmented into disconnected sub-models and H05 was a clear full-sequence failure case. On the matched 20-frame windows the gap was narrower than the full-sequence numbers suggest: COLMAP failed to register one H07 window (2 of 20 frames) on which DUSt3R recovered all poses. The most consistent finding is a confidence collapse: DUSt3R's maximum per-pixel confidence underwater (2.68–3.17) barely reached its own default filtering threshold of 3.0, so default filtering discarded 99.5–100% of predicted points, and a threshold near 1.5 was needed to retain a usable fraction. DUSt3R also showed a large scale ambiguity (Sim(3) scale 3.2–6.3). **Conclusions:** COLMAP is the more accurate and interpretable method when it converges, but underwater conditions make it fragment and occasionally fail; DUSt3R recovered all poses on the three short windows tested, but it remained up-to-scale and exhibited a pronounced confidence collapse underwater. As DUSt3R was evaluated on only three 20-frame windows, the trajectory comparison should be treated as indicative rather than definitive. Neither method is universally superior, and their failure modes differ.

**Keywords:** underwater 3D reconstruction, Structure-from-Motion, COLMAP, DUSt3R, multi-view stereo, learned 3D vision, AQUALOC, absolute trajectory error, domain shift, confidence filtering, photogrammetry, visual localization

## 1. Introduction

Three-dimensional reconstruction from underwater images supports cave mapping, marine archaeology, robot navigation, and coastal-ecosystem and aquaculture monitoring. A concrete current example is aquaculture site assessment, where mapping seabed terrain and structure helps growers (for instance, oyster farmers) judge whether a candidate site is viable; this is exactly the kind of task that needs a low-cost, reproducible reconstruction pipeline that runs on commodity hardware, which is the practical motivation for the comparison reported here. Unlike terrestrial photogrammetry, underwater imagery is affected by wavelength-dependent attenuation, scattering, artificial lighting gradients, moving particles, dynamic marine life, and refraction through flat or dome camera ports. These effects reduce keypoint repeatability and violate the simple pinhole assumptions used by many reconstruction systems.

This paper compares two representative methods. COLMAP is a mature Structure-from-Motion (SfM) and Multi-View Stereo (MVS) system that estimates camera poses, sparse geometry, and dense reconstructions from image correspondences (1, 2). DUSt3R, introduced at CVPR 2024, is a transformer-based method that directly predicts pairwise 3D pointmaps and can recover dense geometry, correspondences, camera parameters, and poses from uncalibrated image collections (3). The present study evaluates registration, sparse reconstruction statistics, trajectory accuracy, and DUSt3R confidence filtering; it does not validate dense 3D geometry.

Many published comparisons of such methods underwater report only aggregate or qualitative outcomes. The objective of this work is narrower and verifiable: to run both methods on a real, publicly available underwater dataset on a single consumer workstation, to compare the recovered camera trajectories against the available reference trajectory, and to report the exact configuration, the per-sequence results, and the failure cases. The research question is:

> When run under fully specified conditions on real underwater data — both on full sequences and on identical short windows — how do classical SfM/MVS (COLMAP) and learned dense reconstruction (DUSt3R) compare in registration completeness, sparse-reconstruction quality, trajectory accuracy, and practical robustness?

To keep the comparison fair despite the two methods' different memory footprints, COLMAP is reported both on the full sequences and on the exact same 20-frame windows that DUSt3R is limited to (Section 4.2); trajectory accuracy is assessed only on the recovered camera centers, and dense-geometry quality is explicitly left to future work (Section 7), so no claim is made here about validated dense geometry.

**Scope.** This study evaluates the seven AQUALOC harbor sequences. Two additional underwater datasets, CIRS Underwater Caves and FLSea, are described in Section 2 because they motivate the problem and are the intended targets of follow-up work; they were **not** evaluated here. Limiting the scope to one dataset keeps every reported number traceable to an executed experiment.

**Plain-English overview of the key concepts.** For readers new to 3D vision, the core ideas used throughout this paper are:

- **COLMAP** is the classical "stitch photos into 3D" approach. It finds distinctive points (features) in each image, matches the same point across overlapping images, and from those matches solves for where each camera was and where the 3D points are. When it works, the answer is metric (real-world scale) and fully auditable; when too few points match, it can break the scene into disconnected pieces or fail.
- **DUSt3R** is a neural network that, given a pair of images, directly predicts a 3D point for every pixel, without needing a calibrated camera. Predictions from many pairs are then stitched into one coordinate frame ("global alignment"). It is more tolerant of weak texture but inherits whatever biases its training data carried.
- **Camera pose / trajectory** is the path the camera took — its position and orientation over time. Comparing the estimated path to a known reference path is how we judge accuracy.
- **ATE (Absolute Trajectory Error)** measures, after best-fit alignment, how far the estimated camera positions are from the reference positions on average (reported here in meters/centimeters). **RPE (Relative Pose Error)** measures how well short, local motions between consecutive frames are recovered, so it captures drift rather than global placement.
- **Sim(3) alignment** is the best-fit rotation, translation, and single uniform scale that overlays the estimate onto the reference. It is required because a single moving camera recovers shape only up to an unknown overall scale; the fitted **scale factor** itself tells us how far off the reconstruction's size is (a value near 1 means roughly correct scale).
- **Confidence threshold** is DUSt3R's own per-pixel reliability score: each predicted 3D point comes with a confidence, and points below a chosen threshold are discarded. A key finding below is that the model's default threshold throws away almost everything underwater.

## 2. Datasets

### 2.1 AQUALOC (evaluated in this work)

AQUALOC was designed for underwater visual-inertial-pressure localization (4). It includes 17 sequences collected with ROV-mounted monocular monochrome cameras, MEMS IMUs, pressure sensors, and embedded Jetson TX2 computers, covering a shallow harbor (~4 m), a first archaeological site (~270 m), and a second archaeological site (~380 m). For every sequence the authors provide a ground-truth camera trajectory computed offline with a full-batch bundle-adjustment SfM pipeline, which enables quantitative trajectory evaluation. This study uses the seven harbor sequences, whose key quantities are summarized below.

| Item | Value |
|---|---:|
| Harbor sequences evaluated | 7 |
| Camera | monocular monochrome, 640 x 512 |
| Camera model | pinhole + equidistant (fisheye) distortion |
| Camera rate | 20 Hz |
| IMU rate | 200 Hz |
| Harbor depth | approx. 4 m |
| Ground truth | per-sequence offline COLMAP/SfM trajectory |

These sequences contain turbidity, backscatter, repetitive texture, sandy clouds, dynamic animals, and robot-arm occlusions, making them a realistic stress test for image-based reconstruction.

### 2.2 CIRS Underwater Caves (motivation / future work)

The CIRS dataset (5) was collected inside an underwater cave complex with an AUV testbed (imaging sonars, Doppler velocity log, IMUs, pressure sensor, and a vertically mounted camera). Its dark, low-texture cave imagery and added sonar/navigation data make it a hard, multimodal target for future evaluation; it was not processed here.

### 2.3 FLSea (motivation / future work)

FLSea (6) is a forward-looking underwater dataset (8 visual-inertial and 5 stereo sequences) that provides dense ground-truth depth maps validated against known-size objects. Those depth maps make it the natural dataset for validating DUSt3R's predicted depth and the primary target of follow-up work; it was not processed here.

## 3. Methods Compared

### 3.1 COLMAP

COLMAP is an end-to-end SfM and MVS pipeline that detects local features, matches images, estimates camera poses through incremental SfM, triangulates sparse 3D points, and optionally performs dense MVS (1, 2). Its advantages are interpretability, mature bundle adjustment, strong camera modeling, and well-established quality metrics (registered images, sparse-point count, track length, reprojection error). For underwater imagery its main weakness is that standard pinhole modeling does not explicitly represent refraction through housings; recent work on Refractive COLMAP shows that approximating refraction with pinhole intrinsics can introduce biased or curved geometry, especially with flat-port cameras (7).

### 3.2 DUSt3R

DUSt3R takes image pairs and predicts 3D pointmaps with per-pixel confidence maps; for multiple images, pairwise pointmaps are globally aligned into a common frame (3). It is attractive underwater because it operates without known calibration or poses, and its learned priors may help when feature matching is weak. However, its training mixture is not underwater-specific, so its behavior under this domain shift must be measured empirically.

## 4. Experimental Setup

All experiments were conducted by the author on a single workstation:

| Hardware/software | Specification |
|---|---|
| GPU | NVIDIA RTX 4070 Super, 12 GB |
| CPU | 16-core |
| Storage | 2 TB NVMe SSD |
| COLMAP | 3.11.1 (native build, CUDA) |
| DUSt3R | official release, checkpoint `DUSt3R_ViTLarge_BaseDecoder_512_dpt` |
| PyTorch | 2.5.1 + CUDA 12.1 |

### 4.1 COLMAP configuration

For each harbor sequence, images were sampled at a fixed stride (one in five frames for the smaller sequences; one in ten for the two largest sequences, which did not complete incremental mapping within a practical time budget at stride five — itself reported in Section 5). The dataset's published fisheye intrinsics were supplied to COLMAP using the `OPENCV_FISHEYE` camera model (the four intrinsics plus the four equidistant-distortion coefficients) with a single shared camera, so no intrinsics were invented.

The remaining settings were COLMAP 3.11.1 defaults, recorded here for reproducibility. **Feature extraction:** GPU SIFT with the default maximum image size of 3200 px and up to 8192 features per image (the AQUALOC frames are 640 × 512, so the size cap never binds), default first octave −1, edge threshold 10, and peak threshold 0.0067. **Matching:** GPU `sequential_matcher` with an overlap window of 10 (each frame matched to its 10 neighbors on either side), suited to the video-rate acquisition; vocabulary-tree loop detection was left at its default (disabled), so no loop closures across distant parts of a trajectory were added. **Mapping:** the default incremental `mapper`. Following COLMAP's defaults, the supplied intrinsics were used as initialization and then self-calibrated during bundle adjustment — focal length and the distortion (extra) parameters were refined while the principal point was held fixed (`ba_refine_focal_length` and `ba_refine_extra_params` true, `ba_refine_principal_point` false). Reported quantities are the native COLMAP outputs: number of reconstructed sub-models, registered images in the largest connected model, total registered images across all sub-models, sparse 3D points, mean reprojection error, and mean track length.

### 4.2 DUSt3R configuration

DUSt3R was run with the official default inference settings: each image rescaled so its longer side measured 512 px, pairwise inference with a sliding-window scene graph, and global alignment with the point-cloud optimizer for 300 iterations using a cosine schedule and initial learning rate 0.01. Because the 12 GB GPU could not hold the global-alignment state for a full sequence (an attempt at 40 images exhausted memory), DUSt3R was evaluated on contiguous 20-frame windows, each consisting of frames that all have ground-truth poses, taken from three sequences (H01, H02, H07) so that the pose-accuracy and confidence findings rest on more than one segment. The three windows were selected deterministically as the first 20 sorted PNG files in `datasets/aqualoc/seq01_stride5`, `datasets/aqualoc/seq02_stride5`, and `datasets/aqualoc/seq07_stride5`; those stride-5 directories are generated by `run_colmap_gpu.py` by copying every fifth sorted raw frame. The same selection is implemented in `run_dust3r_sweep.py` as `sorted(images.glob("*.png"))[:N]` with `N = 20`. The default per-pixel confidence threshold is 3.0; a sweep over thresholds was performed on each window to characterize how many points survive filtering.

**Matched-window COLMAP.** To make the COLMAP–DUSt3R comparison fair despite this memory limit, COLMAP was additionally run on the *exact same* 20-frame windows (the same first-20-frame selections used for DUSt3R), with the identical pipeline and intrinsics described in Section 4.1. This selection is implemented in `run_colmap_window.py` using the same sorted-file expression as the DUSt3R script. The matched-window run yields a like-for-like comparison on identical input, alongside the full-sequence COLMAP results; the matched-window numbers are reported in Section 5.2 (Table 2b).

### 4.3 Evaluation metrics

Reconstruction quality was measured by registration completeness, sparse-point count, mean reprojection error, and mean track length. Trajectory accuracy was measured by Absolute Trajectory Error (ATE) and a translation Relative Pose Error (RPE) of the recovered camera centers against the dataset's ground-truth trajectory. Because monocular reconstruction is recovered only up to a similarity transform, estimated camera centers were aligned to ground truth with a Umeyama Sim(3) transform (8) before computing ATE/RPE; the recovered scale factor is reported as a direct measure of scale ambiguity. Runtime and peak GPU memory were taken from process and CUDA logs. AQUALOC's reference trajectories were themselves computed offline with an SfM pipeline, so the ATE/RPE comparison is not an external metric ground truth and may favor COLMAP; this bias does not affect registration completeness or DUSt3R confidence-threshold statistics.

## 5. Experimental Results

### 5.1 COLMAP across the seven harbor sequences

Tables 1a and 1b report the per-sequence COLMAP results. Across all seven sequences COLMAP reconstructed 334,906 sparse 3D points (largest models) with a median mean-reprojection-error of 0.35 px (mean 0.45 px, range 0.29–1.09 px). Trajectory accuracy was excellent on five sequences (ATE RMSE 1.0–5.3 cm) but poor on two (H03: 24.5 cm; H05: 1.87 m), giving a median ATE of 3.8 cm.

The most important qualitative finding is that **registration was frequently fragmented**: COLMAP often split a sequence into several disconnected sub-models rather than one continuous reconstruction. The largest connected model covered only 33–50% of frames on five of the seven sequences, even though total registration across all sub-models reached 52–94%. Two short, well-textured sequences reconstructed essentially completely: H03 reconstructed as a single complete model, and H06's largest model covered all input frames despite an extra small sub-model (see the Table 1a footnote). Sequence H05 is a clear failure case: high reprojection error (1.09 px), a near-degenerate recovered scale (Sim(3) scale 0.19), and 1.87 m ATE.

**Table 1a.** Per-sequence COLMAP reconstruction and registration on the AQUALOC harbor sequences. "Largest model" is the largest connected reconstruction; "total reg." sums registered images across all sub-models.

| Seq | Stride | Input imgs | Sub-models | Largest model (imgs / %) | Total reg. | Sparse points | Reproj. err (px) | Track len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H01 | 5 | 918 | 2 | 306 / 33.3% | 475 | 17,629 | 0.351 | 11.64 |
| H02 | 5 | 1629 | 5 | 772 / 47.4% | 1341 | 55,653 | 0.422 | 9.51 |
| H03 | 10 | 516 | 1 | 516 / 100% | 516 | 109,666 | 0.325 | 7.84 |
| H04 | 10 | 413 | 4 | 170 / 41.2% | 310 | 19,838 | 0.308 | 5.16 |
| H05 | 10 | 346 | 4 | 173 / 50.0% | 324 | 38,680 | 1.087 | 4.67 |
| H06 | 10 | 254 | 2 | 254 / 100% | 256 | 54,257 | 0.352 | 6.20 |
| H07 | 5 | 453 | 4 | 191 / 42.2% | 422 | 39,183 | 0.293 | 5.92 |

*Footnote (Table 1a):* "Total reg." is the sum of registered images over all sub-models and can therefore exceed the number of input images, because COLMAP can register the same physical frame in more than one sub-model. This is why H06 shows 256 total registered images from 254 inputs: its largest model registered all 254 frames and a second small sub-model independently re-registered 2 of them (the largest-model column is never affected by this double-counting).

**Table 1b.** Per-sequence COLMAP trajectory accuracy on the same sequences. ATE/RPE are computed against the dataset ground-truth trajectory after Sim(3) alignment; the Sim(3) scale is reported as a direct measure of scale ambiguity.

| Seq | ATE RMSE (m) | ATE std (m) | RPE mean (m) | Sim(3) scale |
|---|---:|---:|---:|---:|
| H01 | 0.0149 | 0.0123 | 0.0037 | 0.921 |
| H02 | 0.0248 | 0.0153 | 0.0010 | 0.697 |
| H03 | 0.2453 | 0.1627 | 0.0007 | 0.440 |
| H04 | 0.0384 | 0.0164 | 0.0021 | 1.425 |
| H05 | 1.8662 | 0.8643 | 0.0848 | 0.192 |
| H06 | 0.0525 | 0.0351 | 0.0017 | 0.508 |
| H07 | 0.0096 | 0.0059 | 0.0012 | 0.404 |

Complete registration does not by itself guarantee an accurate trajectory: registration only means every frame was placed into one connected reconstruction, whereas ATE measures how metrically correct those placements are after Sim(3) alignment. H03 illustrates this directly — it registers 100% of frames yet has the second-worst ATE (24.5 cm) and the second-smallest recovered scale (0.44), indicating that although the frames form one model, the reconstruction is geometrically distorted (a global scale/shape bias, consistent with weak parallax and unmodeled refraction) rather than fragmented. Notably its very low RPE (0.7 mm) shows that local frame-to-frame motion is recovered well even while the global trajectory drifts.

On a representative run (H07, 453 images), GPU feature extraction took 2.6 s, sequential matching 3.1 s, and incremental mapping 91 s. Mapping time grew sharply with image count and scene difficulty: H02 (1629 images) required 552 s, and at stride five the two largest sequences did not converge within a ~25-minute budget, motivating the stride-ten setting for the larger sequences.

[Figure 1 about here. Figure legends are provided in the separate figure legends file.]

[Figure 2 about here. Figure legends are provided in the separate figure legends file.]

### 5.2 DUSt3R on harbor windows

DUSt3R was run on 20-frame windows (120 image pairs each) from three sequences. Each window completed inference and 300-iteration global alignment in ~47 s with a peak GPU memory of 4.44 GB, and recovered poses for all 20 frames. Compared against the **COLMAP full-sequence reference values** for the same sequences (Table 2), DUSt3R's window trajectories were less accurate (ATE RMSE 5.8–7.2 cm versus COLMAP's full-sequence 1.0–2.5 cm), but that comparison mixes 20-frame windows against full sequences; the matched-window comparison below removes that confound. The recovered Sim(3) scale ranged from 3.2 to 6.3 on every window — never close to 1 — confirming a large and consistent reconstruction-scale ambiguity that must be resolved with an external reference before the geometry is metric.

A scaling limit was also observed: global alignment of 40 images exceeded the 12 GB budget, so a full sequence could not be processed at once on this hardware.

**Table 2.** DUSt3R on 20-frame windows of three harbor sequences, with the COLMAP **full-sequence** reference ATE for the same sequences. ATE/RPE are Sim(3)-aligned against the dataset ground truth.

| Seq | Pairs | Runtime (s) | Peak VRAM (GB) | ATE RMSE (m) | RPE mean (m) | Sim(3) scale | COLMAP full-seq ATE (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| H01 | 120 | 46.7 | 4.44 | 0.0591 | 0.0151 | 6.34 | 0.0149 |
| H02 | 120 | 46.9 | 4.44 | 0.0580 | 0.0175 | 3.18 | 0.0248 |
| H07 | 120 | 46.6 | 4.44 | 0.0720 | 0.0431 | 4.50 | 0.0096 |

**Matched-window comparison.** Running COLMAP on the *exact same* 20-frame windows (Table 2b) tells a different and fairer story than the full-sequence numbers. On these short windows COLMAP no longer dominates: on the **H07** window it failed to reconstruct, registering only 2 of 20 frames (no trajectory), whereas DUSt3R recovered all 20 poses on that same window. On the **H01** window COLMAP registered all 20 frames but produced a near-degenerate reconstruction (Sim(3) scale 0.03, ATE 9.8 cm) that was actually *worse* than DUSt3R's window result (5.9 cm). Only on the **H02** window was COLMAP clearly better (sub-millimeter ATE at this short scale). Two caveats apply to the COLMAP window ATEs: a 20-frame harbor window spans a short, nearly straight path, so Sim(3) alignment can fit it almost exactly (as on H02) and the very small recovered scales (0.03–0.04) indicate the short-baseline geometry is close to degenerate, so these window ATEs should be read as indicative rather than as stable accuracy estimates. The practical takeaway is the qualitative one: on identical short windows, DUSt3R registered every frame on all three windows while COLMAP failed on one and was near-degenerate on another — concrete evidence for COLMAP's registration fragility and DUSt3R's registration robustness underwater.

**Table 2b.** Matched-window comparison: COLMAP and DUSt3R on the **same** 20-frame windows. COLMAP metrics are for its largest connected sub-model; "—" denotes a window COLMAP failed to reconstruct (too few frames registered for a trajectory). ATE/RPE are Sim(3)-aligned against the dataset ground truth.

| Seq | Method | Frames reg. | Sparse pts | Reproj. err (px) | ATE RMSE (m) | RPE mean (m) | Sim(3) scale |
|---|---|---:|---:|---:|---:|---:|---:|
| H01 | COLMAP (window) | 20 / 20 | 1,774 | 0.835 | 0.0975 | 0.0159 | 0.031 |
| H01 | DUSt3R (window) | 20 / 20 | — | — | 0.0591 | 0.0151 | 6.34 |
| H02 | COLMAP (window) | 20 / 20 | 3,908 | 0.350 | 0.0004 | 0.0002 | 0.040 |
| H02 | DUSt3R (window) | 20 / 20 | — | — | 0.0580 | 0.0175 | 3.18 |
| H07 | COLMAP (window) | 2 / 20 | 100 | 0.182 | — | — | — |
| H07 | DUSt3R (window) | 20 / 20 | — | — | 0.0720 | 0.0431 | 4.50 |

### 5.3 DUSt3R confidence collapse underwater

The most striking DUSt3R result concerns confidence. On underwater imagery the model's per-pixel confidence was uniformly low across all three windows: the maximum confidence reached only 2.68–3.17 and the mean was 1.3–2.0. Because the model's default filtering threshold is 3.0, **the default setting retained 0–0.5% of predicted points** across the three windows (exactly 0% on two of them). Table 3 and Figure 3 show the survival curves: a threshold of 1.5 retained 33–83% of points depending on the sequence, and 1.1 retained 68–95%. Any underwater use of DUSt3R must therefore re-tune this threshold; reporting "points after default filtering" without adjustment would be meaningless here. This confidence collapse is the single most consistent and, we argue, most transferable finding of the study: it held on all three windows, does not depend on the ground-truth reference trajectory, and gives a concrete, actionable number (use a threshold near 1.5, not 3.0) for anyone applying DUSt3R underwater. It is examined further in the Discussion.

**Table 3.** Fraction of DUSt3R dense points surviving confidence filtering across the three windows (each 4,096,000 total points).

| Confidence threshold | H01 surviving | H02 surviving | H07 surviving |
|---:|---:|---:|---:|
| 1.01 | 95.8% | 95.3% | 76.8% |
| 1.10 | 94.6% | 93.7% | 68.2% |
| 1.25 | 91.7% | 89.9% | 53.4% |
| 1.50 | 83.4% | 79.7% | 33.3% |
| 2.00 | 49.1% | 40.1% | 7.9% |
| 2.50 | 12.6% | 2.7% | 0.4% |
| 3.00 (default) | 0.5% | 0.0% | 0.0% |

[Figure 3 about here. Figure legends are provided in the separate figure legends file.]

## 6. Discussion

Three conclusions follow from the measurements. First, **COLMAP is the more accurate and interpretable method when it converges**: on five of seven harbor sequences it achieved centimeter-level trajectory accuracy and sub-half-pixel reprojection error, with fully auditable matches, tracks, and bundle-adjustment statistics. Second, **underwater conditions make COLMAP fragile in a specific, measurable way**: rather than always producing one continuous model, it tends to fragment a sequence into several disconnected sub-models (largest model 33–100% of frames), produced a clear full-sequence failure case on H05, and failed outright on the matched H07 short window. This fragmentation, not raw reprojection error, is the practical limitation underwater, and it is exactly the kind of failure that a single aggregate "mean reprojection error" statistic would hide. The present experiments do not isolate the cause of fragmentation, which is most likely dominated by weak texture, turbidity, and lighting changes that break feature tracks; however, the unmodeled refraction of the flat camera port is a plausible additional contributor, since the pinhole intrinsics used here cannot represent it and refractive bias is known to corrupt large-scale underwater reconstructions (7). Disentangling these factors — for example by comparing against Refractive COLMAP (7) — is left to future work.

Third, on the windows tested, **DUSt3R recovered all camera poses but, after Sim(3) alignment, had higher trajectory error than full-sequence COLMAP, and it shows a clear domain-shift signature**. It registered every frame without calibration and was recovered only up to a large unknown scale (3.2–6.3×) on every window. The matched-window experiment sharpens this picture: on identical 20-frame windows, COLMAP did *not* uniformly outperform DUSt3R — it failed to reconstruct the H07 window (2 of 20 frames) and was near-degenerate on H01, while DUSt3R registered all frames on all three windows. So rather than a blanket "accuracy versus robustness" trade-off, the fairer statement is narrower: where feature matching is too weak for COLMAP to even register a short clip, DUSt3R still returns a complete (if up-to-scale and uncertain) pose set. The strongest and most transferable evidence of domain shift is the confidence collapse: a model confident enough on terrestrial data to ship with a default filtering threshold of 3.0 barely reaches it underwater (maximum 2.68–3.17), so the default discards 99.5–100% of points and a threshold near 1.5 is required to retain a usable fraction. Because the trajectory results come from three short 20-frame windows, the accuracy comparison should be read as indicative rather than definitive; the confidence-collapse and scale-ambiguity findings, which were consistent across all three windows and do not depend on the reference trajectory, are the more robust conclusions. The practical consequence is unambiguous: any underwater use of DUSt3R must lower the default confidence filter, or essentially all output is discarded.

One caveat applies to the trajectory comparison specifically. The AQUALOC ground-truth trajectories were themselves produced by an offline COLMAP/SfM bundle-adjustment pipeline, so the ATE/RPE evaluation compares an online COLMAP reconstruction against a COLMAP-derived reference. This likely biases the trajectory comparison in COLMAP's favor and may partly explain its very low ATE on the well-behaved sequences; the registration-fragmentation and confidence-collapse findings, which do not depend on the reference trajectory, are not affected by this bias. An external metric ground truth (for example, FLSea's object-validated depth maps) would be needed to remove it.

Taken together, the two methods are complementary rather than ranked: COLMAP for metric accuracy and interpretability where it converges, DUSt3R for calibration-free coverage where feature matching fails, provided its scale and confidence are handled explicitly.

## 7. Limitations

This study evaluated a single environment (the AQUALOC harbor site) on a single 12 GB workstation. The CIRS and FLSea datasets, and the deep AQUALOC archaeological sequences, were not processed, so the conclusions should not be extended to caves, forward-looking stereo, or deep-sea scenes without further experiments. DUSt3R was evaluated on three 20-frame windows because of the memory limit, so its trajectory statistics are based on short segments rather than full sequences and the accuracy comparison is indicative rather than definitive; the matched-window COLMAP comparison (Section 5.2) removes the full-sequence-versus-window confound but is itself limited to those same three short windows, on which Sim(3) alignment of a nearly straight path can be near-degenerate. Dense MVS for COLMAP and dense depth-map validation for DUSt3R (which requires FLSea's ground-truth depth) were out of scope. Critically, the reconstructions were evaluated only through quantitative pose and reprojection metrics — no qualitative visual inspection or perceptual scoring of the actual 3D point clouds (e.g., side-by-side rendering, expert rating of geometric plausibility, or measurement of known structures) was performed, so failures that the metrics do not capture (such as locally plausible but globally warped geometry) may remain undetected. Two larger sequences used a coarser frame stride than the others, so per-sequence numbers should be read together with the stride column. Finally, ATE/RPE are reported against the dataset's offline COLMAP/SfM trajectory, which is itself an estimate rather than an external metric ground truth and likely favors COLMAP in the trajectory comparison (see Section 6).

## 8. Conclusion

This paper reported a fully specified, reproducible comparison of COLMAP and DUSt3R on the seven AQUALOC harbor sequences, run end to end on one RTX 4070 Super workstation, including a matched-window comparison on identical 20-frame clips. COLMAP reconstructed 334,906 sparse points with a median 0.35 px reprojection error and centimeter-level trajectory accuracy on most full sequences, but its registration was frequently fragmented and H05 was a clear full-sequence failure case; on the matched short windows it also failed to register one H07 clip (2 of 20 frames) that DUSt3R handled completely. DUSt3R recovered all poses on the three tested windows without calibration but at lower, up-to-scale accuracy, and its per-pixel confidence never reached the level required by its own default filter, so default filtering removed essentially all points — the study's most consistent and transferable result. Neither method universally outperforms the other; their failure modes differ, with COLMAP failing visibly through fragmentation and DUSt3R producing complete but uncertain, up-to-scale pose and pointmap estimates. Future work will extend this protocol to FLSea (for dense depth validation), the CIRS caves, and the deep AQUALOC archaeological sites, and will investigate hybrid pipelines that combine COLMAP's metric accuracy with DUSt3R's robustness.

## Data and Code Availability

The datasets are publicly available: AQUALOC (https://www.lirmm.fr/aqualoc/), CIRS Underwater Caves (https://cirs.udg.edu/caves-dataset/), and FLSea (arXiv:2302.12772). The reconstruction and evaluation scripts used here (`run_colmap_gpu.py`, `run_colmap_window.py`, `run_dust3r_sweep.py`, `make_figures.py`), together with the per-sequence metric logs (JSON) and exact software versions, are archived in a public repository at https://github.com/dev1014/research (commit `e0b09c0f3358cd8fa070885d3e2762b145b27b58`). Each result table and figure in this paper can be regenerated from these scripts and the public datasets, with the configuration fully specified in Section 4.

**Author contributions.** D.N. designed and ran all experiments, wrote the reconstruction and evaluation scripts (`run_colmap_gpu.py`, `run_colmap_window.py`, `run_dust3r_sweep.py`, `make_figures.py`), performed the analysis, produced the figures, and wrote the manuscript. The work uses publicly released software and model weights without modification: COLMAP (the native CUDA build) and DUSt3R (the official `DUSt3R_ViTLarge_BaseDecoder_512_dpt` checkpoint). No new model was trained or fine-tuned, and no dataset was collected by the author; all imagery and ground-truth trajectories are from the public AQUALOC dataset.

## Acknowledgments

The author thanks the authors of the AQUALOC dataset for making the sequences and ground-truth trajectories publicly available, and the developers of COLMAP and DUSt3R for releasing their software and model weights.

## References

1. Schönberger, J. L., & Frahm, J.-M. (2016). Structure-from-motion revisited. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 4104–4113). https://doi.org/10.1109/CVPR.2016.445
2. Schönberger, J. L., Zheng, E., Pollefeys, M., & Frahm, J.-M. (2016). Pixelwise view selection for unstructured multi-view stereo. In *European Conference on Computer Vision (ECCV)* (pp. 501–518). https://doi.org/10.1007/978-3-319-46487-9_31
3. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., & Revaud, J. (2024). DUSt3R: Geometric 3D vision made easy. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 20697–20709). https://doi.org/10.1109/CVPR52733.2024.01956
4. Ferrera, M., Creuze, V., Moras, J., & Trouvé-Peloux, P. (2019). AQUALOC: An underwater dataset for visual-inertial-pressure localization. *The International Journal of Robotics Research, 38*(14), 1549–1559. https://doi.org/10.1177/0278364919883346
5. Mallios, A., Vidal, E., Campos, R., & Carreras, M. (2017). Underwater caves sonar data set. *The International Journal of Robotics Research, 36*(12), 1247–1251. https://doi.org/10.1177/0278364917732838
6. Randall, Y., & Treibitz, T. (2023). FLSea: Underwater visual-inertial and stereo-vision forward-looking datasets. *arXiv*. https://doi.org/10.48550/arXiv.2302.12772
7. She, M., Seegräber, F., Nakath, D., & Köser, K. (2024). Refractive structure-from-motion revisited. *arXiv*. https://doi.org/10.48550/arXiv.2403.08640
8. Umeyama, S. (1991). Least-squares estimation of transformation parameters between two point patterns. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 13*(4), 376–380. https://doi.org/10.1109/34.88573
