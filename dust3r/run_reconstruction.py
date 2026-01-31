"""
Dust3r 3D Reconstruction Script for Gerrard Hall Dataset
Runs Dust3r inference + global alignment, exports point cloud (.ply) and scene (.glb)
"""
import os
import sys
import argparse
import glob
import numpy as np
import torch
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation

# Add dust3r to path
DUST3R_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DUST3R_DIR)

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

torch.backends.cuda.matmul.allow_tf32 = True


def export_ply(pts3d, colors, mask, output_path):
    """Export point cloud as PLY file."""
    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([c[m] for c, m in zip(colors, mask)])

    # Ensure colors are in 0-255 uint8 range
    if col.max() <= 1.0:
        col = (col * 255).astype(np.uint8)
    else:
        col = col.astype(np.uint8)

    pcd = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
    pcd.export(output_path)
    print(f"Exported point cloud: {output_path} ({pts.reshape(-1, 3).shape[0]} points)")


def export_glb(imgs, pts3d, mask, focals, cams2world, output_path, as_pointcloud=True, cam_size=0.05):
    """Export scene as GLB file with cameras."""
    scene = trimesh.Scene()

    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    for i, pose_c2w in enumerate(cams2world):
        camera_edge_color = CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color, imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    scene.export(file_obj=output_path)
    print(f"Exported scene: {output_path}")


def export_cameras(cams2world, focals, img_names, output_path):
    """Export camera poses to a text file."""
    with open(output_path, 'w') as f:
        f.write("# image_name focal_length tx ty tz qw qx qy qz\n")
        for i, (pose, focal, name) in enumerate(zip(cams2world, focals, img_names)):
            t = pose[:3, 3]
            r = Rotation.from_matrix(pose[:3, :3])
            q = r.as_quat()  # xyzw format
            f.write(f"{name} {focal:.4f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}\n")
    print(f"Exported camera poses: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Dust3r 3D Reconstruction")
    parser.add_argument("--image_dir", type=str,
                        default=os.path.join(DUST3R_DIR, "..", "datasets", "gerrard-hall", "images"),
                        help="Path to image directory")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(DUST3R_DIR, "output"),
                        help="Path to output directory")
    parser.add_argument("--model_name", type=str,
                        default="DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                        help="Dust3r model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--image_size", type=int, default=512, help="Image size for Dust3r")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--niter", type=int, default=300, help="Global alignment iterations")
    parser.add_argument("--schedule", type=str, default="cosine", help="LR schedule")
    parser.add_argument("--scene_graph", type=str, default="swin-5",
                        help="Scene graph type: complete, swin-N, oneref-N")
    parser.add_argument("--min_conf_thr", type=float, default=3.0, help="Min confidence threshold")
    parser.add_argument("--subsample", type=int, default=3,
                        help="Use every Nth image (1=all, 3=every 3rd, etc.)")
    parser.add_argument("--max_images", type=int, default=50,
                        help="Maximum number of images to use")
    parser.add_argument("--clean_depth", action="store_true", default=True,
                        help="Clean up depth maps")
    args = parser.parse_args()

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect images
    image_dir = os.path.abspath(args.image_dir)
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.JPG")) +
                         glob.glob(os.path.join(image_dir, "*.jpg")) +
                         glob.glob(os.path.join(image_dir, "*.png")))

    if not image_files:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images in {image_dir}")

    # Subsample images
    if args.subsample > 1:
        image_files = image_files[::args.subsample]
        print(f"After subsampling (every {args.subsample}): {len(image_files)} images")

    if len(image_files) > args.max_images:
        image_files = image_files[:args.max_images]
        print(f"Capped at {args.max_images} images")

    print(f"Using {len(image_files)} images for reconstruction")
    img_names = [os.path.basename(f) for f in image_files]

    # Load model
    print(f"\nLoading Dust3r model: {args.model_name}...")
    weights_path = "naver/" + args.model_name
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)
    print("Model loaded successfully!")

    # Load images
    print(f"\nLoading and preprocessing images (size={args.image_size})...")
    imgs = load_images(image_files, size=args.image_size, verbose=True)
    print(f"Loaded {len(imgs)} images")

    # Make pairs
    print(f"\nCreating image pairs (scene_graph={args.scene_graph})...")
    pairs = make_pairs(imgs, scene_graph=args.scene_graph, prefilter=None, symmetrize=True)
    print(f"Created {len(pairs)} image pairs")

    # Run inference
    print(f"\nRunning Dust3r inference on {len(pairs)} pairs...")
    output = inference(pairs, model, args.device, batch_size=args.batch_size, verbose=True)
    print("Inference complete!")

    # Global alignment
    print(f"\nRunning global alignment ({args.niter} iterations, schedule={args.schedule})...")
    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=args.device, mode=mode, verbose=True)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=args.niter, schedule=args.schedule, lr=0.01)
        print(f"Global alignment complete! Final loss: {loss:.4f}")

    # Clean depth if requested
    if args.clean_depth:
        scene = scene.clean_pointcloud()

    # Extract results
    print("\nExtracting results...")
    rgbimg = scene.imgs
    focals = to_numpy(scene.get_focals().cpu())
    cams2world = to_numpy(scene.get_im_poses().cpu())
    pts3d = to_numpy(scene.get_pts3d())

    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(args.min_conf_thr)))
    mask = to_numpy(scene.get_masks())

    # Export PLY point cloud
    ply_path = os.path.join(args.output_dir, "dust3r_pointcloud.ply")
    export_ply(pts3d, rgbimg, mask, ply_path)

    # Export GLB scene
    glb_path = os.path.join(args.output_dir, "dust3r_scene.glb")
    export_glb(rgbimg, pts3d, mask, focals, cams2world, glb_path, as_pointcloud=True)

    # Export camera poses
    cam_path = os.path.join(args.output_dir, "dust3r_cameras.txt")
    export_cameras(cams2world, focals.flatten(), img_names[:len(cams2world)], cam_path)

    print(f"\n{'='*60}")
    print("Dust3r Reconstruction Complete!")
    print(f"  Point cloud: {ply_path}")
    print(f"  Scene (GLB): {glb_path}")
    print(f"  Camera poses: {cam_path}")
    print(f"  Images used: {len(imgs)}")
    num_points = sum(m.sum() for m in mask)
    print(f"  Total points: {num_points}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
