import subprocess
import shutil
import os
import sys
import argparse
from pathlib import Path
import constants as const

def run_command(cmd, description):
    """Runs a shell command and prints status."""
    print(f"--- Running {description} ---")
    print(f"Command: {' '.join(cmd)}")
    try:
        # Run command and stream output to verify progress
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"✅ {description} completed successfully.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed.")
        print(e.output)
        return False

def setup_directories():
    """Creates necessary output directories."""
    const.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    const.SPARSE_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Directories checked at {const.OUTPUT_DIR}")

def feature_extraction():
    """Extracts features from images."""
    if const.DATABASE_PATH.exists():
        print(f"Database already exists at {const.DATABASE_PATH}, skipping extraction (delete db to redo).")
        return True
        
    cmd = [
        const.COLMAP_EXE, "feature_extractor",
        "--database_path", str(const.DATABASE_PATH),
        "--image_path", str(const.IMAGES_PATH),
    ]
    return run_command(cmd, "Feature Extraction")

def feature_matching():
    """Matches features between images."""
    cmd = [
        const.COLMAP_EXE, const.MATCH_TYPE,
        "--database_path", str(const.DATABASE_PATH),
    ]
    return run_command(cmd, "Feature Matching")

def sparse_reconstruction():
    """Runs the mapper for sparse reconstruction (SfM)."""
    # Check if sparse model already exists to avoid re-running if mostly done, 
    # but for this script we usually want to ensure it runs or use --skip options.
    # We will just run it. COLMAP usually handles incremental or overwrites.
    
    cmd = [
        const.COLMAP_EXE, "mapper",
        "--database_path", str(const.DATABASE_PATH),
        "--image_path", str(const.IMAGES_PATH),
        "--output_path", str(const.SPARSE_OUTPUT_PATH)
    ]
    return run_command(cmd, "Sparse Reconstruction")

def image_undistortion():
    """Undistorts images for dense reconstruction."""
    const.DENSE_WORKSPACE.mkdir(parents=True, exist_ok=True)
    
    # We need to find the sparse model directory (usually '0')
    subdir = const.SPARSE_OUTPUT_PATH / "0"
    if not subdir.exists():
        print(f"❌ Sparse model not found at {subdir}. Cannot proceed to dense.")
        return False
        
    cmd = [
        const.COLMAP_EXE, "image_undistorter",
        "--image_path", str(const.IMAGES_PATH),
        "--input_path", str(subdir),
        "--output_path", str(const.DENSE_WORKSPACE),
        "--output_type", "COLMAP",
        "--max_image_size", "2000"
    ]
    return run_command(cmd, "Image Undistortion")

def patch_match_stereo():
    """Runs patch match stereo for depth estimation."""
    cmd = [
        const.COLMAP_EXE, "patch_match_stereo",
        "--workspace_path", str(const.DENSE_WORKSPACE),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true"
    ]
    return run_command(cmd, "Patch Match Stereo")

def stereo_fusion():
    """Fuses stereo depth maps into a dense point cloud."""
    output_ply = const.DENSE_WORKSPACE / "fused.ply"
    cmd = [
        const.COLMAP_EXE, "stereo_fusion",
        "--workspace_path", str(const.DENSE_WORKSPACE),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(output_ply)
    ]
    return run_command(cmd, "Stereo Fusion")

def analyze_results():
    """Analyzes the reconstruction results."""
    # Relying on standard '0' folder
    results_path = const.SPARSE_OUTPUT_PATH / "0"
    if results_path.exists():
        print(f"Analyzing model in: {results_path}")
        cmd = [const.COLMAP_EXE, "model_analyzer", "--path", str(results_path)]
        run_command(cmd, "Model Analysis")

def run_sparse():
    print("--- Starting Sparse Reconstruction ---")
    setup_directories()
    if not feature_extraction(): return False
    if not feature_matching(): return False
    if not sparse_reconstruction(): return False
    analyze_results()
    print(f"✅ Sparse reconstruction available at {const.SPARSE_OUTPUT_PATH}")
    return True

def run_dense():
    print("--- Starting Dense Reconstruction ---")
    # Dense requires sparse output. 
    # We assume sparse is done if we are running dense, OR we check.
    # If sparse '0' doesn't exist, we must run sparse first.
    if not (const.SPARSE_OUTPUT_PATH / "0").exists():
        print("Sparse model not found. Running sparse pipeline first...")
        if not run_sparse(): return False
    
    if not image_undistortion(): return False
    if not patch_match_stereo(): return False
    if not stereo_fusion(): return False
    
    print(f"✅ Dense reconstruction completed.")
    print(f"Result: {const.DENSE_WORKSPACE / 'fused.ply'}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run COLMAP reconstruction pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sparse", action="store_true", help="Run only Sparse Reconstruction (SfM)")
    group.add_argument("--dense", action="store_true", help="Run Sparse (if needed) + Dense Reconstruction (MVS)")
    
    args = parser.parse_args()
    
    if args.sparse:
        run_sparse()
    elif args.dense:
        run_dense()

if __name__ == "__main__":
    main()
