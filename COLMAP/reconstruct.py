import subprocess
import shutil
import os
import sys
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
    if const.OUTPUT_DIR.exists():
        print(f"Warning: Output directory {const.OUTPUT_DIR} already exists.")
        # Optional: Ask to clean? For now, we'll just carry on or cleanup if needed.
        # For a clean test, maybe we should start fresh?
        # shutil.rmtree(const.OUTPUT_DIR) # DATA LOSS RISK, better not auto-delete unless sure.
    
    const.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    const.SPARSE_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Directories setup at {const.OUTPUT_DIR}")

def feature_extraction():
    """Extracts features from images."""
    cmd = [
        const.COLMAP_EXE, "feature_extractor",
        "--database_path", str(const.DATABASE_PATH),
        "--image_path", str(const.IMAGES_PATH),
        # Optimal settings usually default, but we can enforce SIFT GPU usage if available
        # "--SiftExtraction.use_gpu", "1" 
    ]
    return run_command(cmd, "Feature Extraction")

def feature_matching():
    """Matches features between images."""
    cmd = [
        const.COLMAP_EXE, const.MATCH_TYPE,
        "--database_path", str(const.DATABASE_PATH),
        # "--SiftMatching.use_gpu", "1"
    ]
    return run_command(cmd, "Feature Matching")

def sparse_reconstruction():
    """Runs the mapper for sparse reconstruction (SfM)."""
    cmd = [
        const.COLMAP_EXE, "mapper",
        "--database_path", str(const.DATABASE_PATH),
        "--image_path", str(const.IMAGES_PATH),
        "--output_path", str(const.SPARSE_OUTPUT_PATH)
    ]
    return run_command(cmd, "Sparse Reconstruction")

def analyze_results():
    """Analyzes the reconstruction results."""
    # Check if model files exist
    model_files = ["cameras.bin", "images.bin", "points3D.bin"]
    # Mapper output is typically in a subdirectory '0', '1' etc. if multiple models found.
    # Usually '0' is the largest component.
    
    # Mapper might output directly to SPARSE_OUTPUT_PATH if only one model, 
    # OR it creates generated subfolders like "0", "1".
    # Let's check subdirectories.
    
    # Reload directory contents
    subdirs = [x for x in const.SPARSE_OUTPUT_PATH.iterdir() if x.is_dir()]
    
    if not subdirs:
        # Sometimes it saves directly if configured, but default 'mapper' saves to subdirs
        # Let's check if files are in the root of sparse output
        if all((const.SPARSE_OUTPUT_PATH / f).exists() for f in model_files):
             results_path = const.SPARSE_OUTPUT_PATH
        else:
            print("❌ No reconstruction sub-folders found in sparse output.")
            return
    else:
        # Assume '0' is the best model
        results_path = subdirs[0]
    
    print(f"Analyzing model in: {results_path}")
    
    cmd = [
        const.COLMAP_EXE, "model_analyzer",
        "--path", str(results_path)
    ]
    run_command(cmd, "Model Analysis")

def main():
    print("Starting COLMAP Reconstruction Pipeline Test")
    
    setup_directories()
    
    if not feature_extraction():
        sys.exit(1)
        
    if not feature_matching():
        sys.exit(1)
        
    if not sparse_reconstruction():
        sys.exit(1)
        
    analyze_results()
    
    print("\n---------------------------------------")
    print(f"Process Finished. Results are in {const.SPARSE_OUTPUT_PATH}")
    print("You can view them by running COLMAP GUI and importing the model.")

if __name__ == "__main__":
    main()
