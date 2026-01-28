from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.resolve()
DATASET_ROOT = Path(r"c:\Users\devna\OneDrive\Desktop\Projects\research\datasets\gerrard-hall")

# Input Paths
IMAGES_PATH = DATASET_ROOT / "images"

# Output Paths
OUTPUT_DIR = BASE_DIR / "output"
DATABASE_PATH = OUTPUT_DIR / "database.db"
SPARSE_OUTPUT_PATH = OUTPUT_DIR / "sparse"

# Parameters
# "HIGH" quality is generally default/optimal for good reconstruction
# but for testing scripts "MEDIUM" might be faster. 
# We'll stick to defaults (which are high quality) for "optimal results".
MATCH_TYPE = "exhaustive_matcher" # simple and effective for <200 images

# Executable
COLMAP_EXE = "colmap"
