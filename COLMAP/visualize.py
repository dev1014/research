import open3d as o3d
import numpy as np
import constants as const
import sys
from pathlib import Path

def render_ply(ply_path):
    print(f"Loading point cloud from: {ply_path}")
    
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    if pcd.is_empty():
        print("❌ Point cloud is empty or could not be loaded.")
        return

    print(f"✅ Loaded {len(pcd.points)} points.")
    
    # Create a visualizer window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="COLMAP Dense Reconstruction", width=1280, height=720)
    
    # Add geometry
    vis.add_geometry(pcd)
    
    # Optional: Set better initial view
    # This aligns the view roughly to look at the object center
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # Render option customization (optional)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1]) # Dark grey background
    opt.point_size = 2.0 # Make points slightly visible
    
    print("Opening interactive viewer... (Close window to exit)")
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def main():
    # Path to the fused ply file
    ply_file = Path(r"C:\Users\devna\OneDrive\Desktop\Projects\research\dust3r\output\dust3r_pointcloud.ply")
    
    # Check if exists
    if not ply_file.exists():
        print(f"❌ File not found: {ply_file}")
        print("Did you run 'python reconstruct.py --dense' yet?")
        sys.exit(1)
        
    render_ply(ply_file)

if __name__ == "__main__":
    main()
