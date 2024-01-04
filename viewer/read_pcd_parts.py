import open3d as o3d
import os

def combine_pcd_files(folder_path):
    """
    Reads all PCD (Point Cloud Data) files in a specified folder and combines them into one point cloud.
    
    Args:
    folder_path (str): The path to the folder containing PCD files.
    
    Returns:
    o3d.geometry.PointCloud: Combined point cloud from all PCD files in the folder.
    """
    combined_pcd = o3d.geometry.PointCloud()

    for filename in os.listdir(folder_path):
        if filename.endswith(".pcd"):
            pcd_path = os.path.join(folder_path, filename)
            pcd = o3d.io.read_point_cloud(pcd_path)
            combined_pcd += pcd

    return combined_pcd
pcd = combine_pcd_files('ur5_pcd')
o3d.io.write_point_cloud('ur5.pcd', pcd)
o3d.visualization.draw_geometries([pcd])