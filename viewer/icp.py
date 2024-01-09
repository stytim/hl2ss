import open3d as o3d
import numpy as np
import copy
import DracoPy

# Helper function to load point cloud
def load_point_cloud(file_name):
    return o3d.io.read_point_cloud(file_name)

# Helper function for point cloud preprocessing
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200))
    return pcd_down, pcd_fpfh

# Main registration function
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

# ICP refinement
def refine_registration(source, target, transformation, voxel_size):
    distance_threshold = voxel_size 
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=90000))
    return result


def point_cloud_registration(source, target):
    # Preprocess point clouds
    voxel_size = 0.03  # A parameter to tune depending on your data
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # Global registration
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh, voxel_size)
    # Refine registration
    result_icp = refine_registration(source, target, result_ransac.transformation, voxel_size)
    print("Refinement result:", result_icp)
    print("Matrix:", result_icp.transformation)
    return result_icp.transformation


if __name__ == '__main__':
    # Load point clouds
    source = load_point_cloud("hl2.pcd")
    # target = load_point_cloud("ur5.pcd")

    with open('ur5_t.drc', 'rb') as draco_file:
        mesh = DracoPy.decode(draco_file.read())
    
    target = o3d.geometry.PointCloud()
    # invert z axis
    target.points = o3d.utility.Vector3dVector(mesh.points * np.array([1, 1, -1]))

    transformation_matrix = point_cloud_registration(source, target)

    # Visualization (optional)
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation_matrix)
    o3d.visualization.draw_geometries([source_temp, target])
