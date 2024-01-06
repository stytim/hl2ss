import DracoPy
import open3d as o3d

# # Read pcd file
# pcd = o3d.io.read_point_cloud("ur5.pcd")

# binary = DracoPy.encode(pcd.points)

# with open('ur5.drc', 'wb') as test_file:
#   test_file.write(binary)


# Read drc file
with open('ur5.drc', 'rb') as draco_file:
    mesh = DracoPy.decode(draco_file.read())
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(mesh.points)
o3d.visualization.draw_geometries([pcd])