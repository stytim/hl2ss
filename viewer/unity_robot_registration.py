#------------------------------------------------------------------------------
# This script adds a cube to the Unity scene and animates it.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import multiprocessing as mp

import hl2ss
import hl2ss_lnm
import hl2ss_rus
import hl2ss_3dcv
import hl2ss_mp
import open3d as o3d
import cv2

# Settings --------------------------------------------------------------------

# HoloLens address
host = '192.168.0.27'

# Calibration path (must exist but can be empty)
calibration_path = '../calibration'

# Buffer length in seconds
buffer_length = 10

# Integration parameters
voxel_length = 1/100
sdf_trunc = 0.04
max_depth = 3.0

def unity_to_open3d_bounds(unity_data):
    # Extracting position and scale from the Unity data
    position = unity_data[:3]
    scale = unity_data[3:]
    
    # Convert Unity position to Open3D (adjusting Z-axis sign)
    position[2] = -position[2]
    tmp = scale[0]
    scale[0] = scale[2]
    scale[2] = tmp

    # Calculate min and max bounds for Open3D
    half_scale = scale / 2
    min_bound = position - half_scale
    max_bound = position + half_scale

    return min_bound, max_bound


if __name__ == '__main__':
    enable = True
    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
    ipc.open()

    key = 0
    bbox = 0

    print("Waiting for data...")
    listening = True
    # Loop until receive data
    while(listening):
        data = ipc.pull_msg()
        if (data is not None):
            bbox = data
            print(bbox)
            listening = False
            break


    # Get RM Depth Long Throw calibration -------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)

    uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
    
    # Create Open3D integrator and visualizer ---------------------------------
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=sdf_trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    intrinsics_depth = o3d.camera.PinholeCameraIntrinsic(hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT, calibration_lt.intrinsics[0, 0], calibration_lt.intrinsics[1, 1], calibration_lt.intrinsics[2, 0], calibration_lt.intrinsics[2, 1])
    
    first_pcd = True

    # Start RM Depth Long Throw stream ----------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS * buffer_length)
    producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()    
    sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, ...)

    sink_depth.get_attach_response()

    print("Waiting for point cloud...")
    while (enable):

        # Wait for RM Depth Long Throw frame ----------------------------------
        sink_depth.acquire()

        # Get RM Depth Long Throw frame ---------------------------------------
        _, data_depth = sink_depth.get_most_recent_frame()
        if ((data_depth is None) or (not hl2ss.is_valid_pose(data_depth.pose))):
            continue

        # Preprocess frames ---------------------------------------------------
        depth = hl2ss_3dcv.rm_depth_undistort(data_depth.payload.depth, calibration_lt.undistort_map)
        depth = hl2ss_3dcv.rm_depth_normalize(depth, scale)
        color = cv2.remap(data_depth.payload.ab, calibration_lt.undistort_map[:, :, 0], calibration_lt.undistort_map[:, :, 1], cv2.INTER_LINEAR)
        
        # Convert to Open3D RGBD image ----------------------------------------
        color = hl2ss_3dcv.rm_depth_to_uint8(color)
        color = hl2ss_3dcv.rm_depth_to_rgb(color)
        color_image = o3d.geometry.Image(color)
        depth_image = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, depth_scale=1, depth_trunc=max_depth, convert_rgb_to_intensity=False)
        
        # Compute world to RM Depth Long Throw camera transformation matrix ---
        depth_world_to_camera = hl2ss_3dcv.world_to_reference(data_depth.pose) @ hl2ss_3dcv.rignode_to_camera(calibration_lt.extrinsics)

        # Integrate RGBD  ------------------------------
        volume.integrate(rgbd, intrinsics_depth, depth_world_to_camera.transpose())
        print("Acquired point cloud")


    # Stop RM Depth Long Throw stream -----------------------------------------
    sink_depth.detach()
    producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    listener.join()
    pcd = volume.extract_point_cloud()

    # Crop point cloud --------------------------------------------------------
    min_bound, max_bound = unity_to_open3d_bounds(bbox)

    # cropping
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped_pcd = pcd.crop(bounding_box)

    # o3d.visualization.draw_geometries([cropped_pcd], "Cropped Point Cloud")

    # o3d.io.write_point_cloud("cropped_hl2.pcd", cropped_pcd)
    print("Prepare for registartion")
    ipc.close()
    

