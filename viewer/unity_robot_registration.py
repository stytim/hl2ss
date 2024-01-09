from pynput import keyboard

import multiprocessing as mp

import hl2ss
import hl2ss_lnm
import hl2ss_rus
import hl2ss_3dcv
import hl2ss_mp
import DracoPy
import open3d as o3d
import cv2
import numpy as np

import icp
import mac
import copy
from scipy.spatial.transform import Rotation as R

# Settings --------------------------------------------------------------------

# HoloLens address
host = '192.168.1.185'

# Calibration path (must exist but can be empty)
calibration_path = '../calibration'

# Buffer length in seconds
buffer_length = 10

# Integration parameters
voxel_length = 1/100
sdf_trunc = 0.04
max_depth = 3.0

def on_press(key):
    global running
    global enable
    global listening
    if key == keyboard.Key.esc:
        # Stop the loop
        running = False
        enable = False
        listening = False
        print("Esc pressed")
        return False

def unity_to_open3d_bounds(unity_data):
    # Extracting position and scale from the Unity data
    position = unity_data[:3]
    scale = unity_data[3:]
    
    # Convert Unity position to OpenGL (adjusting Z-axis sign)
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

    running = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while running:
        enable = True
        ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
        ipc.open()

        key = 0
        bbox = 0
        source = o3d.geometry.PointCloud()

        print("Waiting for data...")
        bbox_received = False
        point_cloud_received = False
        listening = True

        # Loop until receive bounding box and point cloud
        while (listening):
            command, data = ipc.pull_msg()
            if (data is not None):
                if (command == 0):
                    bbox = data
                    bbox_received = True
                    print("Received bounding box")
                    print(bbox)
                elif (command == 1):
                    mesh = DracoPy.decode(bytes(data))
                    with open('ur5_t.drc', 'wb') as test_file:
                        test_file.write(bytes(data))
                    source.points = o3d.utility.Vector3dVector(mesh.points * np.array([1, 1, -1]))
                    point_cloud_received = True
                    print("Received point cloud")
            listening = not (bbox_received and point_cloud_received)
            
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

        frame_id = 0
        calibrate = False
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

            frame_id += 1
            print("Acquired point cloud" )
            if (frame_id > 30):
                enable = False
                print("30 frames of point cloud acquired")
                calibrate = True
                break


        # Stop RM Depth Long Throw stream -----------------------------------------
        sink_depth.detach()
        producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

        if (not calibrate):
            continue

        pcd = volume.extract_point_cloud()
        min_bound, max_bound = unity_to_open3d_bounds(bbox)

        # cropping
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        target = pcd.crop(bounding_box)
        o3d.io.write_point_cloud("hl2.pcd", target)

        print("Prepare for registartion")
        # Start point cloud registration -------------------------------------------

        # transformation_matrix = icp.point_cloud_registration(source, target)
        transformation_matrix = mac.point_cloud_registration(source, target)
        # OpenGL <-> Unity
        flip_z_matrix = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])
        transformed_matrix = flip_z_matrix @ transformation_matrix @ flip_z_matrix
        # End point cloud registration -------------------------------------------

        # Extract translation
        position = transformed_matrix[0:3, 3]

        # Extract rotation
        rotation_matrix = transformed_matrix[0:3, 0:3]
        rotation = R.from_matrix(rotation_matrix)
        rotation = rotation.as_quat()  

        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list() # Begin command sequence
        display_list.set_registration(111, position, rotation, [1.0,1.0,1.0])
        display_list.end_display_list() # End command sequence
        ipc.push(display_list) # Send commands to server
        results = ipc.pull(display_list) # Get results from server
        print(results)

        ipc.close()

        # Visualization (optional)
        # source_temp = copy.deepcopy(source)
        # source_temp.transform(transformation_matrix)
        # o3d.visualization.draw_geometries([source_temp, target])

    listener.join()
    

