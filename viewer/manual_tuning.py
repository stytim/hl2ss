import hl2ss
import hl2ss_lnm
import hl2ss_rus
import hl2ss_3dcv
import hl2ss_mp
from scipy.spatial.transform import Rotation as R
import numpy as np



# HoloLens address
host = '192.168.1.181'


transformed_matrix =  np.array([[ 1,  0,  0, 0],
                                [0,   1,  0, 0],
                                [0,   0,  1, 0],
                                [0,   0,  0, 1]])

# Extract translation
position = transformed_matrix[0:3, 3]

# Extract rotation
rotation_matrix = transformed_matrix[0:3, 0:3]
rotation = R.from_matrix(rotation_matrix)
rotation = rotation.as_quat()  


print(position)
print(rotation)

ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()

display_list = hl2ss_rus.command_buffer()
display_list.begin_display_list() # Begin command sequence

display_list.set_registration(111, position, rotation, [1.0,1.0,1.0])
display_list.end_display_list() # End command sequence
ipc.push(display_list) # Send commands to server
results = ipc.pull(display_list) # Get results from server
print(results)

ipc.close()