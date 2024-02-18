import numpy as np
import cv2
import os
import h5py
import binvox_rw_customized
import mcubes
import cutils
import argparse
import time

#python bin_voxelization/2_floodfill.py 0 1 /local-scratch2/fenggeny/SECAD-Net-main/data/mingrui_data/shapes

parser = argparse.ArgumentParser()
parser.add_argument("share_id", type=int, help="id of the share [0]")
parser.add_argument("share_total", type=int, help="total num of shares [1]")
parser.add_argument("target_dir", type=str, help="total num of shares [1]")
FLAGS = parser.parse_args()

target_dir = FLAGS.target_dir + '/'
if not os.path.exists(target_dir):
    print("ERROR: this dir does not exist: "+target_dir)
    exit()

obj_names = os.listdir(target_dir)
obj_names = sorted(obj_names)

share_id = FLAGS.share_id
share_total = FLAGS.share_total

start = int(share_id*len(obj_names)/share_total)
end = int((share_id+1)*len(obj_names)/share_total)
obj_names = obj_names[start:end]

def write_ply_triangle(name, vertices, triangles):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("element face "+str(len(triangles))+"\n")
	fout.write("property list uchar int vertex_index\n")
	fout.write("end_header\n")
	for ii in range(len(vertices)):
		fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	for ii in range(len(triangles)):
		fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
	fout.close()


queue = np.zeros([1024*1024*32,3], np.int32)
state_ctr = np.zeros([1024*1024*32,2], np.int32)

start_time = time.time()
for i in range(len(obj_names)):
    this_name = target_dir + obj_names[i] + "/model.binvox"
    out_name = target_dir + obj_names[i] + "/model_filled.binvox"
    print(i,this_name)
    if os.path.exists(out_name):
        continue
    voxel_model_file = open(this_name, 'rb')
    vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file,fix_coords=False)

    batch_voxels = vox_model.data.astype(np.uint8) + 1
    cutils.floodfill(batch_voxels,queue)
    cutils.get_state_ctr(batch_voxels,state_ctr)

    with open(out_name, 'wb') as fout:
        binvox_rw_customized.write(vox_model, fout, state_ctr)
    print('left time: ', ( (time.time() - start_time) * (len(obj_names) - i)/(i+1) ))
    '''
    voxel_model_file = open(out_name, 'rb')
    vox_model = binvox_rw_customized.read_as_3d_array(voxel_model_file)
    batch_voxels = vox_model.data.astype(np.uint8)
    vertices, triangles = mcubes.marching_cubes(batch_voxels, 0.5)
    write_ply_triangle("vox.ply", vertices, triangles)

    exit(0)
    '''

