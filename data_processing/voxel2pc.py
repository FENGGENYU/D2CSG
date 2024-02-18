import numpy as np
import h5py, sys, os
from scipy.spatial import cKDTree as KDTree
import time
import random

batch_size_64 = 16*16*16*8

def get_points_from_vox(Pvoxel_model_64):
	
	# --- P 64 ---
	dim_voxel = 64
	batch_size = batch_size_64
	voxel_model_temp = Pvoxel_model_64
	
	sample_points = np.zeros([batch_size,3],np.uint8)
	sample_values = np.zeros([batch_size,1],np.uint8)
	batch_size_counter = 0
	voxel_model_temp_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
	nei = 2
	temp_range = list(range(nei,dim_voxel-nei,4))+list(range(nei+1,dim_voxel-nei,4))+list(range(nei+2,dim_voxel-nei,4))+list(range(nei+3,dim_voxel-nei,4))
	
	for j in temp_range:
		if (batch_size_counter>=batch_size): break
		for i in temp_range:
			if (batch_size_counter>=batch_size): break
			for k in temp_range:
				if (batch_size_counter>=batch_size): break
				if (np.max(voxel_model_temp[i-nei:i+nei+1,j-nei:j+nei+1,k-nei:k+nei+1])!= \
					np.min(voxel_model_temp[i-nei:i+nei+1,j-nei:j+nei+1,k-nei:k+nei+1])):
					#si,sj,sk = sample_point_in_cube(voxel_model_256_temp[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
					sample_points[batch_size_counter,0] = i#si+i*multiplier
					sample_points[batch_size_counter,1] = j#sj+j*multiplier
					sample_points[batch_size_counter,2] = k#sk+k*multiplier
					sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
					voxel_model_temp_flag[i,j,k] = 1
					batch_size_counter +=1
	if (batch_size_counter>=batch_size):
		print("64-- batch_size exceeded!")
		exceed_64_flag = 1
	else:
		exceed_64_flag = 0
		#fill other slots with random points
		while (batch_size_counter<batch_size):
			while True:
				i = random.randint(0,dim_voxel-1)
				j = random.randint(0,dim_voxel-1)
				k = random.randint(0,dim_voxel-1)
				if voxel_model_temp_flag[i,j,k] != 1: break
			#si,sj,sk = sample_point_in_cube(voxel_model_256_temp[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
			sample_points[batch_size_counter,0] = i#si+i*multiplier
			sample_points[batch_size_counter,1] = j#sj+j*multiplier
			sample_points[batch_size_counter,2] = k#sk+k*multiplier
			sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
			voxel_model_temp_flag[i,j,k] = 1
			batch_size_counter +=1
	
	# Psample_points_64 = sample_points
	# Psample_values_64 = sample_values
	points_value = np.concatenate((sample_points, sample_values), 1)
	return points_value

def voxel2pc_batch(dataset_dir):

	data_dict_name = os.path.join(dataset_dir, 'ae_voxel_points_samples.hdf5')
	data_dict = h5py.File(data_dict_name, 'r')
	voxels_all = data_dict['voxels'][:]
	data_dict.close()

	shape_number = voxels_all.shape[0]
	sampled_points = np.zeros((shape_number, batch_size_64, 4), np.int8)

	for index in range(shape_number):
		start_time = time.time()
		print(f'processing, {index}')

		voxels = voxels_all[index]
		voxels = voxels.reshape(64, 64, 64)

		points_value = get_points_from_vox(voxels)
		sampled_points[index, :, :] = points_value
		print('left time, ', (time.time() - start_time) * (shape_number - index) / 3600)

	hdf5_path = os.path.join(dataset_dir, 'voxel2pc.hdf5')
	hdf5_file = h5py.File(hdf5_path, 'w')
	hdf5_file.create_dataset("points", [shape_number, batch_size_64, 4], np.int8, compression=9)
	hdf5_file.create_dataset("voxels", [shape_number, 64, 64, 64, 1], np.int8, compression=9)
	hdf5_file["points"][:] = sampled_points
	hdf5_file["voxels"][:] = voxels_all
	hdf5_file.close()
	

def voxel2pc_single(dataset_dir):
	npz = np.load(dataset_dir + '/names.npz',  allow_pickle=True)
	names = npz["names"]

	# init kdtree
	shape_number = len(names)
	for index in range(shape_number):
		start_time = time.time()
		mesh_fn = names[index]
		print('processing, ', mesh_fn)
		pc_path = os.path.join(dataset_dir, 'shapes', mesh_fn, 'voxel_64.npz')

		voxels = np.load(pc_path)['voxels']
		voxels = voxels.reshape(64, 64, 64)
		#partialpc = partialpc*0.8 # scale the mesh

		points_value = get_points_from_vox(voxels)
		out_file = os.path.join(dataset_dir, 'shapes', mesh_fn, 'voxel2pc.npz')
		np.savez(out_file, points = points_value, names = mesh_fn)

		print('left time, ', (time.time() - start_time) * (len(names) - index) / 3600)
		
if __name__ == '__main__':
	dataset_dir = sys.argv[2]
	data_flag = int(sys.argv[1])
	if data_flag == 0:
		#voxel file npz
		voxel2pc_single(dataset_dir)
	else:
		#batch file hdf5
		voxel2pc_batch(dataset_dir)
