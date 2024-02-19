import trimesh
import numpy as np
import implicit_waterproofing as iw
import glob
import multiprocessing as mp
import argparse
import os
import traceback
import time
import sys
# sys.path.append('/local-scratch2/data_processing/my_utils')
# from pc_utils import *
from sklearn.neighbors import KDTree
import trimesh
import h5py
import alphashape

def sample_points(mesh, surface_points_num, outside_points_num, sigma):
	N = 256

	points = mesh.sample(surface_points_num)
	points = points + sigma * np.random.randn(points.shape[0], points.shape[1])
	samples = (np.random.randint(N, size=(outside_points_num, 3)) + 0.5)/N - 0.5
	points = np.concatenate([points, samples], 0)
	sdfs = iw.implicit_waterproofing(mesh, points)[0]
	im_points = np.concatenate([points, sdfs.reshape(-1, 1)], 1)
	return im_points

def sample_points_with_normal(mesh, surface_points_num):
	points, indexs = mesh.sample(surface_points_num, return_index=True)
	normals = mesh.face_normals[indexs]
	im_points = np.concatenate([points, normals], 1)
	return im_points

def sample_points_with_values(points, normals, sigma, near_points_num, outside_points_num):
	N = 256

	repeat = 8
	noise_64 = np.random.randn(repeat*points.shape[0])
	values_64 = noise_64 < 0
	noise_64 = noise_64.reshape(noise_64.shape[0], 1).repeat(3, axis=1)
	surface_points_repeat = points.repeat(repeat, axis=0)
	surface_normals_repeat = normals.repeat(repeat, axis=0)

	repeat_points = surface_points_repeat + sigma * surface_normals_repeat * noise_64
	
	# repeat_points = np.concatenate((points, points, points), axis=0)
	# repeat_points = repeat_points + sigma * np.random.randn(int(points.shape[0]*3), points.shape[1])
	samples = (np.random.randint(N, size=(outside_points_num, 3)) + 0.5)/N - 0.5
	#all_sampled_points = np.concatenate([repeat_points, samples], 0)

	all_sampled_points = samples

	tree = KDTree(points)
	dist, inds = tree.query(all_sampled_points, k=1)
	# print('all_sampled_points, ', all_sampled_points.shape)
	# print('inds, ', inds.shape)
	inds = inds.reshape(-1)
	dist = dist.reshape(-1)

	values = np.sum((all_sampled_points - points[inds, :]) * normals[inds, :], axis=1)
	#print('values, ', values)
	values = values < 0

	#set extra constraints for random sampled points
	#convex_mesh = trimesh.convex.convex_hull(points)
	alpha_shape = alphashape.alphashape(points, 1.5)

	convex_mesh = trimesh.base.Trimesh(alpha_shape.vertices, alpha_shape.faces)

	sdfs = iw.implicit_waterproofing(convex_mesh, all_sampled_points)[0]
	values = np.minimum(values.reshape(-1), sdfs.reshape(-1))

	sdfs_repeat = iw.implicit_waterproofing(convex_mesh, repeat_points)[0]
	values_64 = np.minimum(values_64.reshape(-1), sdfs_repeat.reshape(-1))

	all_sampled_points = all_sampled_points[dist > sigma, :]
	values = values[dist > sigma]

	while all_sampled_points.shape[0] < outside_points_num:
		all_sampled_points = np.concatenate([all_sampled_points, all_sampled_points], 0)
		values = np.concatenate([values, values], 0)
	all_sampled_points = all_sampled_points[:outside_points_num, :]
	values = values[:outside_points_num]

	final_p = np.concatenate((repeat_points, all_sampled_points), axis=0)
	final_v = np.concatenate((values_64.reshape(-1, 1), values.reshape(-1, 1)), axis=0)

	out_points = np.concatenate((final_p, final_v), axis=1)
	return out_points

def sample_points_with_values_near_neighbor_only(points, normals, sigma, near_points_num, outside_points_num):
	N = 256

	repeat_points = np.concatenate((points, points, points), axis=0)
	repeat_points = repeat_points + sigma * np.random.randn(int(points.shape[0]*3), points.shape[1])
	samples = (np.random.randint(N, size=(outside_points_num, 3)) + 0.5)/N - 0.5
	all_sampled_points = np.concatenate([repeat_points, samples], 0)

	repeat_num = 21
	tree = KDTree(points)
	dist, inds = tree.query(all_sampled_points, k=repeat_num)
	# print('all_sampled_points, ', all_sampled_points.shape)
	# print('inds, ', inds.shape)
	# inds = inds.reshape(-1)
	# dist = dist.reshape(-1)

	all_sampled_points_repeat = all_sampled_points.reshape(-1, 1, 3).repeat(repeat_num, axis=1)
	values = np.sum((all_sampled_points_repeat - points[inds, :]) * normals[inds, :], axis=2)
	#print('values, ', values)
	values = values < 0
	values = np.sum(values, axis=1)/repeat_num
	values = values > 0.5

	alpha_shape = alphashape.alphashape(points, 1.5)

	convex_mesh = trimesh.base.Trimesh(alpha_shape.vertices, alpha_shape.faces)

	sdfs = iw.implicit_waterproofing(convex_mesh, all_sampled_points)[0]
	values = np.minimum(values.reshape(-1), sdfs.reshape(-1))

	out_points = np.concatenate((all_sampled_points, values.reshape(-1, 1)), axis=1)
	return out_points

def farthest_sample_points_with_normal(mesh, surface_points_num):
	points, indexs = mesh.sample(100000, return_index=True)
	normals = mesh.face_normals[indexs]
	points, indexs = farthest_point_sample_with_index(points, surface_points_num)
	normals = normals[indexs, :]
	im_points = np.concatenate([points, normals], 1)
	return im_points

#@profile
def im_sampling(path):

	surface_points_num_16 = 16*16*16
	outside_points_num_16 = 16*16*2
	sigma_16 = 1/16
	surface_points_num_32 = 16*16*16*2 #8K
	outside_points_num_32 = 16*16*4
	sigma_32 = 1/32
	surface_points_num_64 = 16*16*16*6 #24K
	outside_points_num_64 = 16*16*8
	sigma_64 = 1/64

	batch_size_16 = surface_points_num_16 + outside_points_num_16
	batch_size_32 = surface_points_num_32 + outside_points_num_32
	batch_size_64 = surface_points_num_64 + outside_points_num_64

	models = np.load(path + '/test_names.npz')['test_names']

	data_path = os.path.join(path, 'shapes')

	shape_number = len(models)
	
	# out_dir = os.path.join(path, 'tmp_gt')
	# if not os.path.exists(out_dir):
	# 	os.makedirs(out_dir)

	hdf5_file_points_16 = np.zeros([shape_number,batch_size_16,4], np.float64)
	hdf5_file_points_32 = np.zeros([shape_number,batch_size_32,4], np.float64)
	hdf5_file_points_64 = np.zeros([shape_number,batch_size_64,4], np.float64)

	for i in range(shape_number):
		start_time = time.time()
		print(models[i])
		shape_path = os.path.join(data_path, models[i])
		file_name = os.path.join(shape_path, 'model_normalized.obj')
		
		mesh = trimesh.load(file_name)
		im_points_16 = sample_points(mesh, surface_points_num_16, outside_points_num_16, sigma_16)
		im_points_32 = sample_points(mesh, surface_points_num_32, outside_points_num_32, sigma_32)
		im_points_64 = sample_points(mesh, surface_points_num_64, outside_points_num_64, sigma_64)

		hdf5_file_points_16[i] = im_points_16
		hdf5_file_points_32[i] = im_points_32
		hdf5_file_points_64[i] = im_points_64
	
		print('left time: ', (time.time() - start_time)*(shape_number - i)/(3600))

	hdf5_path = os.path.join(path, 'ifnet_gt_points.hdf5')
	hdf5_file_out = h5py.File(hdf5_path, 'w')
	hdf5_file_out.create_dataset("points_16", [shape_number,batch_size_16,4], np.float64, compression=9)
	hdf5_file_out.create_dataset("points_32", [shape_number,batch_size_32,4], np.float64, compression=9)
	hdf5_file_out.create_dataset("points_64", [shape_number,batch_size_64,4], np.float64, compression=9)
	hdf5_file_out["points_16"][:] = hdf5_file_points_16
	hdf5_file_out["points_32"][:] = hdf5_file_points_32
	hdf5_file_out["points_64"][:] = hdf5_file_points_64
	hdf5_file_out.close()
	print('Finished {}'.format(path))

def normal_sampling(path, read_flag):

	surface_points = 8192

	surface_points_num_64 = 16*16*16*2*8 #64K
	outside_points_num_64 = 8192
	sigma_64 = 1/64
	batch_size_64 = surface_points_num_64 + outside_points_num_64

	models = np.load(path + '/test_names.npz')['test_names']

	data_path = os.path.join(path, 'shapes')

	shape_number = len(models)

	#hdf5_file_surface_points = np.zeros([shape_number,batch_size_64,4], np.float64)

	out_dir = os.path.join(path, 'tmp_along_normal')
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	if read_flag:
		input_file = os.path.join(path, 'pcr_test_8192.hdf5')
		data_dict = h5py.File(input_file)
		all_points_normal = data_dict['points'][:]
		data_dict.close()

	hdf5_file_points_64 = np.zeros([shape_number,batch_size_64,4], np.float64)

	for i in range(shape_number):
		start_time = time.time()
		print(models[i])
		shape_path = os.path.join(data_path, models[i])
		file_name = os.path.join(shape_path, 'model.obj')
		
		if read_flag:
			points_normal = all_points_normal[i]
		else:
			mesh = trimesh.load(file_name)
			points_normal = sample_points_with_normal(mesh, surface_points)

		data_points = sample_points_with_values(points_normal[:, :3], points_normal[:, 3:], 
			sigma_64, surface_points_num_64, outside_points_num_64)

		hdf5_file_points_64[i] = data_points

		# new_points = data_points[:, :3]
		# values = data_points[:, 3]
		# colors = np.zeros(new_points.shape)
		# colors[values < 1, 0] = 1

		# out_file = os.path.join(out_dir, f'{models[i]}.obj')
		# save_obj_data_with_color(out_file, vertex=new_points, face = np.array([]), colors=colors)

		print('left time: ', (time.time() - start_time)*(shape_number - i)/(3600))

		# out_file = os.path.join(shape_path, 'surface_points_8192.npz')
		# os.remove(os.path.join(shape_path, 'surface_points_8192*8.npz'))
		# np.savez(out_file, points = im_points_64, names = models[i])

	hdf5_path = os.path.join(path, 'pcr_occ_data.hdf5')
	hdf5_file_out = h5py.File(hdf5_path, 'w')
	hdf5_file_out.create_dataset("points", [shape_number,batch_size_64,4], np.float64, compression=9)
	hdf5_file_out["points"][:] = hdf5_file_points_64
	hdf5_file_out.close()

	print('Finished {}'.format(path))

if __name__ == '__main__':
	ROOT = sys.argv[1]
	sample_flag = int(sys.argv[2])
	read_flag = int(sys.argv[3])
	print('path, ', ROOT)
	if sample_flag == 0:
		print('gt mesh points sampling')
		im_sampling(ROOT)
	elif sample_flag == 1:
		print('surface points sampling')
		normal_sampling(ROOT, read_flag)
