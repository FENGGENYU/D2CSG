#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import skimage.measure
import time
import torch
import utils.utils
import os
import mcubes
import pymesh
from torch.autograd import Variable
#from utils.bspt import digest_bsp, get_mesh, get_mesh_watertight
import trimesh
import math

def create_mesh_mc(
	decoder, surface_points, phase, quads, filename, N=256, max_batch=32 ** 3, threshold=0.5, sample_flag=False, mc_shape =0 
):
	start = time.time()
	ply_filename = filename
	decoder.eval()
	#find neighbor cells for each sample point

	# NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
	voxel_origin = [0, 0, 0]#[-0.51, -0.51, -0.51]
	voxel_size = 1#1.02 / (N - 1)

	overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
	samples = torch.zeros(N ** 3, 6) # x y z sdf cell_num
	# p *5 
	# transform first 3 columns
	# to be the x, y, z index
	samples[:, 2] = overall_index % N
	samples[:, 1] = (overall_index.long() / N) % N
	samples[:, 0] = ((overall_index.long() / N) / N) % N

	# transform first 3 columns
	# to be the x, y, z coordinate
	samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
	samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
	samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

	samples[:, :3] = (samples[:, :3]+0.5)/N-0.5

	num_samples = N ** 3

	samples.requires_grad = False

	#sdf_box_inside_block = torch.zeros(num_samples)
	head = 0
	
	while head < num_samples:
		sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

		assign_matrix, s1, s2, s3, s4 = decoder(surface_points, sample_subset.unsqueeze(0), quads, phase)
		if mc_shape == 1:
			samples[head : min(head + max_batch, num_samples), 3] = (
				s1.reshape(-1)
				.detach()
				.cpu()
			)
		elif mc_shape == 2:
			samples[head : min(head + max_batch, num_samples), 3] = (
				s2.reshape(-1)
				.detach()
				.cpu()
			)
		else:
			samples[head : min(head + max_batch, num_samples), 4] = (
				s1.reshape(-1)
				.detach()
				.cpu()
			)
			samples[head : min(head + max_batch, num_samples), 5] = (
				s2.reshape(-1)
				.detach()
				.cpu()
			)
			samples[head : min(head + max_batch, num_samples), 3] = (
				s3.reshape(-1)
				.detach()
				.cpu()
			)
		head += max_batch

	#samples[:, 3] = samples[:, 3]/samples[:, 4]
	if mc_shape == 1 or mc_shape == 2:
		samples[:, 3] = (samples[:, 3] < 0.01).float()
	#samples[:, 3] = torch.min((samples[:, 4] < 0.01).float(), (samples[:, 5] > 0.01).float())
	if phase > 0:
		if torch.sum((samples[:, 5] > 0.01).float()) > 0:
			samples[:, 3] = 1 - torch.max(samples[:, 4], 0.2 - samples[:, 5])
		else:
			samples[:, 3] = 1 - samples[:, 4]

	sdf_values = samples[:, 3]
	
	sdf_values = sdf_values.reshape(N, N, N)
	np.savez('./sdfs.npz', points = samples.numpy())
			
	end = time.time()
	print("sampling takes: %f" % (end - start))

	numpy_3d_sdf_tensor = sdf_values.numpy()
	print('numpy_3d_sdf_tensor, ', np.sum(numpy_3d_sdf_tensor))
	# verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
	#	numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
	# )

	verts, faces = mcubes.marching_cubes(numpy_3d_sdf_tensor, threshold)
	print('verts, ', verts)
	mesh_points = verts
	
	mesh_points = (mesh_points+0.5)/N-0.5

	# if not shapenet:
	#	mesh_points = mesh_points * 1.25
	# try writing to the ply file
	if sample_flag:
		mesh = trimesh.Trimesh(mesh_points, faces)
		points, indexs = mesh.sample(16384, return_index=True)
		normals = mesh.face_normals[indexs]
		utils.utils.write_ply_point_normal(ply_filename + "_pc_normal.ply", points, normals)
		
	if not os.path.exists(os.path.dirname(ply_filename)):
		os.makedirs(os.path.dirname(ply_filename))

	utils.utils.write_ply_triangle(ply_filename + ".ply", mesh_points, faces)
	#output seg results
	seg_results = False
	if seg_results:
		_, indices = torch.max(assign_matrix.squeeze(), dim=1)
		output_points = surface_points.squeeze().cpu().numpy()
		output_points = output_points[:, :3]
		all_colors = (np.random.random_sample((assign_matrix.size(2), 3))*255).astype(np.int8)
		indices = indices.cpu().numpy()
		print('indices, ', np.unique(indices))
		print('indices, ', len(np.unique(indices)))
		colors = all_colors[indices, :]
		utils.utils.write_ply_point_color(ply_filename + "_seg.ply", output_points, colors)


def pass_generator(decoder, samples, surface_points, quads, c_dim, phase=2, max_batch=int(2 ** 14)):
	samples = torch.from_numpy(samples)
	quads = torch.from_numpy(quads)

	samples.requires_grad = False
	model_float = torch.ones((samples.shape[0], c_dim))
	num_samples = samples.shape[0]
	head = 0
	while head < num_samples:
		sample_subset = samples[head : min(head + max_batch, num_samples), :].cuda()
		_, _, _, h3, h2 = decoder(surface_points, sample_subset.unsqueeze(0), quads.cuda(), phase=phase)
		#print('h2,', h2.shape)
		#print('model_float,', model_float.shape)
		#h2, inside 1, outside 0-1
		model_float[head : min(head + max_batch, num_samples), :] = h2.reshape(-1, c_dim).detach().cpu()
		head += max_batch
	return model_float.numpy()


def pass_generator_and_sample(decoder, samples, surface_points, quads, N, c_dim, threshold=0.5, phase=2, max_batch=int(2 ** 14)):
	samples = torch.from_numpy(samples)
	quads = torch.from_numpy(quads)
	samples.requires_grad = False
	model_float = torch.ones(N ** 3, 3)
	num_samples = N ** 3
	head = 0
	while head < num_samples:
		sample_subset = samples[head : min(head + max_batch, num_samples), :].cuda()
		_, h1, h0, h3, h2 = decoder(surface_points, sample_subset.unsqueeze(0), quads.cuda(), phase=phase)
		model_float[head : min(head + max_batch, num_samples), 0] = h3.reshape(-1).detach().cpu()
		model_float[head : min(head + max_batch, num_samples), 1] = h1.reshape(-1).detach().cpu()
		model_float[head : min(head + max_batch, num_samples), 2] = h0.reshape(-1).detach().cpu()
		head += max_batch

	#model_float[:, 0] = torch.min((model_float[:, 1] < 0.01).float(), (model_float[:, 2] > 0.01).float())
	#model_float[:, 0] = 1 - torch.max(model_float[:, 1], 0.2 - model_float[:, 2])
	if torch.sum((model_float[:, 2] > 0.01).float()) > 0:
		model_float[:, 0] = 1 - torch.max(model_float[:, 1], 0.2 - model_float[:, 2])
	else:
		model_float[:, 0] = 1 - model_float[:, 1]
	sdf_values = model_float[:, 0]
	sdf_values = sdf_values.reshape(N, N, N)

	verts, faces = mcubes.marching_cubes(sdf_values.numpy(), threshold)

	mesh_points = verts
	
	mesh_points = (mesh_points+0.5)/N-0.5

	mesh = trimesh.Trimesh(mesh_points, faces)
	if len(mesh.vertices) == 0:
		return None
	points, indexs = mesh.sample(50000, return_index=True)

	sigma = 1/64
	points = points + sigma * (np.random.random_sample((points.shape[0], points.shape[1]))*2 - 1)
	print(points.dtype)
	return points.astype(np.float32)


def mask_generator_new(model_float, convex_mask, c_dim_half, differ=True, print_flag=False):
	#model_float = np.expand_dims(model_float, axis=0)
	#model_float = model_float.unsqueeze(0)
	#h3_1 = np.amin(model_float[:,:,:c_dim_half], axis=2, keepdims=True)
	#h3_0 = np.amin(model_float[:,:,c_dim_half:], axis=2, keepdims=True)
	if differ:
		model_float = model_float * (1 - convex_mask)
		# h3_1 = np.amax(model_float[:,:,:c_dim_half]-convex_mask[:,:,:c_dim_half], axis=2, keepdims=True)
		# h3_0 = np.amax(model_float[:,:,c_dim_half:]-convex_mask[:,:,c_dim_half:], axis=2, keepdims=True)
		# if print_flag:
		# 	print('h3_1, ', h3_1[:, 20040, :])
		# 	print('h3_0, ', h3_0[:, 20040, :])

		# h301 = np.minimum(h3_1, 1-h3_0)
	else:
		h301 = np.amax(model_float-convex_mask, axis=2, keepdims=True)
	#zmG = 1-h301
	return model_float


def get_surface_grid(points, plane_m, quadric=True):
	plane_m = torch.from_numpy(plane_m)
	#points = torch.from_numpy(points)
	points = torch.from_numpy(points).float().requires_grad_(True)
	
	ones = torch.ones((points.shape[0], points.shape[1], 1))
	if quadric:
		pointsx = torch.cat([(points)**2, points, ones], 2)
	else:
		pointsx = torch.cat([points, ones], 2)
	h1 = torch.matmul(pointsx, plane_m)
	#h1 = h1.clamp(min=-100, max=100)
	#h1.backward(torch.ones((h1.shape[0], h1.shape[1], h1.shape[2])))
	#h1_grad = torch.autograd.grad(h1, points)
	#h1_grad = tf.gradients(h1, points)[0]

	return h1.detach().numpy(), None

def get_samples(N):
	voxel_origin = [0, 0, 0]
	overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
	voxel_size = 1

	samples = torch.ones(N ** 3, 3)

	# transform first 3 columns
	# to be the x, y, z index
	samples[:, 2] = overall_index % N
	samples[:, 1] = (overall_index.long() / N) % N
	samples[:, 0] = ((overall_index.long() / N) / N) % N

	# transform first 3 columns
	# to be the x, y, z coordinate
	samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
	samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
	samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

	samples = (samples+0.5)/N-0.5

	return samples.numpy()


def dualcsg_mask(decoder, primitives, 
	connection_t_convex, connection_t_concave, 
	p_dim=4096, c_dim=256, N=256, max_batch=32 ** 3, 
	quadric=True, differ=True, unused_convex=None,
	filter_threshold = 0, concave_level_set = 0.01
):
	surface_points = None
	start_time = time.time()

	original_plane = np.copy(primitives.detach().cpu().numpy())

	# connection_t_convex = connection_t_convex.unsqueeze(0)
	# connection_t_concave = connection_t_concave.unsqueeze(0)

	quads = primitives.detach().cpu().numpy()
	
	mc_res = 64
	sampling_threshold = 0.9 #0.5?
	samples_grid = get_samples(N)
	samples = pass_generator_and_sample(decoder, samples_grid, surface_points, quads, N, c_dim, \
		threshold = sampling_threshold, phase=3)
	if samples is None:
		print('samples none')
		return connection_t_convex, connection_t_concave, unused_convex, True

	#samples = samples_points.detach().cpu().numpy()
	samples_half = get_samples(mc_res)

	#sdf to occ
	quads = quads*( (np.sum(np.abs(quads[:,0:3,:]), axis=1, keepdims=True)>1e-3) | (np.sum(np.abs(quads[:,3:6,:]), axis=1, keepdims=True)>1e-3) ).astype(np.float32)
	quads = quads/np.maximum(np.sqrt(np.sum(np.square(quads[:,0:6,:]), axis=1, keepdims=True)),1e-6)
	quads = quads*1e4

	#normalize
	print('model float time:, ', time.time() - start_time)
	#original_plane = np.copy(plane_m)
	
	model_float = pass_generator(decoder, samples, surface_points, quads, c_dim, phase=3)
	#model float, 
	model_float = (model_float > sampling_threshold).astype(np.uint8)
	model_float_test = np.copy(model_float)

	c_dim_half = c_dim//2
	if differ:
		model_float_sum = np.minimum(np.max(model_float[:,:c_dim_half],axis=1),1-np.max(model_float[:,c_dim_half:],axis=1))
	else:
		model_float_sum = np.max(model_float,axis=1)

	for i in range(c_dim):
		if np.max(model_float[:,i])>0:
			if unused_convex[i] == 0:
				#print('check, ', i)
				model_float_test[:,i] = 0
				if differ:
					t_output = np.minimum(np.max(model_float_test[:,:c_dim_half],axis=1),1-np.max(model_float_test[:,c_dim_half:],axis=1))
					varied_output = (model_float_sum != t_output).astype(int)
					changed_point_num = np.sum(varied_output)
					if changed_point_num > filter_threshold:
						model_float_test[:,i] = model_float[:,i]
						unused_convex[i] = 0
						#print('keep, ', i)
					else:
						model_float[:,i] = 0
						unused_convex[i] = 1
						model_float_sum = t_output
				else:
					if np.any(np.max(model_float_test,axis=1)!=model_float_sum):
						model_float_test[:,i] = model_float[:,i]
						unused_convex[i] = 0
						#print('keep, ', i)
					else:
						model_float[:,i] = 0
						unused_convex[i] = 1
		else:
			unused_convex[i] = 1
	print('convex num, ', c_dim - torch.sum(unused_convex))
	
	convex_mask = unused_convex.reshape(1, -1).cpu().numpy()

	for i in range(c_dim):
		if i<c_dim/2:
			if unused_convex[i]:
				connection_t_convex[:,i] = 0
		else:
			if unused_convex[i]:
				connection_t_concave[:,i-int(c_dim/2)] = 0

	print('remove convex time:, ', time.time() - start_time)

	zg = pass_generator(decoder, samples, surface_points, quads, c_dim, phase=3)
	correct_output = mask_generator_new(zg, convex_mask, c_dim_half, differ)
	correct_output = correct_output > sampling_threshold

	primitive_count = 0
	primitive_count_plane = 0
	primitive_count_quads = 0

	connection_t_temp_convex = (connection_t_convex>0.01)#.squeeze()
	connection_t_temp_convex = np.sum(connection_t_temp_convex.detach().cpu().numpy(), axis=1)
	connection_t_temp_convex = np.argwhere(connection_t_temp_convex > 0)
	print('convex primi needs to be checked, ', len(connection_t_temp_convex))
	plane_m_test = np.copy(quads)
	
	left_convex = np.minimum(np.max(correct_output[:,:c_dim_half],axis=1),1-np.max(correct_output[:,c_dim_half:],axis=1))
	
	for i in connection_t_temp_convex:
		plane_m_test[0,:,i] = 0
		zg = pass_generator(decoder, samples, surface_points, plane_m_test, c_dim, phase=3)
		this_output = mask_generator_new(zg, convex_mask, c_dim_half, differ)
		this_output = this_output>sampling_threshold
		#print('correct_output, ', correct_output.shape)
		
		t_left_convex = np.minimum(np.max(this_output[:,:c_dim_half],axis=1),1-np.max(this_output[:,c_dim_half:],axis=1))
	
		varied_output = (t_left_convex^left_convex).astype(int)
		changed_point_num = np.sum(varied_output)
		
		if changed_point_num > filter_threshold:
			#print('changed_point_num, ', changed_point_num)
			plane_m_test[0,:,i] = quads[0,:,i]
			primitive_count_plane += 1
		else:
			quads[0,:,i] = 0
			left_convex = t_left_convex#np.logical_or(correct_output, this_output)

	connection_t_temp_concave = (connection_t_concave>0.01)#.squeeze()
	connection_t_temp_concave = np.sum(connection_t_temp_concave.detach().cpu().numpy(), axis=1)
	connection_t_temp_concave = np.argwhere(connection_t_temp_concave > 0)
	#plane_m_test = np.copy(plane_m)
	print('concave primi needs to be checked, ', len(connection_t_temp_concave))
	#right_convex = np.minimum(np.max(correct_output[:,:c_dim_half],axis=1),1-np.max(correct_output[:,c_dim_half:],axis=1))
	quads_test = np.copy(quads)
	right_convex = left_convex

	for i in connection_t_temp_concave:
		quads_test[0,:,i+int(p_dim/2)] = 0
		#zg, h2
		zg = pass_generator(decoder, samples, surface_points, quads_test, c_dim, phase=3)
		# if i == 166:
		# 	this_output = mask_generator(zg, convex_mask, c_dim_half, print_flag=True)
		# else:
		this_output = mask_generator_new(zg, convex_mask, c_dim_half, differ)
		this_output = this_output>sampling_threshold

		t_right_convex = np.minimum(np.max(this_output[:,:c_dim_half],axis=1),1-np.max(this_output[:,c_dim_half:],axis=1))
	
		varied_output = (t_right_convex^right_convex).astype(int)
		changed_point_num = np.sum(varied_output)
		
		# ttt = (correct_output^this_output)
		# if i == 166:
		# 	print('ttt', ttt.shape)
		# 	print('sample indexes', np.argwhere(ttt>0))
		# 	print('correct_output, ', correct_output[:, 20040, :])
		# 	print('this_output, ', this_output[:, 20040, :])
		# 	print('samples,', samples[20040, :])
		if changed_point_num > filter_threshold:
			#print('changed_point_num, ', changed_point_num)
			quads_test[0,:,i+int(p_dim/2)] = quads[0,:,i+int(p_dim/2)]
			primitive_count_quads += 1
		else:
			quads[0,:,i+int(p_dim/2)] = 0
			#correct_output = np.logical_or(correct_output, this_output)
			right_convex = t_right_convex
	
	print('primitive_count_plane, ', primitive_count_plane)
	print('primitive_count_quads, ', primitive_count_quads)
	
	# if c_dim - np.sum(unused_convex) > 10:
	# 	return 0
	print('remove plane time:, ', time.time() - start_time)
	#exit(0)
	# for i in connection_t_temp:
	# 	if i < p_dim:
	# 		if plane_norm[0, i] < 2:
	# 			plane_m[0,:,i] = 0

	connection_t_convex = (connection_t_convex>0.01).float()
	connection_t_concave = (connection_t_concave>0.01).float()
	for i in range(p_dim):
		if np.all(quads[0,:,i]==0):
			if i < p_dim/2:
				connection_t_convex[i,:] = 0
			else:
				connection_t_concave[i - int(p_dim/2),:] = 0
	p_count_before = len(connection_t_temp_concave) + len(connection_t_temp_convex)
	print(f'primi count before filtering {p_count_before}')
	#c_count = c_dim - np.sum(unused_convex)
	p_count_after = primitive_count_plane + primitive_count_quads
	print(f'primi count after filtering {p_count_after}')
	if p_count_before - p_count_after < 3:
		stop_flag = True
	else:
		stop_flag = False
	return connection_t_convex, connection_t_concave, unused_convex, stop_flag


def create_mesh_dualcsg_mask(decoder, primitives, 
	connection_t_convex, connection_t_concave, unused_convex, filename, file_type='obj', 
	p_dim=4096, c_dim=256, N=256, max_batch=32 ** 3, primi_count=False,
	filter_threshold = 0, concave_level_set = 0.01, primi_out = False,scad_flag=False
):
	surface_points = None
	differ=True
	quadric=True
	start_time = time.time()
	ply_filename = filename

	# connection_t_convex = connection_t_convex.unsqueeze(0)
	# connection_t_concave = connection_t_concave.unsqueeze(0)

	original_plane = np.copy(primitives.detach().cpu().numpy())
	quads = primitives.detach().cpu().numpy()

	if primi_out:
		if (not os.path.isdir(ply_filename)) and (not primi_count):
			os.makedirs(ply_filename)
	
	#high res for filtering
	sampling_threshold = 0.9 #0.5?
	samples_grid = get_samples(N)
	samples = pass_generator_and_sample(decoder, samples_grid, surface_points, quads,\
		N, c_dim, threshold = sampling_threshold, phase=3)
	if samples is None:
		print('samples none')
	#sdf to occ
	quads = quads*( (np.sum(np.abs(quads[:,0:3,:]), axis=1, keepdims=True)>1e-3) | (np.sum(np.abs(quads[:,3:6,:]), axis=1, keepdims=True)>1e-3) ).astype(np.float32)
	quads = quads/np.maximum(np.sqrt(np.sum(np.square(quads[:,0:6,:]), axis=1, keepdims=True)),1e-6)
	quads = quads*1e4

	#normalize
	print('model float time:, ', time.time() - start_time)
	#original_plane = np.copy(plane_m)

	model_float = pass_generator(decoder, samples, surface_points, quads, c_dim, phase=3)
	#model float, 
	model_float = (model_float > sampling_threshold).astype(np.uint8)
	model_float_test = np.copy(model_float)

	c_dim_half = c_dim//2
	if differ:
		model_float_sum = np.minimum(np.max(model_float[:,:c_dim_half],axis=1),1-np.max(model_float[:,c_dim_half:],axis=1))
	else:
		model_float_sum = np.max(model_float,axis=1)
	
	#unused_convex = np.ones([c_dim], np.uint8)

	for i in range(c_dim):
		if np.max(model_float[:,i])>0:
			if unused_convex[i] == 0:
				#print('check, ', i)
				model_float_test[:,i] = 0
				if differ:
					t_output = np.minimum(np.max(model_float_test[:,:c_dim_half],axis=1),1-np.max(model_float_test[:,c_dim_half:],axis=1))
					varied_output = (model_float_sum != t_output).astype(int)
					changed_point_num = np.sum(varied_output)
					if changed_point_num > filter_threshold:
						model_float_test[:,i] = model_float[:,i]
						unused_convex[i] = 0
						#print('keep, ', i)
					else:
						model_float[:,i] = 0
						unused_convex[i] = 1
						model_float_sum = t_output
				else:
					if np.any(np.max(model_float_test,axis=1)!=model_float_sum):
						model_float_test[:,i] = model_float[:,i]
						unused_convex[i] = 0
						#print('keep, ', i)
					else:
						model_float[:,i] = 0
						unused_convex[i] = 1
		else:
			unused_convex[i] = 1
	#unused_convex[50] = 1
	print('convex num, ', c_dim - torch.sum(unused_convex))
	
	convex_mask = unused_convex.reshape(1, -1).cpu().numpy()

	for i in range(c_dim):
		if i<c_dim/2:
			if unused_convex[i]:
				connection_t_convex[:,i] = 0
		else:
			if unused_convex[i]:
				connection_t_concave[:,i-int(c_dim/2)] = 0

	print('remove convex time:, ', time.time() - start_time)

	zg = pass_generator(decoder, samples, surface_points, quads, c_dim, phase=3)
	correct_output = mask_generator_new(zg, convex_mask, c_dim_half, differ)
	correct_output = correct_output > sampling_threshold

	primitive_count = 0
	primitive_count_plane = 0
	primitive_count_quads = 0

	connection_t_temp_convex = (connection_t_convex>0.01)#.squeeze()
	connection_t_temp_convex = np.sum(connection_t_temp_convex.detach().cpu().numpy(), axis=1)
	connection_t_temp_convex = np.argwhere(connection_t_temp_convex > 0)
	print('convex primi needs to be checked, ', len(connection_t_temp_convex))
	plane_m_test = np.copy(quads)
	#left_convex = np.amax(correct_output[:,:c_dim_half],axis=1)
	left_convex = np.minimum(np.max(correct_output[:,:c_dim_half],axis=1),1-np.max(correct_output[:,c_dim_half:],axis=1))
	
	for i in connection_t_temp_convex:
		plane_m_test[0,:,i] = 0
		zg = pass_generator(decoder, samples, surface_points, plane_m_test, c_dim, phase=3)
		this_output = mask_generator_new(zg, convex_mask, c_dim_half, differ)
		this_output = this_output>sampling_threshold
		#print('correct_output, ', correct_output.shape)
		
		t_left_convex = np.minimum(np.max(this_output[:,:c_dim_half],axis=1),1-np.max(this_output[:,c_dim_half:],axis=1))
	
		varied_output = (t_left_convex^left_convex).astype(int)
		changed_point_num = np.sum(varied_output)
		
		if changed_point_num > filter_threshold:
			#print('changed_point_num, ', changed_point_num)
			plane_m_test[0,:,i] = quads[0,:,i]
			primitive_count_plane += 1
		else:
			quads[0,:,i] = 0
			left_convex = t_left_convex#np.logical_or(correct_output, this_output)

	connection_t_temp_concave = (connection_t_concave>0.01)#.squeeze()
	connection_t_temp_concave = np.sum(connection_t_temp_concave.detach().cpu().numpy(), axis=1)
	connection_t_temp_concave = np.argwhere(connection_t_temp_concave > 0)
	#plane_m_test = np.copy(plane_m)
	print('concave primi needs to be checked, ', len(connection_t_temp_concave))
	#right_convex = np.minimum(np.max(correct_output[:,:c_dim_half],axis=1),1-np.max(correct_output[:,c_dim_half:],axis=1))
	right_convex = left_convex
	quads_test = np.copy(quads)

	for i in connection_t_temp_concave:
		quads_test[0,:,i+int(p_dim/2)] = 0
		#zg, h2
		zg = pass_generator(decoder, samples, surface_points, quads_test, c_dim, phase=3)
		# if i == 166:
		# 	this_output = mask_generator(zg, convex_mask, c_dim_half, print_flag=True)
		# else:
		this_output = mask_generator_new(zg, convex_mask, c_dim_half, differ)
		this_output = this_output>sampling_threshold

		t_right_convex = np.minimum(np.max(this_output[:,:c_dim_half],axis=1),1-np.max(this_output[:,c_dim_half:],axis=1))
	
		varied_output = (t_right_convex^right_convex).squeeze().astype(int)
		changed_point_num = np.sum(varied_output)
		
		# ttt = (correct_output^this_output)
		# if i == 166:
		# 	print('ttt', ttt.shape)
		# 	print('sample indexes', np.argwhere(ttt>0))
		# 	print('correct_output, ', correct_output[:, 20040, :])
		# 	print('this_output, ', this_output[:, 20040, :])
		# 	print('samples,', samples[20040, :])
		if changed_point_num > filter_threshold:
			#print('changed_point_num, ', changed_point_num)
			quads_test[0,:,i+int(p_dim/2)] = quads[0,:,i+int(p_dim/2)]
			primitive_count_quads += 1
		else:
			quads[0,:,i+int(p_dim/2)] = 0
			#correct_output = np.logical_or(correct_output, this_output)
			right_convex = t_right_convex
	
	print('primitive_count_plane, ', primitive_count_plane)
	print('primitive_count_quads, ', primitive_count_quads)
	
	# if c_dim - np.sum(unused_convex) > 10:
	# 	return 0
	print('remove plane time:, ', time.time() - start_time)
	#exit(0)
	# for i in connection_t_temp:
	# 	if i < p_dim:
	# 		if plane_norm[0, i] < 2:
	# 			plane_m[0,:,i] = 0

	connection_t_convex = (connection_t_convex>0.01).float()
	connection_t_concave = (connection_t_concave>0.01).float()
	for i in range(p_dim):
		if np.all(quads[0,:,i]==0):
			if i < p_dim/2:
				connection_t_convex[i,:] = 0
			else:
				connection_t_concave[i - int(p_dim/2),:] = 0

	if primi_count:
		return c_dim - torch.sum(unused_convex), primitive_count_plane, primitive_count_quads, connection_t_convex, connection_t_concave, unused_convex

	#low res for mesh
	mc_res = 64
	samples_half = get_samples(mc_res)

	#csg
	print('csg start time:, ', time.time() - start_time)
	dimf = mc_res
	model_float = np.full([dimf+2,dimf+2,dimf+2],100,np.float32)
	output_mesh = None
	concave_mesh_left = None
	concave_mesh_right = None
	if primi_out:
		primi_dir = ply_filename + '/primi_for_connection' 
		primi_dir2 = ply_filename + '/primi_for_render'
		convex_dir = ply_filename + '/convex'
		concave_dir = ply_filename + '/concave'

		if not os.path.isdir(primi_dir):
			os.makedirs(primi_dir)
		if not os.path.isdir(primi_dir2):
			os.makedirs(primi_dir2)
		if not os.path.isdir(convex_dir):
			os.makedirs(convex_dir)
		if not os.path.isdir(concave_dir):
			os.makedirs(concave_dir)
	if scad_flag:
		numpy_file = ply_filename + '/csg.npz'
		print(f'scad output: {numpy_file}')
		np.savez(numpy_file, unused_convex = unused_convex.cpu().numpy(), left_ops = connection_t_convex.cpu().numpy(), right_ops = connection_t_concave.cpu().numpy())
	#convex_points = np.zeros((int(c_dim - np.sum(unused_convex)), 8192, 3))
	#unused_convex[182] = 0

	c_index = 0
	for i in range(c_dim):
		if unused_convex[i]==0:
			print('convex %d, time: %f'%(i, time.time() - start_time))
			
			convex_mesh = None
			for j in range(p_dim):
				if (i<c_dim/2 and j<p_dim/2 and connection_t_convex[j,i]>0) or \
					(i>=c_dim/2 and j>=p_dim/2 and connection_t_concave[j-int(p_dim/2),i-int(c_dim/2)]>0):
				#if connection_t[0,j,i]>0:
					#print(f'{i}, {j}, {original_plane[:,:,j:j+1]}, {plane_norm[:, j:j+1]}')
					model_out, _ = get_surface_grid(np.expand_dims(samples_half, axis=0), original_plane[:,:,j:j+1], quadric)
					#with boundary closed
					model_float[1:-1,1:-1,1:-1] = np.reshape(model_out, [dimf,dimf,dimf])
					if i<c_dim_half:
						vertices, triangles = mcubes.marching_cubes(-model_float, -0)
					else:
						# if i == 57:
						# 	vertices, triangles = mcubes.marching_cubes(-model_float, -0.2)
						# else:
						vertices, triangles = mcubes.marching_cubes(-model_float, -concave_level_set)

					vertices = (vertices-0.5)/dimf-0.5 #+ np.array([0.01, 0.01, 0]).reshape(-1, 3)
					
					if convex_mesh is None:
						convex_mesh = pymesh.form_mesh(vertices, triangles)
					else:
						convex_mesh = pymesh.boolean(convex_mesh, pymesh.form_mesh(vertices, triangles), operation="intersection")

					if primi_out:
						#open boundary
						model_out = np.reshape(model_out, [dimf,dimf,dimf])
						if i<c_dim_half:
							vertices, triangles = mcubes.marching_cubes(-model_out, -0)
						else:
							#vertices, triangles = mcubes.marching_cubes(-model_out, -0.001)
							vertices, triangles = mcubes.marching_cubes(-model_out, -0)

						vertices = (vertices+0.5)/dimf-0.5

						#pymesh.meshio.save_mesh_raw(filename, vertices, faces)

						pymesh.meshio.save_mesh_raw(primi_dir +"/"+str(i)+ "_"+ str(j) +"." + file_type, vertices, triangles)
						pymesh.meshio.save_mesh_raw(primi_dir2 +"/" + str(j) +"." + file_type, vertices, triangles)
			if scad_flag:
				continue
			if primi_out:		
				pymesh.meshio.save_mesh_raw(convex_dir +"/"+str(i)+"." + file_type, convex_mesh.vertices, convex_mesh.faces)
			
			c_index += 1

			if i < c_dim_half:
				if concave_mesh_left is None:
					concave_mesh_left = convex_mesh
				else:
					concave_mesh_left = pymesh.boolean(concave_mesh_left, convex_mesh, operation="union")
			else:
				if concave_mesh_right is None:
					concave_mesh_right = convex_mesh
				else:
					concave_mesh_right = pymesh.boolean(concave_mesh_right, convex_mesh, operation="union")	
	if scad_flag:
		return
	if differ:
		if concave_mesh_left is None:
			if concave_mesh_right is None:
				print('empty mesh')
				exit()
			else:
				output_mesh = concave_mesh_right
				if primi_out:
					pymesh.meshio.save_mesh_raw(concave_dir +"/right"+"." + file_type, concave_mesh_right.vertices, concave_mesh_right.faces)

		else:
			if concave_mesh_right is None:
				output_mesh = concave_mesh_left
				if primi_out:
					pymesh.meshio.save_mesh_raw(concave_dir +"/left"+"." + file_type, concave_mesh_left.vertices, concave_mesh_left.faces)

			else:
				output_mesh = pymesh.boolean(concave_mesh_left, concave_mesh_right, operation="difference")
				if primi_out:
					pymesh.meshio.save_mesh_raw(concave_dir +"/left"+"." + file_type, concave_mesh_left.vertices, concave_mesh_left.faces)
					pymesh.meshio.save_mesh_raw(concave_dir +"/right"+"." + file_type, concave_mesh_right.vertices, concave_mesh_right.faces)
	else:
		output_mesh = pymesh.boolean(concave_mesh_left, concave_mesh_right, operation="union")
		if primi_out:
			pymesh.meshio.save_mesh_raw(concave_dir +"/left"+"." + file_type, concave_mesh_left.vertices, concave_mesh_left.faces)
			pymesh.meshio.save_mesh_raw(concave_dir +"/right"+"." + file_type, concave_mesh_right.vertices, concave_mesh_right.faces)

	if primi_out:
		file_name = os.path.basename(ply_filename)
		pymesh.meshio.save_mesh_raw(os.path.join(ply_filename, file_name + "." + file_type), output_mesh.vertices, output_mesh.faces)
		print('primitive meshing done')
	else:
		dirname = os.path.dirname(ply_filename)
		if not os.path.exists(dirname):
			os.makedirs(dirname)

		pymesh.meshio.save_mesh_raw(ply_filename + "." + file_type, output_mesh.vertices, output_mesh.faces)
		print('csg meshing done')
		
def create_mesh_csg_primi_compact_one_primi(decoder, surface_points, samples_points, primitives, 
	connection_t_convex, connection_t_concave, filename, file_type='obj', 
	p_dim=4096, c_dim=256, N=256, max_batch=32 ** 3, primi_count=False, quadric=True, differ=True, 
	filter_threshold = 0, concave_level_set = 0.01, primi_out = False
):
	start_time = time.time()
	ply_filename = filename

	original_plane = np.copy(primitives.detach().cpu().numpy())

	connection_t_convex = connection_t_convex.unsqueeze(0)
	connection_t_concave = connection_t_concave.unsqueeze(0)

	quads = primitives.detach().cpu().numpy()

	if primi_out:
		if (not os.path.isdir(ply_filename)) and (not primi_count):
			os.makedirs(ply_filename)
	
	mc_res = 64
	sampling_threshold = 0.9 #0.5?
	samples_grid = get_samples(N)
	samples = pass_generator_and_sample(decoder, samples_grid, surface_points, quads, N, c_dim, threshold = sampling_threshold)
	#samples = samples_points.detach().cpu().numpy()
	samples_half = get_samples(mc_res)

	#sdf to occ
	quads = quads*( (np.sum(np.abs(quads[:,0:3,:]), axis=1, keepdims=True)>1e-3) | (np.sum(np.abs(quads[:,3:6,:]), axis=1, keepdims=True)>1e-3) ).astype(np.float32)
	quads = quads/np.maximum(np.sqrt(np.sum(np.square(quads[:,0:6,:]), axis=1, keepdims=True)),1e-6)
	quads = quads*1e4

	#normalize
	print('model float time:, ', time.time() - start_time)
	#original_plane = np.copy(plane_m)

	model_float = pass_generator(decoder, samples, surface_points, quads, c_dim)
	#model float, 
	model_float = (model_float > sampling_threshold).astype(np.uint8)
	model_float_test = np.copy(model_float)

	c_dim_half = c_dim//2
	if differ:
		left_convex = np.amax(model_float[:,:c_dim_half],axis=1)
		right_convex = np.amax(model_float[:,c_dim_half:],axis=1)
		#model_float_sum = np.minimum(np.max(model_float[:,:c_dim_half],axis=1),1-np.max(model_float[:,c_dim_half:],axis=1))
	else:
		model_float_sum = np.max(model_float,axis=1)
	
	unused_convex = np.ones([c_dim], np.uint8)

	for i in range(c_dim):
		if np.max(model_float[:,i])>0:
			#print('check, ', i)
			model_float_test[:,i] = 0
			if differ:
				t_left_convex = np.amax(model_float_test[:,:c_dim_half],axis=1)
				t_right_convex = np.amax(model_float_test[:,c_dim_half:],axis=1)
				varied_output_left = (left_convex!=t_left_convex).astype(int)
				changed_point_num_left = np.sum(varied_output_left)

				varied_output_right = (right_convex!=t_right_convex).astype(int)
				changed_point_num_right = np.sum(varied_output_right)
				changed_point_num = changed_point_num_left + changed_point_num_right
				if changed_point_num > filter_threshold:
				#if np.any(np.minimum(np.max(model_float_test[:,:c_dim_half],axis=1),1-np.max(model_float_test[:,c_dim_half:],axis=1))!=model_float_sum):
					model_float_test[:,i] = model_float[:,i]
					unused_convex[i] = 0
					#print('keep, ', i)
				else:
					model_float[:,i] = 0
			else:
				if np.any(np.max(model_float_test,axis=1)!=model_float_sum):
					model_float_test[:,i] = model_float[:,i]
					unused_convex[i] = 0
					#print('keep, ', i)
				else:
					model_float[:,i] = 0
	print('convex num, ', c_dim - np.sum(unused_convex))
	if c_dim - np.sum(unused_convex) > 990:
		return
	#1 means not used
	#1, 1, c_dim
	#unused_convex = np.ones([c_dim], np.uint8)
	#unused_convex[[41, 201, 233]] = 0
	convex_mask = np.reshape(unused_convex, [1,1,-1])
	
	for i in range(c_dim):
		if i<c_dim/2:
			if unused_convex[i]:
				connection_t_convex[0,:,i] = 0
		else:
			if unused_convex[i]:
				connection_t_concave[0,:,i-int(c_dim/2)] = 0

	print('remove convex time:, ', time.time() - start_time)

	zg = pass_generator(decoder, samples, surface_points, quads, c_dim)
	correct_output = mask_generator_new(zg, convex_mask, c_dim_half, differ)
	correct_output = correct_output > sampling_threshold

	primitive_count = 0
	primitive_count_plane = 0
	primitive_count_quads = 0

	connection_t_temp_convex = (connection_t_convex>0.01).squeeze()
	connection_t_temp_convex = np.sum(connection_t_temp_convex.detach().cpu().numpy(), axis=1)
	connection_t_temp_convex = np.argwhere(connection_t_temp_convex > 0)
	print('convex primi needs to be checked, ', len(connection_t_temp_convex))
	plane_m_test = np.copy(quads)

	for i in connection_t_temp_convex:
		plane_m_test[0,:,i] = 0
		zg = pass_generator(decoder, samples, surface_points, plane_m_test, c_dim)
		this_output = mask_generator_new(zg, convex_mask, c_dim_half, differ)
		this_output = this_output>sampling_threshold
		varied_output = (correct_output^this_output).squeeze().astype(int)
		changed_point_num = np.sum(varied_output)
		
		if changed_point_num > filter_threshold:
			plane_m_test[0,:,i] = quads[0,:,i]
			primitive_count_plane += 1
		else:
			quads[0,:,i] = 0
			correct_output = np.logical_or(correct_output, this_output)

	connection_t_temp_concave = (connection_t_concave>0.01).squeeze()
	connection_t_temp_concave = np.sum(connection_t_temp_concave.detach().cpu().numpy(), axis=1)
	connection_t_temp_concave = np.argwhere(connection_t_temp_concave > 0)
	#plane_m_test = np.copy(plane_m)
	print('convex primi needs to be checked, ', len(connection_t_temp_concave))

	quads_test = np.copy(quads)

	for i in connection_t_temp_concave:
		quads_test[0,:,i+int(p_dim/2)] = 0
		#zg, h2
		zg = pass_generator(decoder, samples, surface_points, quads_test, c_dim)
		# if i == 166:
		# 	this_output = mask_generator(zg, convex_mask, c_dim_half, print_flag=True)
		# else:
		this_output = mask_generator_new(zg, convex_mask, c_dim_half, differ)
		this_output = this_output>sampling_threshold
		varied_output = (correct_output^this_output).squeeze().astype(int)
		changed_point_num = np.sum(varied_output)
		
		# ttt = (correct_output^this_output)
		# if i == 166:
		# 	print('ttt', ttt.shape)
		# 	print('sample indexes', np.argwhere(ttt>0))
		# 	print('correct_output, ', correct_output[:, 20040, :])
		# 	print('this_output, ', this_output[:, 20040, :])
		# 	print('samples,', samples[20040, :])
		if changed_point_num > filter_threshold:
			quads_test[0,:,i+int(p_dim/2)] = quads[0,:,i+int(p_dim/2)]
			primitive_count_quads += 1
		else:
			quads[0,:,i+int(p_dim/2)] = 0
			correct_output = np.logical_or(correct_output, this_output)
	
	print('primitive_count_plane, ', primitive_count_plane)
	print('primitive_count_quads, ', primitive_count_quads)
	
	# if c_dim - np.sum(unused_convex) > 10:
	# 	return 0
	print('remove plane time:, ', time.time() - start_time)
	#exit(0)
	# for i in connection_t_temp:
	# 	if i < p_dim:
	# 		if plane_norm[0, i] < 2:
	# 			plane_m[0,:,i] = 0

	connection_t_convex = (connection_t_convex>0.01).float()
	connection_t_concave = (connection_t_concave>0.01).float()
	for i in range(p_dim):
		if np.all(quads[0,:,i]==0):
			if i < p_dim/2:
				connection_t_convex[0,i,:] = 0
			else:
				connection_t_concave[0,i - int(p_dim/2),:] = 0

	if primi_count:
		return c_dim - np.sum(unused_convex), primitive_count_plane, primitive_count_quads, connection_t_convex, connection_t_concave, unused_convex

	#csg
	print('csg start time:, ', time.time() - start_time)
	dimf = mc_res
	model_float = np.full([dimf+2,dimf+2,dimf+2],100,np.float32)
	output_mesh = None
	concave_mesh_left = None
	concave_mesh_right = None
	if primi_out:
		primi_dir = ply_filename + '/primi_for_connection' 
		primi_dir2 = ply_filename + '/primi_for_render'
		convex_dir = ply_filename + '/convex'
		concave_dir = ply_filename + '/concave'

		if not os.path.isdir(primi_dir):
			os.makedirs(primi_dir)
		if not os.path.isdir(primi_dir2):
			os.makedirs(primi_dir2)
		if not os.path.isdir(convex_dir):
			os.makedirs(convex_dir)
		if not os.path.isdir(concave_dir):
			os.makedirs(concave_dir)

	#convex_points = np.zeros((int(c_dim - np.sum(unused_convex)), 8192, 3))
	#unused_convex[182] = 0
	c_index = 0
	for i in range(c_dim):
		if unused_convex[i]==0:
			print('convex %d, time: %f'%(i, time.time() - start_time))
			
			convex_mesh = None
			for j in range(p_dim):
				if (i<c_dim/2 and j<p_dim/2 and connection_t_convex[0,j,i]>0) or \
					(i>=c_dim/2 and j>=p_dim/2 and connection_t_concave[0,j-int(p_dim/2),i-int(c_dim/2)]>0):
				#if connection_t[0,j,i]>0:
					#print(f'{i}, {j}, {original_plane[:,:,j:j+1]}, {plane_norm[:, j:j+1]}')
					model_out, _ = get_surface_grid(np.expand_dims(samples_half, axis=0), original_plane[:,:,j:j+1], quadric)
					#with boundary closed
					model_float[1:-1,1:-1,1:-1] = np.reshape(model_out, [dimf,dimf,dimf])
					if i<c_dim_half:
						vertices, triangles = mcubes.marching_cubes(-model_float, -0)
					else:
						# if i == 57:
						# 	vertices, triangles = mcubes.marching_cubes(-model_float, -0.2)
						# else:
						vertices, triangles = mcubes.marching_cubes(-model_float, -concave_level_set)

					vertices = (vertices-0.5)/dimf-0.5 #+ np.array([0.01, 0.01, 0]).reshape(-1, 3)
					
					if convex_mesh is None:
						convex_mesh = pymesh.form_mesh(vertices, triangles)
					else:
						convex_mesh = pymesh.boolean(convex_mesh, pymesh.form_mesh(vertices, triangles), operation="intersection")

					if primi_out:
						#open boundary
						model_out = np.reshape(model_out, [dimf,dimf,dimf])
						if i<c_dim_half:
							vertices, triangles = mcubes.marching_cubes(-model_out, -0)
						else:
							#vertices, triangles = mcubes.marching_cubes(-model_out, -0.001)
							vertices, triangles = mcubes.marching_cubes(-model_out, -0)

						vertices = (vertices+0.5)/dimf-0.5

						#pymesh.meshio.save_mesh_raw(filename, vertices, faces)

						pymesh.meshio.save_mesh_raw(primi_dir +"/"+str(i)+ "_"+ str(j) +"." + file_type, vertices, triangles)
						pymesh.meshio.save_mesh_raw(primi_dir2 +"/" + str(j) +"." + file_type, vertices, triangles)
			if primi_out:		
				pymesh.meshio.save_mesh_raw(convex_dir +"/"+str(i)+"." + file_type, convex_mesh.vertices, convex_mesh.faces)
			
			c_index += 1

			if i < c_dim_half:
				if concave_mesh_left is None:
					concave_mesh_left = convex_mesh
				else:
					concave_mesh_left = pymesh.boolean(concave_mesh_left, convex_mesh, operation="union")
			else:
				if concave_mesh_right is None:
					concave_mesh_right = convex_mesh
				else:
					concave_mesh_right = pymesh.boolean(concave_mesh_right, convex_mesh, operation="union")	
	if differ:
		if concave_mesh_left is None:
			if concave_mesh_right is None:
				print('empty mesh')
				exit()
			else:
				output_mesh = concave_mesh_right
				if primi_out:
					pymesh.meshio.save_mesh_raw(concave_dir +"/right"+"." + file_type, concave_mesh_right.vertices, concave_mesh_right.faces)

		else:
			if concave_mesh_right is None:
				output_mesh = concave_mesh_left
				if primi_out:
					pymesh.meshio.save_mesh_raw(concave_dir +"/left"+"." + file_type, concave_mesh_left.vertices, concave_mesh_left.faces)

			else:
				output_mesh = pymesh.boolean(concave_mesh_left, concave_mesh_right, operation="difference")
				if primi_out:
					pymesh.meshio.save_mesh_raw(concave_dir +"/left"+"." + file_type, concave_mesh_left.vertices, concave_mesh_left.faces)
					pymesh.meshio.save_mesh_raw(concave_dir +"/right"+"." + file_type, concave_mesh_right.vertices, concave_mesh_right.faces)
	else:
		output_mesh = pymesh.boolean(concave_mesh_left, concave_mesh_right, operation="union")
		if primi_out:
			pymesh.meshio.save_mesh_raw(concave_dir +"/left"+"." + file_type, concave_mesh_left.vertices, concave_mesh_left.faces)
			pymesh.meshio.save_mesh_raw(concave_dir +"/right"+"." + file_type, concave_mesh_right.vertices, concave_mesh_right.faces)

	if primi_out:
		file_name = os.path.basename(ply_filename)
		pymesh.meshio.save_mesh_raw(os.path.join(ply_filename, file_name + "." + file_type), output_mesh.vertices, output_mesh.faces)
		print('primitive meshing done')
	else:
		dirname = os.path.dirname(ply_filename)
		if not os.path.exists(dirname):
			os.makedirs(dirname)

		pymesh.meshio.save_mesh_raw(ply_filename + "." + file_type, output_mesh.vertices, output_mesh.faces)
		print('csg meshing done')

#designed to take 64^3 voxels!
def sample_points_polygon_vox64(vertices, polygons, voxel_model_64, num_of_points):
	#convert polygons to triangles
	triangles = []
	for ii in range(len(polygons)):
		for jj in range(len(polygons[ii])-2):
			triangles.append( [polygons[ii][0], polygons[ii][jj+1], polygons[ii][jj+2]] )
	triangles = np.array(triangles, np.int32)
	vertices = np.array(vertices, np.float32)

	small_step = 1.0/64
	epsilon = 1e-6
	triangle_area_list = np.zeros([len(triangles)],np.float32)
	triangle_normal_list = np.zeros([len(triangles),3],np.float32)
	for i in range(len(triangles)):
		#area = |u x v|/2 = |u||v|sin(uv)/2
		a,b,c = vertices[triangles[i,1]]-vertices[triangles[i,0]]
		x,y,z = vertices[triangles[i,2]]-vertices[triangles[i,0]]
		ti = b*z-c*y
		tj = c*x-a*z
		tk = a*y-b*x
		area2 = math.sqrt(ti*ti+tj*tj+tk*tk)
		if area2<epsilon:
			triangle_area_list[i] = 0
			triangle_normal_list[i,0] = 0
			triangle_normal_list[i,1] = 0
			triangle_normal_list[i,2] = 0
		else:
			triangle_area_list[i] = area2
			triangle_normal_list[i,0] = ti/area2
			triangle_normal_list[i,1] = tj/area2
			triangle_normal_list[i,2] = tk/area2
	
	triangle_area_sum = np.sum(triangle_area_list)
	sample_prob_list = (num_of_points/triangle_area_sum)*triangle_area_list

	triangle_index_list = np.arange(len(triangles))

	point_normal_list = np.zeros([num_of_points,6],np.float32)
	count = 0
	watchdog = 0

	while(count<num_of_points):
		np.random.shuffle(triangle_index_list)
		watchdog += 1
		if watchdog>100:
			print("infinite loop here!")
			return point_normal_list
		for i in range(len(triangle_index_list)):
			if count>=num_of_points: break
			dxb = triangle_index_list[i]
			prob = sample_prob_list[dxb]
			prob_i = int(prob)
			prob_f = prob-prob_i
			if np.random.random()<prob_f:
				prob_i += 1
			normal_direction = triangle_normal_list[dxb]
			u = vertices[triangles[dxb,1]]-vertices[triangles[dxb,0]]
			v = vertices[triangles[dxb,2]]-vertices[triangles[dxb,0]]
			base = vertices[triangles[dxb,0]]
			for j in range(prob_i):
				#sample a point here:
				u_x = np.random.random()
				v_y = np.random.random()
				if u_x+v_y>=1:
					u_x = 1-u_x
					v_y = 1-v_y
				ppp = u*u_x+v*v_y+base

				#verify normal
				pppn1 = (ppp+normal_direction*small_step+0.5)*64
				px1 = int(pppn1[0])
				py1 = int(pppn1[1])
				pz1 = int(pppn1[2])

				ppx = int((ppp[0]+0.5)*64)
				ppy = int((ppp[1]+0.5)*64)
				ppz = int((ppp[2]+0.5)*64)
				
				if ppx<0 or ppx>=64 or ppy<0 or ppy>=64 or ppz<0 or ppz>=64:
					continue
				if voxel_model_64[ppx,ppy,ppz]>1e-3 or px1<0 or px1>=64 or py1<0 or py1>=64 or pz1<0 or pz1>=64 or voxel_model_64[px1,py1,pz1]>1e-3:
					#valid
					point_normal_list[count,:3] = ppp
					point_normal_list[count,3:] = normal_direction
					count += 1
					if count>=num_of_points: break

	return point_normal_list
