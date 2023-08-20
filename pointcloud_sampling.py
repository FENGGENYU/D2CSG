import trimesh
import numpy as np
import glob
import multiprocessing as mp
import argparse
import os
import traceback
import time
import sys
from sklearn.neighbors import KDTree
import trimesh
import json

#sample points for each part
def Random_sampleShape(meshes, sample_num):
	segments = []
	
	for si, m in enumerate(meshes):
		f = m.faces 
		segments += [si for _ in range(f.shape[0])]

	shape_mesh = trimesh.util.concatenate(meshes)
	samps, face_inds = shape_mesh.sample(sample_num, return_index=True)

	segments = np.array(segments)

	samp_segments = segments[face_inds]
	normals = shape_mesh.face_normals[face_inds]
	samps = np.concatenate([samps, normals], 1)

	return samps, samp_segments 

def Even_sampleShape(meshes, surface_points_num, min_point_threshold=10, large_part_threshold=100):
	verts = []
	faces = []
	segments = []

	offset = 0
	point_num = np.zeros(len(meshes))
	for si, convex_mesh in enumerate(meshes):
		point_num[si] = convex_mesh.area

	total_area = np.sum(point_num)

	large_part_num = 0
	large_part_area = 0

	for si, convex_mesh in enumerate(meshes):
		point_num[si] = int(point_num[si] * surface_points_num / total_area)
		if point_num[si] < min_point_threshold:
			point_num[si] = min_point_threshold
		if point_num[si] > large_part_threshold:
			large_part_num += 1
			large_part_area += convex_mesh.area

	additional_num = np.sum(point_num) - surface_points_num
	current_id = 0
	for si, convex_mesh in enumerate(meshes):
		if point_num[si]>large_part_threshold:
			point_num[si] = int(point_num[si] - additional_num * convex_mesh.area / large_part_area)
			current_id += 1
			if current_id == large_part_num:
				point_num[si] = int(point_num[si]) + surface_points_num - np.sum(point_num)
	
	min_pos = np.argmin(point_num)
	while point_num[min_pos] <= 0:
		max_pos = np.argmax(point_num)
		change_num = 10 - point_num[min_pos]
		point_num[min_pos] += change_num
		point_num[max_pos] -= change_num
		min_pos = np.argmin(point_num)
	
	assert np.sum(point_num) == surface_points_num

	all_points = []
	for si, convex_mesh in enumerate(meshes):
		points, indexs = convex_mesh.sample(int(point_num[si]), return_index=True)
		normals = convex_mesh.face_normals[indexs]
		s_points = np.concatenate([points, normals], 1)

		segments += [si for _ in range(s_points.shape[0])]
		all_points.append(s_points)

	samps = np.concatenate(all_points, axis=0)
	samp_segments = np.array(segments)
	return samps, samp_segments

def trimesh_normalize(mesh):
	from numpy import linalg as LA
	total_size = (mesh.bounds[1] - mesh.bounds[0])
	total_size = LA.norm(total_size)
	centers = (mesh.bounds[1] + mesh.bounds[0]) /2
	mesh.apply_translation(-centers)
	mesh.apply_scale(1/total_size)
	return mesh, centers, total_size

def node_sampling(inds, in_dir, out_file_name, node_p_num, label_exists):
	
	shape_num = len(inds)
	max_part_count = 150
	points = np.zeros([shape_num,node_p_num,6], np.float)
	segs = np.zeros([shape_num,node_p_num], np.float)
	part_labels = np.zeros([shape_num, max_part_count], np.int) - 1
	part_count = np.zeros([shape_num], np.int)
	all_names = []
	
	for shape_id in range(shape_num):
		shape_name = inds[shape_id]
		print(f'shape id {shape_id} shape name {shape_name}')
		shape_dir = os.path.join(in_dir, shape_name)
		
		convex_dir = os.path.join(shape_dir, 'convex')
		convex_num = len(os.listdir(convex_dir))
		
		#--------------------------------
		#read labels
		if label_exists:
			label_json = os.path.join(shape_dir, 'label.json')

			with open(label_json, 'r') as file:
				labels = json.load(file)
			labels = np.array(labels).reshape(-1)
		#--------------------------------
		#read meshes
		node_convex_meshes = []
		for p_i in range(convex_num):
			convex_name = os.path.join(convex_dir, f'{p_i}.obj')
			mesh = trimesh.load(convex_name)
			node_convex_meshes.append(mesh)
		
		node_mesh = trimesh.util.concatenate(node_convex_meshes)
		#------------------------------------
		#normalization
		node_mesh, centers, total_size = trimesh_normalize(node_mesh)
		for mesh in node_convex_meshes:
			mesh.apply_translation(-centers)
			mesh.apply_scale(1/total_size)
		#------------------------------------
		samps, samp_segments = Even_sampleShape(node_convex_meshes, node_p_num)
		
		#------------------------------------------
		points[shape_id, :, :] = samps
		if label_exists:
			part_labels[shape_id, :len(labels)] = labels
		part_count[shape_id] = convex_num
		segs[shape_id, :] = samp_segments

		all_names.append(shape_name)
	#---------------------------------

	if label_exists:
		np.savez(out_file_name, points = points, 
			part_labels = part_labels, part_count = part_count,
			segments = segs,
			names = all_names)
	else:
		np.savez(out_file_name, points = points, part_count = part_count,
			segments = segs,
			names = all_names)
			
	print('max_part_count ', max_part_count)
	

if __name__ == '__main__':
	#python pointcloud_sampling.py --cate chair --split train
	import argparse

	arg_parser = argparse.ArgumentParser("sampling points")
	arg_parser.add_argument("--cate", dest="cate", default="chair")
	arg_parser.add_argument("--split", dest="split", default="train")
	arg_parser.add_argument("--num", dest="sampling_num", default=8192, type=int)
	arg_parser.add_argument("--out", dest="out_dir", default="/local-scratch2/workshop/data")
	arg_parser.add_argument("--in", dest="in_dir", default="/local-scratch2/workshop/data/abo-part-labels")

	args = arg_parser.parse_args()

	category = args.cate
	sample_num = args.sampling_num
	split = args.split
	if split == 'dev' or split == 'test':
		label_exists = False
	else:
		label_exists = True
	in_dir = os.path.join(args.in_dir, category, split)
	out_dir = os.path.join(args.out_dir, category)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	print(f'category {category}')
	#-------------------------
	ASINs = os.listdir(in_dir)
	ASINs.sort()
	out_file_name = os.path.join(out_dir, f'abo_{category}_{split}.npz')
	node_sampling(ASINs, in_dir, out_file_name, sample_num, label_exists)