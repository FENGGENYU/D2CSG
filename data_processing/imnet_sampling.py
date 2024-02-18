import numpy as np
import os, sys
import h5py
import random
import json
from multiprocessing import Process, Queue
import queue
import time
import binvox_rw
#import mcubes


voxel_dim = 64

batch_size_16 = 16*16*16
batch_size_32 = 16*16*16*2
batch_size_64 = 16*16*16*6
#batch_size_128 = 16*16*16*24

def get_vox_from_binvox_1024(objname, inversed):
	voxel_model_file = open(objname, 'rb')
	#voxel_model_256 = np.load(objname)['voxels']

	voxel_model_1024 = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=True).data.astype(np.uint8)
	step_size = 2

	if inversed:
		voxel_model_1024 = 1 - voxel_model_1024

	voxel_model_512 = voxel_model_1024[0::step_size,0::step_size,0::step_size]
	for i in range(step_size):
		for j in range(step_size):
			for k in range(step_size):
				voxel_model_512 = np.maximum(voxel_model_512,voxel_model_1024[i::step_size,j::step_size,k::step_size])
	voxel_model_1024 = None

	voxel_model_256 = voxel_model_512[0::step_size,0::step_size,0::step_size]
	for i in range(step_size):
		for j in range(step_size):
			for k in range(step_size):
				voxel_model_256 = np.maximum(voxel_model_256,voxel_model_512[i::step_size,j::step_size,k::step_size])
	voxel_model_512 = None
	

	voxel_model_128 = voxel_model_256[0::step_size,0::step_size,0::step_size]
	for i in range(step_size):
		for j in range(step_size):
			for k in range(step_size):
				voxel_model_128 = np.maximum(voxel_model_128,voxel_model_256[i::step_size,j::step_size,k::step_size])

	voxel_model_64 = voxel_model_128[0::step_size,0::step_size,0::step_size]
	for i in range(step_size):
		for j in range(step_size):
			for k in range(step_size):
				voxel_model_64 = np.maximum(voxel_model_64,voxel_model_128[i::step_size,j::step_size,k::step_size])

	voxel_model_32 = voxel_model_64[0::step_size,0::step_size,0::step_size]
	for i in range(step_size):
		for j in range(step_size):
			for k in range(step_size):
				voxel_model_32 = np.maximum(voxel_model_32,voxel_model_64[i::step_size,j::step_size,k::step_size])

	voxel_model_16 = voxel_model_32[0::step_size,0::step_size,0::step_size]
	for i in range(step_size):
		for j in range(step_size):
			for k in range(step_size):
				voxel_model_16 = np.maximum(voxel_model_16,voxel_model_32[i::step_size,j::step_size,k::step_size])

	return voxel_model_256, voxel_model_128, voxel_model_64, voxel_model_32, voxel_model_16


def sample_point_in_cube(block,target_value,halfie):
	halfie2 = halfie*2
	
	for i in range(100):
		x = np.random.randint(halfie2)
		y = np.random.randint(halfie2)
		z = np.random.randint(halfie2)
		if block[x,y,z]==target_value:
			return x,y,z
	
	if block[halfie,halfie,halfie]==target_value:
		return halfie,halfie,halfie
	
	i=1
	ind = np.unravel_index(np.argmax(block[halfie-i:halfie+i,halfie-i:halfie+i,halfie-i:halfie+i], axis=None), (i*2,i*2,i*2))
	if block[ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i]==target_value:
		return ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i
	
	for i in range(2,halfie+1):
		six = [(halfie-i,halfie,halfie),(halfie+i-1,halfie,halfie),(halfie,halfie,halfie-i),(halfie,halfie,halfie+i-1),(halfie,halfie-i,halfie),(halfie,halfie+i-1,halfie)]
		for j in range(6):
			if block[six[j]]==target_value:
				return six[j]
		ind = np.unravel_index(np.argmax(block[halfie-i:halfie+i,halfie-i:halfie+i,halfie-i:halfie+i], axis=None), (i*2,i*2,i*2))
		if block[ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i]==target_value:
			return ind[0]+halfie-i,ind[1]+halfie-i,ind[2]+halfie-i
	print('hey, error in your code!')
	exit(0)


def get_points_from_vox(q, name_list):
	name_num = len(name_list)
	for idx in range(name_num):
		print(idx,'/',name_num)

		Pvoxel_model_256, Pvoxel_model_128, Pvoxel_model_64, Pvoxel_model_32, Pvoxel_model_16 = get_vox_from_binvox_1024(name_list[idx][1], False)
		Nvoxel_model_256, Nvoxel_model_128, Nvoxel_model_64, Nvoxel_model_32, Nvoxel_model_16 = get_vox_from_binvox_1024(name_list[idx][1], True)

		#write voxel
		sample_voxels = np.reshape(Pvoxel_model_64, (64,64,64,1))

		# --- P 64 ---
		dim_voxel = 64
		multiplier = int(256/dim_voxel)
		halfie = int(multiplier/2)
		batch_size = batch_size_64
		voxel_model_temp = Pvoxel_model_64
		voxel_model_256_temp = Pvoxel_model_256
		
		sample_points = np.zeros([batch_size,3],np.uint8)
		sample_values = np.zeros([batch_size,1],np.uint8)
		batch_size_counter = 0
		voxel_model_temp_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		temp_range = list(range(1,dim_voxel-1,4))+list(range(2,dim_voxel-1,4))+list(range(3,dim_voxel-1,4))+list(range(4,dim_voxel-1,4))
		for j in temp_range:
			if (batch_size_counter>=batch_size): break
			for i in temp_range:
				if (batch_size_counter>=batch_size): break
				for k in temp_range:
					if (batch_size_counter>=batch_size): break
					if (np.max(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])!=np.min(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])):
						si,sj,sk = sample_point_in_cube(voxel_model_256_temp[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
						sample_points[batch_size_counter,0] = si+i*multiplier
						sample_points[batch_size_counter,1] = sj+j*multiplier
						sample_points[batch_size_counter,2] = sk+k*multiplier
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
				si,sj,sk = sample_point_in_cube(voxel_model_256_temp[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
				sample_points[batch_size_counter,0] = si+i*multiplier
				sample_points[batch_size_counter,1] = sj+j*multiplier
				sample_points[batch_size_counter,2] = sk+k*multiplier
				sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
				voxel_model_temp_flag[i,j,k] = 1
				batch_size_counter +=1
		
		Psample_points_64 = sample_points
		Psample_values_64 = sample_values

		# --- N 64 ---
		dim_voxel = 64
		multiplier = int(256/dim_voxel)
		halfie = int(multiplier/2)
		batch_size = batch_size_64
		voxel_model_temp = Nvoxel_model_64
		voxel_model_256_temp = Nvoxel_model_256
		
		sample_points = np.zeros([batch_size,3],np.uint8)
		sample_values = np.zeros([batch_size,1],np.uint8)
		batch_size_counter = 0
		voxel_model_temp_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		temp_range = list(range(1,dim_voxel-1,4))+list(range(2,dim_voxel-1,4))+list(range(3,dim_voxel-1,4))+list(range(4,dim_voxel-1,4))
		for j in temp_range:
			if (batch_size_counter>=batch_size): break
			for i in temp_range:
				if (batch_size_counter>=batch_size): break
				for k in temp_range:
					if (batch_size_counter>=batch_size): break
					if (np.max(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])!=np.min(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])):
						si,sj,sk = sample_point_in_cube(voxel_model_256_temp[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
						sample_points[batch_size_counter,0] = si+i*multiplier
						sample_points[batch_size_counter,1] = sj+j*multiplier
						sample_points[batch_size_counter,2] = sk+k*multiplier
						sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
						voxel_model_temp_flag[i,j,k] = 1
						batch_size_counter +=1
		if (batch_size_counter>=batch_size):
			pass
		else:
			#fill other slots with random points
			while (batch_size_counter<batch_size):
				while True:
					i = random.randint(0,dim_voxel-1)
					j = random.randint(0,dim_voxel-1)
					k = random.randint(0,dim_voxel-1)
					if voxel_model_temp_flag[i,j,k] != 1: break
				si,sj,sk = sample_point_in_cube(voxel_model_256_temp[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
				sample_points[batch_size_counter,0] = si+i*multiplier
				sample_points[batch_size_counter,1] = sj+j*multiplier
				sample_points[batch_size_counter,2] = sk+k*multiplier
				sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
				voxel_model_temp_flag[i,j,k] = 1
				batch_size_counter +=1
		
		Nsample_points_64 = sample_points
		Nsample_values_64 = sample_values



		# --- P 32 ---
		dim_voxel = 32
		multiplier = int(256/dim_voxel)
		halfie = int(multiplier/2)
		batch_size = batch_size_32
		voxel_model_temp = Pvoxel_model_32
		voxel_model_256_temp = Pvoxel_model_256
		
		sample_points = np.zeros([batch_size,3],np.uint8)
		sample_values = np.zeros([batch_size,1],np.uint8)
		batch_size_counter = 0
		voxel_model_temp_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		temp_range = list(range(1,dim_voxel-1,4))+list(range(2,dim_voxel-1,4))+list(range(3,dim_voxel-1,4))+list(range(4,dim_voxel-1,4))
		for j in temp_range:
			if (batch_size_counter>=batch_size): break
			for i in temp_range:
				if (batch_size_counter>=batch_size): break
				for k in temp_range:
					if (batch_size_counter>=batch_size): break
					if (np.max(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])!=np.min(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])):
						si,sj,sk = sample_point_in_cube(voxel_model_256_temp[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
						sample_points[batch_size_counter,0] = si+i*multiplier
						sample_points[batch_size_counter,1] = sj+j*multiplier
						sample_points[batch_size_counter,2] = sk+k*multiplier
						sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
						voxel_model_temp_flag[i,j,k] = 1
						batch_size_counter +=1
		if (batch_size_counter>=batch_size):
			print("32-- batch_size exceeded!")
			exceed_32_flag = 1
		else:
			exceed_32_flag = 0
			#fill other slots with random points
			while (batch_size_counter<batch_size):
				while True:
					i = random.randint(0,dim_voxel-1)
					j = random.randint(0,dim_voxel-1)
					k = random.randint(0,dim_voxel-1)
					if voxel_model_temp_flag[i,j,k] != 1: break
				si,sj,sk = sample_point_in_cube(voxel_model_256_temp[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
				sample_points[batch_size_counter,0] = si+i*multiplier
				sample_points[batch_size_counter,1] = sj+j*multiplier
				sample_points[batch_size_counter,2] = sk+k*multiplier
				sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
				voxel_model_temp_flag[i,j,k] = 1
				batch_size_counter +=1
		
		Psample_points_32 = sample_points
		Psample_values_32 = sample_values



		# --- N 32 ---
		dim_voxel = 32
		multiplier = int(256/dim_voxel)
		halfie = int(multiplier/2)
		batch_size = batch_size_32
		voxel_model_temp = Nvoxel_model_32
		voxel_model_256_temp = Nvoxel_model_256
		
		sample_points = np.zeros([batch_size,3],np.uint8)
		sample_values = np.zeros([batch_size,1],np.uint8)
		batch_size_counter = 0
		voxel_model_temp_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
		temp_range = list(range(1,dim_voxel-1,4))+list(range(2,dim_voxel-1,4))+list(range(3,dim_voxel-1,4))+list(range(4,dim_voxel-1,4))
		for j in temp_range:
			if (batch_size_counter>=batch_size): break
			for i in temp_range:
				if (batch_size_counter>=batch_size): break
				for k in temp_range:
					if (batch_size_counter>=batch_size): break
					if (np.max(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])!=np.min(voxel_model_temp[i-1:i+2,j-1:j+2,k-1:k+2])):
						si,sj,sk = sample_point_in_cube(voxel_model_256_temp[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
						sample_points[batch_size_counter,0] = si+i*multiplier
						sample_points[batch_size_counter,1] = sj+j*multiplier
						sample_points[batch_size_counter,2] = sk+k*multiplier
						sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
						voxel_model_temp_flag[i,j,k] = 1
						batch_size_counter +=1
		if (batch_size_counter>=batch_size):
			print("32-- batch_size exceeded!")
			exceed_32_flag = 1
		else:
			exceed_32_flag = 0
			#fill other slots with random points
			while (batch_size_counter<batch_size):
				while True:
					i = random.randint(0,dim_voxel-1)
					j = random.randint(0,dim_voxel-1)
					k = random.randint(0,dim_voxel-1)
					if voxel_model_temp_flag[i,j,k] != 1: break
				si,sj,sk = sample_point_in_cube(voxel_model_256_temp[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
				sample_points[batch_size_counter,0] = si+i*multiplier
				sample_points[batch_size_counter,1] = sj+j*multiplier
				sample_points[batch_size_counter,2] = sk+k*multiplier
				sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
				voxel_model_temp_flag[i,j,k] = 1
				batch_size_counter +=1
		
		Nsample_points_32 = sample_points
		Nsample_values_32 = sample_values



		# --- P 16 ---
		dim_voxel = 16
		multiplier = int(256/dim_voxel)
		halfie = int(multiplier/2)
		batch_size = batch_size_16
		voxel_model_temp = Pvoxel_model_16
		voxel_model_256_temp = Pvoxel_model_256
		
		sample_points = np.zeros([batch_size,3],np.uint8)
		sample_values = np.zeros([batch_size,1],np.uint8)
		batch_size_counter = 0
		for i in range(dim_voxel):
			for j in range(dim_voxel):
				for k in range(dim_voxel):
					si,sj,sk = sample_point_in_cube(voxel_model_256_temp[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
					sample_points[batch_size_counter,0] = si+i*multiplier
					sample_points[batch_size_counter,1] = sj+j*multiplier
					sample_points[batch_size_counter,2] = sk+k*multiplier
					sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
					batch_size_counter +=1
		if (batch_size_counter!=batch_size):
			print("batch_size_counter!=batch_size")
		
		Psample_points_16 = sample_points
		Psample_values_16 = sample_values



		# --- N 16 ---
		dim_voxel = 16
		multiplier = int(256/dim_voxel)
		halfie = int(multiplier/2)
		batch_size = batch_size_16
		voxel_model_temp = Nvoxel_model_16
		voxel_model_256_temp = Nvoxel_model_256
		
		sample_points = np.zeros([batch_size,3],np.uint8)
		sample_values = np.zeros([batch_size,1],np.uint8)
		batch_size_counter = 0
		for i in range(dim_voxel):
			for j in range(dim_voxel):
				for k in range(dim_voxel):
					si,sj,sk = sample_point_in_cube(voxel_model_256_temp[i*multiplier:(i+1)*multiplier,j*multiplier:(j+1)*multiplier,k*multiplier:(k+1)*multiplier],voxel_model_temp[i,j,k],halfie)
					sample_points[batch_size_counter,0] = si+i*multiplier
					sample_points[batch_size_counter,1] = sj+j*multiplier
					sample_points[batch_size_counter,2] = sk+k*multiplier
					sample_values[batch_size_counter,0] = voxel_model_temp[i,j,k]
					batch_size_counter +=1
		if (batch_size_counter!=batch_size):
			print("batch_size_counter!=batch_size")
		
		Nsample_points_16 = sample_points
		Nsample_values_16 = sample_values
		
		q.put([name_list[idx][0],exceed_64_flag,exceed_32_flag,Psample_points_64,Psample_values_64,Nsample_points_64,Nsample_values_64,Psample_points_32,Psample_values_32,Nsample_points_32,Nsample_values_32,Psample_points_16,Psample_values_16,Nsample_points_16,Nsample_values_16,sample_voxels])


if __name__ == '__main__':
	#python imnet_sampling.py /local-scratch2/fenggeny/SECAD-Net-main/data/mingrui_data
	input_dir = sys.argv[1]
	voxel_input = os.path.join(input_dir, 'shapes')
	#dir of voxel models
	#voxel_input = input_dir + '/shapes/'

	
	#record statistics
	fstatistics = open(input_dir+'/statistics.txt','w',newline='')
	exceed_32 = 0
	exceed_64 = 0

	# name_list = os.listdir(voxel_input)
	# name_list = sorted(name_list)
	name_list = np.load(os.path.join(input_dir, 'test_names.npz'))['names']
	name_num = len(name_list)
	# fin = open(input_dir + "/names.txt", 'r')
	# name_list = [name.strip() for name in fin.readlines()]
	# fin.close()
	# name_num = len(name_list)

	#obj_list
	#fout = open(input_dir+'/'+ 'vox256.txt','w',newline='')
	
	# for i in range(name_num):
	# 	fout.write(name_list[i]+"\n")
	# fout.close()
	
	#prepare list of names
	num_of_process = 8
	list_of_list_of_names = []
	for i in range(num_of_process):
		list_of_names = []
		for j in range(i,name_num,num_of_process):
			list_of_names.append([j, voxel_input+ '/' + name_list[j] + '/model_filled.binvox'])
		list_of_list_of_names.append(list_of_names)
	
	#map processes
	q = Queue()
	workers = [Process(target=get_points_from_vox, args = (q, list_of_names)) for list_of_names in list_of_list_of_names]

	for p in workers:
		p.start()


	#reduce process
	hdf5_file_voxels = np.zeros([name_num,voxel_dim,voxel_dim,voxel_dim,1], np.uint8)

	hdf5_file_points_16 = np.zeros([name_num,batch_size_16,3], np.uint8)
	hdf5_file_values_16 = np.zeros([name_num,batch_size_16,1], np.uint8)
	hdf5_file_points_32 = np.zeros([name_num,batch_size_32,3], np.uint8)
	hdf5_file_values_32 = np.zeros([name_num,batch_size_32,1], np.uint8)
	hdf5_file_points_64 = np.zeros([name_num,batch_size_64,3], np.uint8)
	hdf5_file_values_64 = np.zeros([name_num,batch_size_64,1], np.uint8)


	while True:
		item_flag = True
		try:
			idx,exceed_64_flag,exceed_32_flag,Psample_points_64,Psample_values_64,Nsample_points_64,Nsample_values_64,Psample_points_32,Psample_values_32,Nsample_points_32,Nsample_values_32,Psample_points_16,Psample_values_16,Nsample_points_16,Nsample_values_16,sample_voxels = q.get(True, 1.0)
		except queue.Empty:
			item_flag = False
		
		if item_flag:
			#process result
			exceed_32+=exceed_32_flag
			exceed_64+=exceed_64_flag
			hdf5_file_points_64[idx,:,:] = Psample_points_64
			hdf5_file_values_64[idx,:,:] = Psample_values_64
			hdf5_file_points_32[idx,:,:] = Psample_points_32
			hdf5_file_values_32[idx,:,:] = Psample_values_32
			hdf5_file_points_16[idx,:,:] = Psample_points_16
			hdf5_file_values_16[idx,:,:] = Psample_values_16
			hdf5_file_voxels[idx,:,:,:,:] = sample_voxels
		
		allExited = True
		for p in workers:
			if p.exitcode is None:
				allExited = False
				break
		if allExited and q.empty():
			break


	fstatistics.write("total: "+str(name_num)+"\n")
	fstatistics.write("exceed_32: "+str(exceed_32)+"\n")
	fstatistics.write("exceed_32_ratio: "+str(float(exceed_32)/name_num)+"\n")
	fstatistics.write("exceed_64: "+str(exceed_64)+"\n")
	fstatistics.write("exceed_64_ratio: "+str(float(exceed_64)/name_num)+"\n")
	fstatistics.close()

	#name of output file
	hdf5_path = input_dir +'/'+'ae_voxel_points_samples.hdf5'
	hdf5_file = h5py.File(hdf5_path, 'w')

	hdf5_file.create_dataset("voxels", [name_num,voxel_dim,voxel_dim,voxel_dim,1], np.uint8, compression=9)
	hdf5_file.create_dataset("points_16", [name_num,batch_size_16,3], np.uint8, compression=9)
	hdf5_file.create_dataset("values_16", [name_num,batch_size_16,1], np.uint8, compression=9)
	hdf5_file.create_dataset("points_32", [name_num,batch_size_32,3], np.uint8, compression=9)
	hdf5_file.create_dataset("values_32", [name_num,batch_size_32,1], np.uint8, compression=9)
	hdf5_file.create_dataset("points_64", [name_num,batch_size_64,3], np.uint8, compression=9)
	hdf5_file.create_dataset("values_64", [name_num,batch_size_64,1], np.uint8, compression=9)

	hdf5_file["points_64"][:] = hdf5_file_points_64
	hdf5_file["values_64"][:] = hdf5_file_values_64
	hdf5_file["points_32"][:] = hdf5_file_points_32
	hdf5_file["values_32"][:] = hdf5_file_values_32
	hdf5_file["points_16"][:] = hdf5_file_points_16
	hdf5_file["values_16"][:] = hdf5_file_values_16
	hdf5_file["voxels"][:] = hdf5_file_voxels

	hdf5_file.close()
	print("finished")


