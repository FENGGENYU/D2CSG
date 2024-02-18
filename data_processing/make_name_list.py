import trimesh
import numpy as np
import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import traceback
import argparse
import sys

####
#input one directory and all files in it
#output npz file
def Read_directory_to_npz(in_path):
	def model_names(model_path):
		""" Return model names"""
		model_names = [name for name in os.listdir(model_path)]
		return model_names

	models = model_names(in_path)
	models.sort()
	#print(models)
	out_path = os.path.join(os.path.dirname(in_path), 'names.npz')
	np.savez(out_path, names = models, test_names=models)
	print('finished {}'.format(in_path))

	print('save to {}'.format(out_path))

def Read_directory_to_txt(in_path):
	def model_names(model_path):
		""" Return model names"""
		model_names = [name for name in os.listdir(model_path)]
		return model_names

	models = model_names(in_path)
	models.sort()
	#print(models)
	out_path = os.path.join(os.path.dirname(in_path), 'names.txt')
	#np.savez(out_path, names = models, test_names=models)
	f = open(out_path, 'w')
	for name in models:
		f.write(name+'\n')
	f.close()
	print('finished {}'.format(in_path))

	print('save to {}'.format(out_path))

def generate_shape_index(in_path):
	def model_names(model_path):
		""" Return model names"""
		model_names = [name for name in os.listdir(model_path)]
		return model_names

	models = model_names(in_path)
	models.sort()
	#print(models)
	all_name_path = os.path.join(os.path.dirname(in_path), 'names.npz')
	all_test_names = np.load(all_name_path)['test_names']
	indexs = []
	for i in range(len(all_test_names)):
		#print('all_test_names[i], ', all_test_names[i])
		#print('model, ', models[i])
		if all_test_names[i]+'.off' in models:
			indexs.append(i)
	print(len(indexs))
	indexs = np.array(indexs)
	out_path = os.path.join(os.path.dirname(in_path), 'fine-tuning_index_all.npz')

	np.savez(out_path, indexes = indexs)
	print('finished {}'.format(in_path))

	print('save to {}'.format(out_path))

def simple_read_names_txt(in_path):
	test_file = os.path.join(in_path, 'names.txt')
	with open(test_file) as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	test_content = [x.strip()[:-4] for x in content]
	out_file = os.path.join(in_path, 'names.npz')

	np.savez(out_file, names = test_content)
	print('save to {}'.format(out_file))

def split_npz(in_path, split_num):
	file_path = os.path.join(in_path, 'names.npz')

	# models = np.load(in_path)['names']

	# models_train = models[:split_num]
	# models_test = models[split_num:]

	models_train = np.load(file_path)['train_names']
	models_test = np.load(file_path)['test_names']

	train_file = os.path.join(in_path, 'train_names.npz')
	test_file = os.path.join(in_path, 'test_names.npz')

	np.savez(train_file, train_names = models_train)
	np.savez(test_file, test_names = models_test)
	#np.savez(in_path, train_names = models_train, test_names = models_test, names=models)
	print('finished {}'.format(in_path))

def read_names_shapenet(in_path):

	#cate = os.path.basename(in_path)#03001627

	#train_file = os.path.join(in_path, '%s_vox256_train.txt'%(cate))
	train_file = os.path.join(in_path, 'train.txt')
	with open(train_file) as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	train_content = [(x.strip()) for x in content]

	test_file = os.path.join(in_path, 'test.txt')
	with open(test_file) as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	test_content = [(x.strip()) for x in content]
	#print(test_content)
	out_path = os.path.join(in_path, 'names.npz')
	np.savez(out_path, train_names = train_content, test_names = test_content, names = train_content + test_content)
	print('finished {}'.format(in_path))

if __name__ == '__main__':
	in_path = sys.argv[1]
	data_flag = int(sys.argv[2])
	#0 split npz, 1 transfer txt to npz, 2 read directory into npz, 3 read shapenet names
	if data_flag == 0:
		split_npz(in_path, split_num=5000)
	elif data_flag == 1:
		simple_read_names_txt(in_path)
	elif data_flag == 2:
		#read names into npz
		Read_directory_to_npz(in_path)
	elif data_flag == 3:
		read_names_shapenet(in_path)
	elif data_flag == 4:
		generate_shape_index(in_path)
	else:
		Read_directory_to_txt(in_path)
	