import argparse
import json
import logging
import os
import random
import time
import torch

import utils
import utils.workspace as ws
import torch.utils.data as data_utils
import numpy as np

if __name__ == "__main__":
	#python fine-tune_test_batch.py -e simple_5000_ae -g 0 -p 2 -c best_stage2_64 --test

	#python fine-tune_test_batch_compact.py -e v0.0 -g 0 -p 2 -c best_stage2_64 --test --start 2 --end 3 --csg --primi
	#python fine-tune_test_batch_compact.py -e v0.0_capri_sym -g 0 -p 2 -c best_stage2_64 --test --start 2 --end 3 --csg --primi

	#python fine-tune_test_batch_generalized_inverse.py -e v0.0_generalized_inverse_voxel_1000 --voxel -g 0 -p 2 -c best_stage2_64 --test --start 0 --end 1000
	#python fine-tune_test_batch_generalized_inverse.py -e v0.0_generalized_inverse_voxel_0004 --voxel -g 0 -p 2 -c best_stage2_64 --test --start 0 --end 125
	#python fine-tune_test_batch_generalized_inverse.py -e dualcsg_pretrain -g 0 -p 2 -c best_stage2_64 --test --start 0 --end 10
	##python test.py -e dualcsg_convex -g 0 -p 3 -c best_stage3_64 --test --start 10 --end 500 --csg
	#python test.py -e dualcsg_abc_rerun -g 0 -p 0 -c best_stage0_64 --test --start 0 --end 500 --mc_threshold 0.5
	##python test.py -e dualcsg -g 0 -p 3 -c best_stage3_64 --test --start 0 --end 1 --csg

	arg_parser = argparse.ArgumentParser(
		description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
		+ "samples."
	)
	arg_parser.add_argument(
		"--experiment",
		"-e",
		dest="experiment_directory",
		required=True,
		help="The experiment directory which includes specifications and saved model "
		+ "files to use for reconstruction",
	)
	arg_parser.add_argument(
		"--checkpoint",
		"-c",
		dest="checkpoint",
		default="latest",
		help="The checkpoint weights to use. This can be a number indicated an epoch "
		+ "or 'latest' for the latest weights (this is the default)",
	)
	arg_parser.add_argument(
		"--grid_sample",
		dest="grid_sample",
		default=64,
		help="dataset option",
	)
	arg_parser.add_argument(
		"--start",
		dest="start",
		default=0,
		help="dataset option",
	)
	arg_parser.add_argument(
		"--end",
		dest="end",
		default=1,
		help="dataset option",
	)
	arg_parser.add_argument(
		"--mc_shape",
		dest="mc_shape",
		default=0,
		help="dataset option",
	)
	arg_parser.add_argument(
		"--mc_threshold",
		dest="mc_threshold",
		default=0.9,
		help="dataset option",
	)
	arg_parser.add_argument(
		"--mc_res",
		dest="mc_res",
		default=256,
		help="mc_res",
	)
	arg_parser.add_argument(
		"--filter_threshold",
		dest="filter_threshold",
		default=20,
		help="filter_threshold",
	)
	arg_parser.add_argument(
		"--concave_level_set",
		dest="concave_level_set",
		default=0.1,
		help="dataset option",
	)

	arg_parser.add_argument(
		"--csg",
		dest="csg",
		action="store_true",
		help="output csg mesh.",
	)

	arg_parser.add_argument(
		"--gpu",
		"-g",
		dest="gpu",
		required=True,
		help="gpu id",
	)
	arg_parser.add_argument(
		"--phase",
		"-p",
		dest="phase",
		required=True,
		help="phase stage",
	)
	arg_parser.add_argument(
		"--shapenet",
		dest="shapenet",
		action="store_true",
		help="dataset option",
	)
	arg_parser.add_argument(
		"--test",
		dest="test_flag",
		action="store_true",
		help="dataset option",
	)
	arg_parser.add_argument(
		"--scad",
		dest="scad_flag",
		action="store_true",
		help="scad option",
	)
	arg_parser.add_argument(
		"--code",
		dest="save_code",
		action="store_true",
		help="dataset option",
	)
	arg_parser.add_argument(
		"--primi_count",
		dest="primi_count",
		action="store_true",
		help="dataset option",
	)
	arg_parser.add_argument(
		"--sample_flag",
		dest="sample_flag",
		action="store_true",
		help="dataset option",
	)
	arg_parser.add_argument(
		"--primi",
		dest="primi",
		action="store_true",
		help="dataset option",
	)
	arg_parser.add_argument(
		"--surface",
		dest="surface",
		action="store_true",
		help="dataset option",
	)
	arg_parser.add_argument(
		"--voxel",
		dest="voxel",
		action="store_true",
		help="dataset option",
	)
	utils.add_common_args(arg_parser)

	args = arg_parser.parse_args()

	utils.configure_logging(args)
	#-----------------------------------------------------
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="%d"%int(args.gpu)
	concave_level_set = float(args.concave_level_set)
	filter_threshold = float(args.filter_threshold)
	phase = int(args.phase)
	start_index = int(args.start)
	end_index = int(args.end)
	csg = int(args.csg)
	grid_sample = int(args.grid_sample)
	mc_threshold = float(args.mc_threshold)
	mc_res = int(args.mc_res)
	mc_shape = int(args.mc_shape)

	print(f'parameters: mc_threshold {mc_threshold}, mc_res {mc_res}, filter_threshold {filter_threshold}')
	print(f'parameters: phase {phase}, grid_sample {grid_sample}, concave_level_set {concave_level_set}')
	print(f'parameters: csg {csg}')
	#----------------------------------------------------------
	specs_filename = os.path.join(args.experiment_directory, "specs.json")

	if not os.path.isfile(specs_filename):
		raise Exception(
			'The experiment directory does not include specifications file "specs.json"'
		)

	specs = json.load(open(specs_filename))
	#-----------------------------------------------------
	arch = __import__("networks." + specs["NetworkArch"], fromlist=["Encoder", "Decoder", "Generator"])
	primi_count = 1024
	convex_count = 64
	
	logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

	#-----------------------------------------------------
	data_source = specs["DataSource"]
	#random.shuffle(npz_filenames)
	#print(single)
	if args.surface:
		sdf_dataset = utils.dataloader.SurfaceSamples(
			data_source, test_flag=args.test_flag
		)
	elif args.voxel:
		sdf_dataset = utils.dataloader.VoxelSamples(
			data_source, test_flag=args.test_flag
		)
	else:
		sdf_dataset = utils.dataloader.SDFSamples(
			data_source, grid_sample=grid_sample, test_flag=args.test_flag, shapenet_flag=args.shapenet
		)

	#-----------------------------------------------------------------
	# logging.debug(decoder)
	# logging.debug(generator)
	
	if args.save_code:
		code_dir = os.path.join(
			args.experiment_directory, ws.latent_codes_subdir
		)	
		if not os.path.isdir(code_dir):
			os.makedirs(code_dir)
		save_code_file = os.path.join(
			code_dir, 'code_%d.pth'%saved_model_epoch
		)	
	else:
		if args.test_flag:
			ws.reconstructions_subdir = ws.reconstructions_subdir + '_test'
		
		if csg:
			if args.primi:
				reconstruction_dir = os.path.join(
					args.experiment_directory, ws.reconstructions_subdir+'_csg_primi'
				)
			else:
				reconstruction_dir = os.path.join(
					args.experiment_directory, ws.reconstructions_subdir+'_csg'
				)
		else:
			reconstruction_dir = os.path.join(
				args.experiment_directory, ws.reconstructions_subdir, 'phase' + args.phase
			)

		if not os.path.isdir(reconstruction_dir):
			os.makedirs(reconstruction_dir)
	#---------------------------------------------------------------
	count = 0
	
	all_primitive_codes = []

	avarage_primitive_count = 0
	avarage_convex_count = 0
	#absolute_index = 50
	if args.shapenet:
		cate_indexes = [0, 809, 1173, 1488, 2988, 4344, 4563, 5027, 5351, 5826, 6461, 8163, 8374]
		cate_shape_indexes = []
		for i in cate_indexes:
			cate_shape_indexes = cate_shape_indexes + list(range(i, i+50))
		shape_indexes = cate_shape_indexes[start_index: end_index]
	else:
		#shape_indexes = list(range(0, 125))
		shape_indexes = list(range(start_index, end_index))
		#shape_indexes = [1, 48, 119, 170]
		#shape_indexes = [36, 59, 108, 111, 121, 142, 158, 166, 167, 172, 175, 180]
		#shape_indexes = [43, 45, 51, 53, 170] #50, #45015 33319 31353 31759 32998
		#shape_indexes = [181]
	#shape_indexes = [15, 18, 20, 175] #50,
	#shape_indexes = [2989, 4575, 4582, 4590, 4607, 5068, 5069]
	#shape_indexes = [819, 1199, 1200, 1497, 1507, 6466, 8419]
	#shape_indexes = [1507]
	#shape_indexes = [822, 834, 835, 1222, 1499, 1514, 1516, 1522, 2989, 2993, 2999, 3017, 4381, 4566, 4568, 4604, 5033, 5059, 5075, 5379, 5387, 5834, 5837, 5848, 6493, 6504] #50,
	#shape_indexes = [9, 829, 1211, 1221, 1518, 3016, 3028, 4354, 4391, 4568, 4578, 5028, 5834, 6477, 8392, 8409]
	#shape_indexes = [4583, 4586, 4598]
	shape_indexes = [5, 12, 18, 43, 49, 59, 66]
	print('shape indexes all: ', shape_indexes)
	#--------------------------------------------------------
	if args.primi_count:
		column_labels = ['mname', 'convex', 'primi']
		log_fn = os.path.join(args.experiment_directory, ws.reconstructions_subdir, 'phase' + args.phase +f'_primi_count_{int(filter_threshold)}.csv')
		fid = open(log_fn, 'w')
		# fid.write('gt_path:%s\n'%(gt_path))
		# fid.write('mesh_path:%s\n'%(mesh_path))
		# fid.write('sampling_path:%s\n'%(sampling_path))
		# fid.write('----------------------------------------------------------------\n')
		for label in column_labels:
			fid.write('%s,'%label)
		fid.write('\n')
		fid.close()

	#for index_bias in shape_indexes_bias:
	for index in shape_indexes:
		decoder = arch.Decoder(p_dim = primi_count).cuda()
		generator = arch.Generator(p_dim = primi_count, c_dim=convex_count).cuda()
		#index = shape_indexes[index_bias]
		shapename = sdf_dataset.data_names[index]

		surface_points = None #sdf_dataset.surface_points[index:index+1]
		samples_points = sdf_dataset.data_points[index, :, :3]
		print(f'index, {index}, shape name, {shapename}, phase {phase}')
		# print('absolute_index, ', absolute_index)
		# absolute_index += 1

		results = ws.load_model_parameters_bsp_shapename(
				args.experiment_directory, shapename, args.checkpoint,
				decoder, 
				generator,
				None,
			)
		if len(results) == 3: 
			saved_model_epoch, planes_code, generator.unused_convex = results
		else:
			saved_model_epoch, planes_code = results
		print('load epoch: %d'%(saved_model_epoch))
		decoder.eval()
		generator.eval()

		start_time = time.time()
		
		primitives = decoder(planes_code.cuda())

		if args.primi_count == False:
			if args.save_code:
				all_primitive_codes.append(planes_code.detach())
			else:
				start_time = time.time()
				mesh_filename = os.path.join(reconstruction_dir, shapename + f'_stage_{phase}')
				#mesh_filename = os.path.join(reconstruction_dir, npz[:-4])
				if csg:
					if phase == 3:
						utils.mesh_global.create_mesh_dualcsg_mask(
							generator, primitives,
							connection_t_convex=generator.convex_layer_weights.detach(), 
							connection_t_concave=generator.concave_layer_weights.detach(), 
							unused_convex = generator.unused_convex,
							filename = mesh_filename, p_dim=primi_count, c_dim=convex_count, N=128, 
							max_batch=int(2 ** 14), primi_count = False, 
							filter_threshold = filter_threshold, concave_level_set = concave_level_set, 
							primi_out = args.primi, scad_flag = args.scad_flag
						)
					elif phase == 2:
						utils.mesh_global.create_mesh_csg_primi_compact_one_primi(
							generator, surface_points, samples_points, primitives, 
							connection_t_convex=generator.convex_layer_weights.detach(), 
							connection_t_concave=generator.concave_layer_weights.detach(), 
							filename = mesh_filename, p_dim=primi_count, c_dim=convex_count, N=128, 
							max_batch=int(2 ** 14), primi_count = False,
							filter_threshold = filter_threshold, concave_level_set = concave_level_set, primi_out = args.primi
						)
					else:
						print('This stage does not support csg output')
				else:
					utils.mesh_global.create_mesh_mc(
						generator, surface_points, phase, primitives, mesh_filename, N=mc_res, max_batch=int(2 ** 14), 
							threshold=mc_threshold, mc_shape = mc_shape
					)
				logging.debug("reconstruct time: {}".format(time.time() - start_time))
		else:
			mesh_filename = os.path.join(reconstruction_dir + f'_{int(filter_threshold)}', shapename + f'_stage_{phase}')
			# print('primitives 1, ', primitives)
			# print('connection_t_convex, ', generator.convex_layer_weights.shape)
			# print('connection_t_concave, ', generator.concave_layer_weights.shape)

			# c_count, p_count_left, p_count_right, connection_t_convex_out, connection_t_concave_out, convex_mask = utils.mesh_global.create_mesh_csg_primi_compact_one_primi(
			# 		generator, surface_points, samples_points, primitives, 
			# 		connection_t_convex=generator.convex_layer_weights.detach(), 
			# 		connection_t_concave=generator.concave_layer_weights.detach(), 
			# 		p_dim=primi_count, c_dim=convex_count, N=mc_res, 
			# 		max_batch=int(2 ** 14), primi_count = True, differ=( 1- nodiffer), filter_threshold = filter_threshold, concave_level_set = concave_level_set
			# 	)
			c_count, p_count_left, p_count_right, _, _, _ = utils.mesh_global.create_mesh_dualcsg_mask(
					generator, primitives,
					connection_t_convex=generator.convex_layer_weights.detach(), 
					connection_t_concave=generator.concave_layer_weights.detach(), 
					unused_convex = generator.unused_convex,
					filename = mesh_filename, p_dim=primi_count, c_dim=convex_count, N=128, 
					max_batch=int(2 ** 14), primi_count = True, 
					filter_threshold = 20, concave_level_set = concave_level_set, primi_out = args.primi
				)
			# print the log
			fid = open(log_fn, 'a')
			fid.write(f'{shapename},{c_count},{p_count_left},{p_count_right}')
			fid.write('\n')
			fid.close()

			p_count = p_count_left + p_count_right
			avarage_primitive_count += p_count
			avarage_convex_count += c_count

			# utils.mesh_global.create_mesh_mc_given_weights(convex_mask, connection_t_convex_out, connection_t_concave_out, phase, 
			# 	primitives, mesh_filename, N=mc_res, max_batch=int(2 ** 14), threshold=mc_threshold)

		count += 1
	if args.primi_count:
		print('avarage_primitive_count, {}, avarage_convex_count  {}, '.format(avarage_primitive_count/count, avarage_convex_count/count))
		fid = open(log_fn, 'a')
		fid.write(f'all,{avarage_convex_count/count},{avarage_primitive_count/count}')
		fid.write('\n')
		for label in column_labels:
			fid.write('%s,'%label)
		fid.write('\n')
		fid.close()
	if args.save_code:
		all_primitive_codes = torch.cat(all_primitive_codes, 0)
		torch.save(
			{"primitive_code": all_primitive_codes.detach()},
			save_code_file,
		)
