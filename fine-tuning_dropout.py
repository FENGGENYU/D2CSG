#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time

import utils
import utils.workspace as ws
from networks.losses import get_composite_loss_compact
import numpy as np

class LearningRateSchedule:
	def get_learning_rate(self, epoch):
		pass


class ConstantLearningRateSchedule(LearningRateSchedule):
	def __init__(self, value):
		self.value = value

	def get_learning_rate(self, epoch):
		return self.value


class StepLearningRateSchedule(LearningRateSchedule):
	def __init__(self, initial, interval, factor):
		self.initial = initial
		self.interval = interval
		self.factor = factor

	def get_learning_rate(self, epoch):

		return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
	def __init__(self, initial, warmed_up, length):
		self.initial = initial
		self.warmed_up = warmed_up
		self.length = length

	def get_learning_rate(self, epoch):
		if epoch > self.length:
			return self.warmed_up
		return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

	schedule_specs = specs["LearningRateSchedule"]

	schedules = []

	for schedule_specs in schedule_specs:

		if schedule_specs["Type"] == "Step":
			schedules.append(
				StepLearningRateSchedule(
					schedule_specs["Initial"],
					schedule_specs["Interval"],
					schedule_specs["Factor"],
				)
			)
		elif schedule_specs["Type"] == "Warmup":
			schedules.append(
				WarmupLearningRateSchedule(
					schedule_specs["Initial"],
					schedule_specs["Final"],
					schedule_specs["Length"],
				)
			)
		elif schedule_specs["Type"] == "Constant":
			schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

		else:
			raise Exception(
				'no known learning rate schedule of type "{}"'.format(
					schedule_specs["Type"]
				)
			)

	return schedules

def save_logs(
	experiment_directory,
	loss_log,
	lr_log,
	timing_log,
	lat_mag_log,
	param_mag_log,
	epoch,
):

	torch.save(
		{
			"epoch": epoch,
			"loss": loss_log,
			"learning_rate": lr_log,
			"timing": timing_log,
			"latent_magnitude": lat_mag_log,
			"param_magnitude": param_mag_log,
		},
		os.path.join(experiment_directory, ws.logs_filename),
	)


def load_logs(experiment_directory):

	full_filename = os.path.join(experiment_directory, ws.logs_filename)

	if not os.path.isfile(full_filename):
		raise Exception('log file "{}" does not exist'.format(full_filename))

	data = torch.load(full_filename)

	return (
		data["loss"],
		data["learning_rate"],
		data["timing"],
		data["latent_magnitude"],
		data["param_magnitude"],
		data["epoch"],
	)

def get_spec_with_default(specs, key, default):
	try:
		return specs[key]
	except KeyError:
		return default

def init_seeds(seed=0):
	print('seed, ', seed)
	torch.manual_seed(seed) # sets the seed for generating random numbers.
	torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
	torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
	#torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True
#-------------------------------------------------------------
def main_function(start_id, end_id, batch_size, experiment_directory, \
	continue_from, grid_sample, leaky, test_flag, sample_surface, \
	sample_voxel, shapenet_flag, no_weight, initial_flag, seed, mid_results, drop_threshould = 1, p_c = 512, c_c = 64):

	
	#--------------------------------
	#exp name
	logging.debug("running " + experiment_directory)

	specs = ws.load_experiment_specifications(experiment_directory)
	logging.info("Experiment description: \n" + specs["Description"])
	#----------------------------------
	
	lr_schedules = get_learning_rate_schedules(specs)
	#------------------------------------------------------------------
	#extra functions
	def save_checkpoints_best_with_threshold(epoch):

		ws.save_model_bsp_shapename(experiment_directory, shapename, "best_stage%d_%d_%d.pth"%(phase, grid_sample, drop_threshould), decoder, generator, planes_code, optimizer_all, epoch)
	
	def save_checkpoints_best(epoch):

		ws.save_model_bsp_shapename(experiment_directory, shapename, "best_stage%d_%d.pth"%(phase, grid_sample), decoder, generator, planes_code, optimizer_all, epoch)
	
	def save_checkpoints(epoch):

		ws.save_model_bsp_shapename(experiment_directory, shapename, "last_%d.pth"%(phase), decoder, generator, planes_code, optimizer_all, epoch)
	
	def signal_handler(sig, frame):
		logging.info("Stopping early...")
		sys.exit(0)

	def adjust_learning_rate(lr_schedules, optimizer, epoch):

		for i, param_group in enumerate(optimizer.param_groups):
			param_group["lr"] = lr_schedules[0].get_learning_rate(epoch)

	signal.signal(signal.SIGINT, signal_handler)
	#----------------------------------------------------------------------
	#load network
	primi_count = p_c#256
	convex_count = c_c

	arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder", "Generator"])
	#encoder = arch.Encoder().cuda()
	# decoder = arch.Decoder(p_dim = primi_count).cuda()
	# generator = arch.Generator(p_dim = primi_count, c_dim=convex_count).cuda()
	# print(generator.state_dict())
	# exit(0)
	#-----------------------------------------------------------------------
	#log_frequency = get_spec_with_default(specs, "LogFrequency", 50)
	#------------------------------------------
	#load data
	data_source = specs["DataSource"]
	if sample_surface:
		sdf_dataset = utils.dataloader.SurfaceSamples(
			data_source, test_flag=test_flag
		)
	elif sample_voxel:
		sdf_dataset = utils.dataloader.VoxelSamples(
			data_source, test_flag=test_flag
		)
	else:
		sdf_dataset = utils.dataloader.SDFSamples(
			data_source, grid_sample=grid_sample, test_flag=test_flag, shapenet_flag=shapenet_flag
		)
	#------------------------------------------
	#set fine-tune shape indexes
	#shape_indexes = np.load(os.path.join(data_source, 'fine-tuning_index.npz'))['index']#[0, 1, 2]
	#shape_indexes = shape_indexes[3:4]
	#shape_indexes = [7]#[224, 271, 276, 282, 297, 365, 377, 378]
	#shapenet 
    #plane, bench, 
	if shapenet_flag:
		cate_indexes = [0, 809, 1173, 1488, 2988, 4344, 4563, 5027, 5351, 5826, 6461, 8163, 8374]
		cate_shape_indexes = []
		for i in cate_indexes:
			cate_shape_indexes = cate_shape_indexes + list(range(i, i+50))
		shape_indexes = cate_shape_indexes[start_id: end_id]
	else:
		shape_indexes = list(range(start_id, end_id))

	shape_indexes = list(range(start_id, end_id))
	print('shape indexes all: ', shape_indexes)
	num_scenes = len(sdf_dataset)

	logging.info("There are {} shapes loaded".format(num_scenes))
	logging.info("There are {} shapes to be fine-tuned".format(len(shape_indexes)))
	
	#---------------------------------------------------------

	load_point_size = sdf_dataset.data_points.shape[1]
	#point_batch_num = int(load_point_batch_size/point_batch_size)
	point_batch_num = batch_size
	point_batch_size = int(load_point_size/point_batch_num)
	print('point batch num, ', point_batch_num)
	print('point_batch_size, ', point_batch_size)
	#---------------------------------------------------------
	#set fine-tuning parameters
	iterations = 40
	stages = [0,1,2,3]

	logging.info("Parameters: test flag {}, expriment {}, grid_sample {}, leaky {}, shapenet {} noweight {}".format(test_flag, \
		experiment_directory, grid_sample, leaky, shapenet_flag, no_weight))
	logging.info("drop {}, primi count {}, convex count {}".format(drop_threshould, \
		p_c, c_c))

	#unused_convex = torch.zeros(convex_count).cuda()
	
	#----------------------------------
	#start fine-tuning
	for index in shape_indexes:
		init_seeds(seed)
		planes_code = torch.ones(1, 256).normal_(mean=0, std=0.01).cuda()
		planes_code.requires_grad = True

		decoder = arch.Decoder(p_dim = primi_count).cuda()
		generator = arch.Generator(p_dim = primi_count, c_dim=convex_count).cuda()
		print('fine-tuning shape index', index)
		optimizer_all = torch.optim.Adam(
			[
				{
					"params": planes_code,
					"lr": lr_schedules[0].get_learning_rate(0),
					"betas": (0.5, 0.999),
				},
				{
					"params": decoder.parameters(),
					"lr": lr_schedules[0].get_learning_rate(0),
					"betas": (0.5, 0.999),
				},
				{
					"params": generator.parameters(),
					"lr": lr_schedules[0].get_learning_rate(0),
					"betas": (0.5, 0.999),
				},
			]
		)
		# for name, param in generator.named_parameters():
		# 	print(name)
		#------------------------------------------------------------
		#read per shape data
		shapename = sdf_dataset.data_names[index]
		print('fine-tuning shape name', shapename)
		sdf_data = sdf_dataset.data_points[index].cuda()
		sdf_data = sdf_data.unsqueeze(0)
		surface_points = None #sdf_dataset.surface_points[index:index+1].cuda()
		if sample_voxel:
			voxels = sdf_dataset.data_voxels[index].unsqueeze(0).cuda()
		#------------------------------------------------------------
		if mid_results:
			reconstruction_dir = os.path.join(experiment_directory, 'tmp_recon', shapename)
			if not os.path.exists(reconstruction_dir):
				os.makedirs(reconstruction_dir)

		for phase in stages:
			print('phase , ', phase)
			if phase == 3:
				num_epochs = 1000
			else:
				num_epochs = 500
			#------------------------------------------
			#load shape code and network weights
			if phase == 0:
				model_epoch = 0
			else:
				continue_from = "best_stage%d_%d"%(phase-1, grid_sample)
				#continue_from = "best_stage%d_%d_%d.pth"%(phase-1, grid_sample, drop_threshould)
				#continue_from = "last_%d"%(phase-1)
				logging.info('continuing from "{}"'.format(continue_from))
				results = ws.load_model_parameters_bsp_shapename(
					experiment_directory, shapename, continue_from, 
					decoder,
					generator,
					optimizer_all
				)
				if len(results) == 3: 
					model_epoch, planes_code, generator.unused_convex = results
				else:
					model_epoch, planes_code = results
					
				print('planes_code loaded, ', planes_code.shape)
				start_epoch = model_epoch +1 
				logging.debug("loaded")

			#------------------------------------------
			best_loss = 999
			start_epoch = model_epoch +1 
			logging.info("starting from epoch {}".format(start_epoch))
			
			start_time = time.time()
			last_epoch_time = 0
			drop_out_times = 1
			for epoch in range(start_epoch, start_epoch + num_epochs):
				#------------------------------------------------
				#calculate drop out primitives and convex shapes
				if phase == 3 and (epoch - start_epoch) % 200 == 0:
					best_loss = 999
					if epoch - start_epoch == 0:
						continue_from = "best_stage%d_%d"%(phase-1, grid_sample)
					else:
						continue_from = "best_stage%d_%d"%(phase, grid_sample)
						#continue_from = "best_stage%d_%d_%d"%(phase, grid_sample, drop_threshould)
					#continue_from = "last_%d"%(phase-1)
					logging.info('continuing from "{}"'.format(continue_from))
					results = ws.load_model_parameters_bsp_shapename(
						experiment_directory, shapename, continue_from, 
						decoder,
						generator,
						optimizer_all
					)
					if len(results) == 3: 
						model_epoch, planes_code, generator.unused_convex = results
					else:
						model_epoch, planes_code = results
						
					print('planes_code loaded, ', planes_code.shape)
					logging.debug(f"loaded epoch {model_epoch}")

					decoder.eval()
					generator.eval()
					primitives = decoder(planes_code.cuda())
					# print('primitives, ', primitives)
					# print('generator, ', generator.convex_layer_weights)
					#100 iteration update selection matrix and convex mask
					if mid_results:
						mesh_filename = os.path.join(reconstruction_dir, f'mc_{epoch - start_epoch}')
						utils.mesh_global.create_mesh_mc(
							generator, None, phase, primitives, mesh_filename, N=128, max_batch=int(2 ** 14), 
							threshold=0.9
						)
					print('drop_threshould, ', drop_threshould)
					print('drop_out_times, ', drop_out_times)

					generator.convex_layer_weights.data, generator.concave_layer_weights.data, generator.unused_convex, stop_flag = utils.mesh_global.dualcsg_mask(
						generator, primitives, 
						connection_t_convex=generator.convex_layer_weights.detach(), 
						connection_t_concave=generator.concave_layer_weights.detach(), 
						p_dim=primi_count, c_dim=convex_count, N=128, 
						max_batch=int(2 ** 14), unused_convex=generator.unused_convex, filter_threshold = drop_threshould*drop_out_times
					)
					if stop_flag:
						#save_checkpoints_best(epoch)
						print('early stop')
						break
					else:
						drop_out_times += 1
						
				#--------------------------------------------------
				decoder.train()
				generator.train()
				adjust_learning_rate(lr_schedules, optimizer_all, epoch - start_epoch)
				#adjust_learning_rate(lr_schedules, optimizer_code, epoch - start_epoch)

				avarage_c_loss = 0
				avarage_l1_loss = 0
				avarage_g_loss = 0
				avarage_ga_loss = 0
				avarage_total_loss = 0
				avarage_num = 0

				for itera in range(iterations):

					which_batch = torch.randint(point_batch_num, (1,))

					xyz = sdf_data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, :3]
					sdf_gt = sdf_data[:,which_batch*point_batch_size:(which_batch+1)*point_batch_size, 3]
					
					optimizer_all.zero_grad()
					#optimizer_code.zero_grad()

					primitives = decoder(planes_code.cuda())
					
					primi_mask, G, G_a, net_out, net_out_convexes = generator(surface_points, xyz, primitives, phase, leaky)
					
					c_loss, l1_loss, value_loss, value_loss_pa, chunk_loss = get_composite_loss_compact(primi_mask, shapenet_flag, 
						phase, net_out_convexes, net_out, G, G_a, sdf_gt, 
						shape_layer = generator.union_layer_weights, convex_layer = generator.convex_layer_weights, 
						concave_layer = generator.concave_layer_weights, no_weight = no_weight, primitives=primitives)
					
					
					chunk_loss.backward()
					optimizer_all.step()
					#optimizer_code.step()

					avarage_c_loss += c_loss.detach().item()
					avarage_l1_loss += l1_loss.detach().item()
					avarage_g_loss += value_loss.detach().item()
					avarage_ga_loss += value_loss_pa.detach().item()
					avarage_total_loss += chunk_loss.detach().item()
					avarage_num += 1
				
				#---------------------------------------------------
				#print loss
				if (epoch - start_epoch+1) % 10 == 0:
					end = time.time()
					seconds_elapsed = end - start_time
					ava_epoch_time = (seconds_elapsed - last_epoch_time)/10
					left_time = ava_epoch_time*(num_epochs+ start_epoch- epoch)/3600
					last_epoch_time = seconds_elapsed
					c_loss = avarage_c_loss/avarage_num
					l1_loss = avarage_l1_loss/avarage_num
					g_loss = avarage_g_loss/avarage_num
					ga_loss = avarage_ga_loss/avarage_num
					t_loss = avarage_total_loss/avarage_num
					logging.debug("epoch = {}/{} closs = {:.6f}, l1 = {:.6f}, errSP = {:.6f},  errsp+pa = {:.6f}, total_loss={:.6f}, 1 epoch time ={:.6f}, left time={:.6f}".format(epoch, num_epochs+start_epoch, c_loss, l1_loss, g_loss, ga_loss, t_loss, ava_epoch_time, left_time))

					if g_loss + ga_loss < best_loss:
						print('best loss updated, ', g_loss + ga_loss)
						save_checkpoints_best(epoch)
						best_loss = g_loss + ga_loss
				if (epoch - start_epoch +1) % num_epochs == 0:
					save_checkpoints(epoch)

			print('stage time:, ', time.time() - start_time)

if __name__ == "__main__":
	
	import argparse

	arg_parser = argparse.ArgumentParser(description="Train a BSP")
	arg_parser.add_argument(
		"--experiment",
		"-e",
		dest="experiment_directory",
		required=True,
		help="The experiment directory",
	)
	arg_parser.add_argument(
		"--continue",
		"-c",
		dest="continue_from",
		help="A snapshot to continue from. This can be 'latest' to continue"
		+ "from the latest running snapshot, or an integer corresponding to "
		+ "an epochal snapshot.",
	)
	arg_parser.add_argument(
		"--leaky",
		dest="leaky",
		action="store_false",
		help="dataset option",
	)
	arg_parser.add_argument(
		"--initial",
		dest="initial",
		action="store_true",
		help="whether there is pre-trained weigths or not",
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
		"--noweight",
		dest="noweight",
		action="store_true",
		help="whether use weights for loss",
	)
	arg_parser.add_argument(
		"--test",
		dest="test",
		action="store_true",
		help="train or test data indicator",
	)
	arg_parser.add_argument(
		"--grid_sample",
		dest="grid_sample",
		default=64,
		help="dataset resolution",
	)
	arg_parser.add_argument(
		"--batch_size",
		dest="batch_size",
		default= 2,
		help="batch size for train and test",
	)
	arg_parser.add_argument(
		"--p_c",
		dest="p_c",
		default= 1024,
		help="primi_count",
	)
	arg_parser.add_argument(
		"--c_c",
		dest="c_c",
		default= 64,
		help="convex_count",
	)
	arg_parser.add_argument(
		"--drop",
		dest="drop",
		default= 3,
		help="dropout",
	)
	arg_parser.add_argument(
		"--seed",
		dest="seed",
		default= 0,
		help="dataset option",
	)
	arg_parser.add_argument(
		"--pcr",
		dest="pcr",
		action="store_true",
		help="point cloud data",
	)
	arg_parser.add_argument(
		"--voxel",
		dest="voxel",
		action="store_true",
		help="voxel data",
	)
	arg_parser.add_argument(
		"--shapenet_flag",
		dest="shapenet_flag",
		action="store_true",
		help="abc or shapenet indicator",
	)
	arg_parser.add_argument(
		"--mid_results",
		dest="mid_results",
		action="store_true",
		help="output tmp mid results or not, for debug",
	)
	arg_parser.add_argument(
		"--gpu",
		"-g",
		dest="gpu",
		required=True,
		help="gpu id",
	)
	utils.add_common_args(arg_parser)

	args = arg_parser.parse_args()

	utils.configure_logging(args)
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="%d"%int(args.gpu)
	print('gpu: ,', int(args.gpu))
	main_function(int(args.start), int(args.end), int(args.batch_size), args.experiment_directory, \
		args.continue_from, int(args.grid_sample), args.leaky, args.test, \
		args.pcr, args.voxel, args.shapenet_flag, args.noweight, args.initial, int(args.seed), args.mid_results, int(args.drop), int(args.p_c), int(args.c_c))
