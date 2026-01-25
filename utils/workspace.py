#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import json
import os
import torch

model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"
latent_codes_subdir = "LatentCodes"
logs_filename = "Logs.pth"
reconstructions_subdir = "Reconstructions"
reconstruction_meshes_subdir = "Meshes"
reconstruction_codes_subdir = "Codes"
interploration_subdir = "Interploration"
specifications_filename = "specs.json"
data_source_map_filename = ".datasources.json"
evaluation_subdir = "Evaluation"
sdf_samples_subdir = "SdfSamples"
surface_samples_subdir = "SurfaceSamples"
normalization_param_subdir = "NormalizationParameters"
training_meshes_subdir = "TrainingMeshes"


def load_experiment_specifications(experiment_directory):

	filename = os.path.join(experiment_directory, specifications_filename)

	if not os.path.isfile(filename):
		raise Exception(
			"The experiment directory ({}) does not include specifications file "
			+ '"specs.json"'.format(experiment_directory)
		)

	return json.load(open(filename))

def load_model_parameters_pcr(experiment_directory, checkpoint, encoder):

	filename = os.path.join(
		experiment_directory, model_params_subdir, checkpoint + ".pth"
	)

	if not os.path.isfile(filename):
		raise Exception('model state dict "{}" does not exist'.format(filename))

	data = torch.load(filename)

	encoder.load_state_dict(data["encoder_state_dict"])
	return data["epoch"]

def save_model_parameters_pcr(experiment_directory, checkpoint, encoder, opt, epoch):

	model_params_dir = get_model_params_dir(experiment_directory, True)

	torch.save(
		{"epoch": epoch,
		"encoder_state_dict": encoder.state_dict(),
		"opt_state_dict": opt.state_dict()},
		os.path.join(model_params_dir, checkpoint),
	)

def save_model_bsp(experiment_directory, filename, encoder, decoder, generator, opt, epoch):

	model_params_dir = get_model_params_dir(experiment_directory, True)

	torch.save(
		{"epoch": epoch,
		"encoder_state_dict": encoder.state_dict(),
		"decoder_state_dict": decoder.state_dict(),
		"generator_state_dict": generator.state_dict(),
		"opt_state_dict": opt.state_dict()},
		os.path.join(model_params_dir, filename),
	)
	
def load_model_parameters_bsp(experiment_directory, checkpoint, encoder, decoder, generator, opt):

	filename = os.path.join(
		experiment_directory, model_params_subdir, checkpoint + ".pth"
	)

	if not os.path.isfile(filename):
		raise Exception('model state dict "{}" does not exist'.format(filename))

	data = torch.load(filename)

	encoder.load_state_dict(data["encoder_state_dict"])
	decoder.load_state_dict(data["decoder_state_dict"])
	generator.load_state_dict(data["generator_state_dict"])
	if opt is not None:
		opt.load_state_dict(data["opt_state_dict"])
	return data["epoch"]

def load_generator_parameters_bsp(experiment_directory, checkpoint, decoder, generator, opt):

	filename = os.path.join(
		experiment_directory, model_params_subdir, checkpoint + ".pth"
	)

	if not os.path.isfile(filename):
		raise Exception('model state dict "{}" does not exist'.format(filename))

	data = torch.load(filename)
	decoder.load_state_dict(data["decoder_state_dict"])
	generator.load_state_dict(data["generator_state_dict"])
	#stage 0 not load opt
	if opt is not None:
		opt.load_state_dict(data["opt_state_dict"])
	return data["epoch"]

def save_model_bsp_shapename(experiment_directory, shapename, filename, decoder, generator, planes, opt, epoch):

	model_params_dir = get_model_params_dir(experiment_directory, True)
	model_params_dir = get_model_params_dir_shapename(model_params_dir, shapename, True)

	torch.save(
		{"epoch": epoch,
		'convex_mask':generator.unused_convex,
		"plane_state_dict": planes,
		"decoder_state_dict": decoder.state_dict(),
		"generator_state_dict": generator.state_dict(),
		"opt_state_dict": opt.state_dict()}, 
		os.path.join(model_params_dir, filename)
	)

def load_model_parameters_bsp_shapename(experiment_directory, shapename, checkpoint, decoder, generator, opt):

	filename = os.path.join(
		experiment_directory, model_params_subdir, shapename, checkpoint + ".pth"
	)

	if not os.path.isfile(filename):
		raise Exception('model state dict "{}" does not exist'.format(filename))

	data = torch.load(filename)

	#test stage no opt
	if opt is not None:
		# print(opt.state_dict())
		# print(data["opt_state_dict"])
		opt.load_state_dict(data["opt_state_dict"])
	decoder.load_state_dict(data["decoder_state_dict"])
	generator.load_state_dict(data["generator_state_dict"])
	if "convex_mask" in data.keys():
		return data["epoch"], data["plane_state_dict"], data["convex_mask"]
	else:
		return data["epoch"], data["plane_state_dict"]


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

	dir = os.path.join(experiment_dir, model_params_subdir)

	if create_if_nonexistent and not os.path.isdir(dir):
		os.makedirs(dir)

	return dir

def get_model_params_dir_shapename(experiment_dir, shapename, create_if_nonexistent=False):

	dir = os.path.join(experiment_dir, shapename)

	if create_if_nonexistent and not os.path.isdir(dir):
		os.makedirs(dir)

	return dir
