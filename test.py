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
from grammar import Grammar 
from networks.dgcnn_cls import DGCNN 

HIER_DATA_PATH='./hier'

def calc_mIoU(labels,preds,grammar):
	ious = []
	for node in grammar.node_map:
		ninds = grammar.node_map[node][0]
		ntype = grammar.node_map[node][1]
		if ntype != 'leaf':
			continue
		#print('ninds,',ninds)
		inter = 0
		union = 0
		for pred, label in zip(preds, labels):
			pn=torch.zeros(pred.shape).int()
			ln=torch.zeros(label.shape).int()
			for ind in ninds:
				pind=(pred ==ind).nonzero().squeeze()
				lind=(label == ind).nonzero().squeeze()
				pn[pind]=1
				ln[lind]=1
				inter += (pn & ln).sum().item()
				union += (pn | ln).sum().item()

			if union == 0:
				continue
			iou=(1.*inter)/(1.*union)
			ious.append(iou)
	miou = torch.tensor(ious).float().mean().item()
	return miou

def calc_seg_miou(pred_label_list,occ_dataset,grammar):
	A_labels = []
	A_preds = []
	for i in range(len(pred_label_list)):
		samp_segments = occ_dataset.segments[i]
		seg_preds = pred_label_list[i]
		gt_labels = occ_dataset.part_labels[i]
		part_num = occ_dataset.part_nums[i]
		samp_preds = torch.zeros(samp_segments.shape[0]).long()-1 
		samp_labels = torch.zeros(samp_segments.shape[0]).long()-1 
	
		for j, (p,g) in enumerate(zip(seg_preds,gt_labels)):
			inds= torch.nonzero((samp_segments == j)).flatten()
			samp_preds[inds]=p.item()
			samp_labels[inds]=g.item()

		assert(samp_preds>=0).all(), 'some label left'
		A_labels.append(samp_labels.cpu())
		A_preds.append(samp_preds.cpu())

	iou = calc_mIoU(A_labels,A_preds,grammar)
	return iou

if __name__ == "__main__":
	#python test_nodes.py -e bed_exp -g 0 --cate bed --split dev
	arg_parser=argparse.ArgumentParser(
	description="test trained model")

	arg_parser.add_argument("--experiment", "-e",
	dest="experiment_directory", required=True,
	help="The experiment directory which includes specifications and saved model ")
	
	arg_parser.add_argument("--checkpoint", "-c",
	dest="checkpoint", default="pretrain",
	help="The checkpoint weights to use. This can be a number indicated an epoch "+
	"or 'latest' for the latest weights (this is the default)")
	
	arg_parser.add_argument(
	"--gpu", "-g", dest="gpu", required=True, help="gpu id")

	arg_parser.add_argument(
	"--start", dest="start", default=0,
	help="start shape index")

	arg_parser.add_argument(
	"--end", dest="end", default=400,
	help="end shape index")
	
	arg_parser.add_argument(
	"--cate", dest="category", required=True, help="category")
	arg_parser.add_argument(
	"--split", dest="split", required=True, help="data split")

	utils.add_common_args(arg_parser)
	args=arg_parser.parse_args()
	
	utils.configure_logging(args)

	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISTBLE_DEVICES"]="%d"%int(args.gpu)

	start_index = int(args.start)
	end_index = int(args.end)

	specs_filename = os.path.join(args.experiment_directory, "specs.json")
	if not os.path.isfile(specs_filename):
		raise Exception('The experiment directory does not include specifications file specs.json')

	specs=json.load(open(specs_filename))
	arch=__import__("networks."+specs["NetworkArch"], fromlist=["DGCNNModel"])

	category = args.category
	max_size = 150

	cate_path=os.path.join(HIER_DATA_PATH,category)
	label_file=f'{cate_path}/level-2.txt'

	file1=open(label_file,'r')
	Lines = file1.readlines()
	max_label= len(Lines)

	print(f' max part size,{max_size}, max_label {max_label} ')
	encoder = arch.DGCNNModel(output_channels=max_label,d_model=512,dropout=0.2,max_part_n=max_size,input_channels=6).cuda()
	
	logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))
	data_source=specs["DataSource"]
	data_source=os.path.join(data_source,category)
	
	occ_dataset = utils.dataloader.GTSamples_abo(data_source, args.split, f'abo_{category}_{args.split}.npz', max_label, max_size)

	saved_model_epoch = ws.load_model_parameters(args.experiment_directory, args.checkpoint, encoder, None)

	print('load epoch: %d'%(saved_model_epoch))
	encoder.eval()
	count = 0
	shape_indexes = list(range(start_index,len(occ_dataset)))

	print('shape indexes all:',shape_indexes)

	avarage_acc=0 
	all_acc=0 
	all_part_count = 0 
	pred_label_list=[]
	start_time = time.time()
	output_dict = {}
	for idx in shape_indexes:
		shapename = occ_dataset.name_list[idx]
		
		points = occ_dataset.part_points[idx:idx+1]
		
		masks = occ_dataset.part_mask[idx:idx+1]
		segments = occ_dataset.segments[idx:idx+1]
		part_num = occ_dataset.part_nums[idx]
		points, masks, segments = points.cuda(), masks.cuda(),segments.cuda()
		start_time = time.time()
		pred = encoder(points,masks,segments)
		pred = torch.softmax(pred,dim = 1)
		pred_l = pred.argmax(dim=1)
		label_list = pred_l.reshape(-1).detach().cpu().tolist()
		
		output_dict[shapename] = label_list[:part_num]

	out_file_name = os.path.join(args.experiment_directory, f'{args.split}.json')
	with open(out_file_name, 'w') as f:
		json.dump(output_dict, f)
	
