import numpy as np
import cv2
import os
import h5py
import mcubes
import argparse

#require ./data/data_name/shapes
#python bin_voxelization/1_voxelize.py 0 1 /local-scratch2/fenggeny/SECAD-Net-main/data/mingrui_data/shapes

parser = argparse.ArgumentParser()
parser.add_argument("share_id", type=int, help="id of the share [0]")
parser.add_argument("share_total", type=int, help="total num of shares [1]")
parser.add_argument("target_dir", type=str, help="total num of shares [1]")
FLAGS = parser.parse_args()

target_dir = FLAGS.target_dir + '/'
if not os.path.exists(target_dir):
    print("ERROR: this dir does not exist: "+target_dir)
    exit()

obj_names = os.listdir(target_dir)
obj_names = sorted(obj_names)

share_id = FLAGS.share_id
share_total = FLAGS.share_total

start = int(share_id*len(obj_names)/share_total)
end = int((share_id+1)*len(obj_names)/share_total)
obj_names = obj_names[start:end]


for i in range(len(obj_names)):
    this_name = os.path.join(target_dir,obj_names[i]+'/model_normalized.obj')
    print(i,this_name)

    command = "./bin_voxelization/binvox -bb -0.5 -0.5 -0.5 0.5 0.5 0.5 -d 1024 -e "+this_name
    #command = "./binvox -d 1024 -e "+this_name
    os.system(command)

