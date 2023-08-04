import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import h5py
import utils.workspace as ws
import pickle
import torch.nn.functional as F


class GTSamples_abo(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        data_split,
        data_name,
        max_label,
        max_size
    ):
        print('data source', data_source)
        self.data_source = data_source
        print('class Samples from GT meshes')
        print('data_name', data_name)

        train_data = np.load(os.path.join(self.data_source, data_name))

        name_list = train_data['names']
        part_num = train_data['part_count']
        part_pcs =	train_data['points']
        samp_segments =	train_data['segments']
        shape_num = part_pcs.shape[0]
        self.train_flag = False

        if data_split == 'train':
            self.train_flag = True
            labels = train_data['part_labels']
            self.part_labels = torch.from_numpy(labels).long()
        
        # -----------------------------------------------------------------------------------------------------------------------------------------
        masks = np.ones((part_pcs.shape[0], max_size), dtype=bool)
        for i in range(shape_num):
            masks[i, :part_num[i]] = False

        self.part_points = torch.from_numpy(part_pcs).float()
        self.part_mask = torch.from_numpy(masks).bool()  # B,P
        self.segments = torch.from_numpy(samp_segments).long()
        self.part_nums = torch.from_numpy(part_num).long()

        self.name_list = name_list

        print(f'load shape,{self.part_points.shape[0]}, data name {data_name}')

    def __len__(self):
        return len(self.part_points)

    def __getitem__(self, idx):
        if self.train_flag:
            return self.part_points[idx], self.part_labels[idx], self.part_mask[idx], self.segments[idx], self.part_nums[idx], self.name_list[idx]
        else:
            return self.part_points[idx], self.part_mask[idx], self.segments[idx], self.part_nums[idx], self.name_list[idx]
        

