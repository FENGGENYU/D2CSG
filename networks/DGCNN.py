import math
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import dataset
from .dgcnn_seg import DGCNN


class DGCNNModel(nn.Module):

    def __init__(self, output_channels: int, d_model: int, dropout: float = 0.2, max_part_n: int = 150, input_channels: int = 3):
        super().__init__()
        
        self.encoder = DGCNN(emb_dims=d_model, input_dim=input_channels)
        self.d_model = d_model
        self.max_part_num = max_part_n
        self.alpha = 0.01
        self.MLP1 = nn.Sequential(nn.Conv1d(d_model*2, 512, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=dropout),
            nn.Conv1d(512, 256, kernel_size=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Conv1d(256, output_channels, kernel_size=1, bias=True)
        )
        self.relu = nn.LeakyReLU(negative_slope=0.001)

    def init_weights(self) -> None:
        initrange = 0.1
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, src: Tensor, src_key_padding_mask: Tensor, segments: Tensor) -> Tensor:
        [B, N, C] = src.size()

        src = src.permute(0, 2, 1)

        src = self.encoder(src)  # DGCNN
        global_src = F.adaptive_max_pool1d(src, 1).view(B, -1)

        # [batch_size,emd,N]
        new_src = []
        max_values = torch.max(segments, 1)[0] + 1
        pad_vector = torch.zeros((1, self.d_model*2)).cuda()
        pad_vector.requires_grad = False

        for i in range(src.size(0)):
            new_f = []
            for j in range(max_values[i]):

                index = torch.nonzero(segments[i] == j).flatten()
                new_feature = torch.index_select(src[i].unsqueeze(0), 2, index)  # 1,emd,N'
                
                newf_max = F.adaptive_max_pool1d(new_feature, 1).view(1, -1)
                newf_max = torch.cat([newf_max, global_src[i].view(1, -1)], 1)
                new_f.append(newf_max)
                
            for j in range(max_values[i], self.max_part_num):
                new_f.append(pad_vector)

            new_f = torch.cat(new_f, 0)  # P,emd
            new_src.append(new_f.unsqueeze(0))

        Fsim = torch.cat(new_src, 0)  # B,P,EMD

        x = Fsim.permute(0, 2, 1)
        label_prob = self.MLP1(x)  # B,C,P

        return label_prob
