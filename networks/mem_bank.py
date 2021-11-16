import h5py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('../')
from networks.q_v_transformer import padding_mask_k, padding_mask_q
from networks import torchnlp_nn as nlpnn

class AttentionScore(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)

        # self.linear_q = nlpnn.WeightDropLinear(
        #     hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        # self.linear_k = nlpnn.WeightDropLinear(
        #     hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k,  scale=None, attn_mask=None, softmax_mask=None):
        """
        Args:
            q: [B, L_q, D_q]
            k: [B, L_k, D_k]
            v: [B, L_v, D_v]
        Return: Same shape to q, but in 'v' space, soft knn
        """

        if attn_mask is None or softmax_mask is None:
            attn_mask = padding_mask_k(q, k)
            softmax_mask = padding_mask_q(q, k)

        # linear projection
        q = self.linear_q(q)
        k = self.linear_k(k)

        scale = q.size(-1)**-0.5

        attention = torch.bmm(q, k.transpose(-2, -1))
        if scale is not None:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = attention.masked_fill(softmax_mask, 0.)

        # attention = self.dropout(attention)
 
        return attention


class MemBank(nn.Module):
    def __init__(self):
        super().__init__()
        bank_path='../mem_bank/msrvtt/app_mot_train.h5' # memory bank is construct by training video
        with h5py.File(bank_path) as f:
            # self.mem_bank=torch.Tensor(f['app_mot_feat'][:][:3000,:,:])
            self.mem_bank=torch.Tensor(f['app_mot_feat'][:])
            # self.mem_bank=torch.zeros(3,5,2)

    def forward(self, bg_mask, vid_feats, vid_idx):
        '''
        input:
            bg_mask: bs,16 float
            vid_feats: bs,16,4096
            vid_idx: bs,

        output: 
            new_vid_feats: bs,16,4096 fix with bg/fg, as indicate by bg_mask
        '''

        self.mem_bank = self.mem_bank.type_as(vid_feats).to(vid_feats.device)
        bs, v_len, hid_dim= vid_feats.size()
        ## for each video, get a vid_feats that the composed by bg (random select from other video)
        weight =  vid_feats.new_ones(bs, self.mem_bank.size(0))
        # exlude video of this sample from sample pool
        if self.training:
            # vid_idx=torch.where(vid_idx<self.mem_bank.size(0), vid_idx, 0)
            weight[torch.arange(bs), vid_idx] = 0
        weight=weight.unsqueeze(-1).expand(-1, -1, v_len) # bs,6513,16
        weight=weight.reshape(bs, -1) 
        sample_idx=torch.multinomial(weight, num_samples=v_len, replacement=True)
        # get bg:bs,16,4096       
        sampled_bg=self.mem_bank.view(-1, hid_dim).unsqueeze(0).expand(bs,-1,-1)[torch.arange(bs).unsqueeze(-1), sample_idx.long()] # bs,16, 4096, all bg
        
        # mix fg/bg, indicated by bg_mask
        vid_feats_new = vid_feats*((1-bg_mask).unsqueeze(-1)) + sampled_bg*(bg_mask.unsqueeze(-1)) 

        return vid_feats_new

if __name__ == "__main__":
    vid_feats=torch.ones(2,5,2).float()
    answers=None
    qns_keys=None
    vid_idx=torch.tensor([1,2],dtype=torch.int64)
    bg_mask=(torch.rand(2,5)>0.5).float()
    model=MemBank()
    out = model(bg_mask, vid_feats, vid_idx)
    print(out.type())