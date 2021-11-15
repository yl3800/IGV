import numpy as np
import torch
import torch.nn as nn
# import random as rd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sys
sys.path.append('../')
from utils.util import pause
from networks.q_v_transformer import CoAttention
from networks.gcn import AdjLearner, GCN
from networks.mem_bank import AttentionScore, MemBank
from networks.util import length_to_mask
from block import fusions #pytorch >= 1.1.0



class EncoderQns(nn.Module):
    def __init__(self, dim_embed, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):

        super(EncoderQns, self).__init__()
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell
        self.q_input_ln = nn.LayerNorm((dim_hidden*2 if bidirectional else dim_hidden), elementwise_affine=False)
        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        # self.embedding = nn.Linear(768, dim_embed)
        self.embedding = nn.Sequential(nn.Linear(768, dim_embed),
                                     nn.ReLU(),
                                     nn.Dropout(input_dropout_p))

        self.rnn = self.rnn_cell(dim_embed, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)
        self._init_weight()
    

    def _init_weight(self):
        nn.init.xavier_normal_(self.embedding[0].weight) 


    def forward(self, qns, qns_lengths):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """
        qns_embed = self.embedding(qns)
        qns_embed = self.input_dropout(qns_embed)
        packed = pack_padded_sequence(qns_embed, qns_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # hidden = torch.squeeze(hidden)
        hidden = hidden.reshape(hidden.size()[1], -1)
        output = self.q_input_ln(output) # bs,q_len,hidden_dim

        return output, hidden



class EncoderVidHGA(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderVidHGA, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell
        self.v_input_ln = nn.LayerNorm((dim_hidden*2 if bidirectional else dim_hidden), elementwise_affine=False)

        self.vid2hid = nn.Sequential(nn.Linear(self.dim_vid, dim_hidden),
                                     nn.ReLU(),
                                     nn.Dropout(input_dropout_p))


        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self._init_weight()


    def _init_weight(self):
        nn.init.xavier_normal_(self.vid2hid[0].weight) 


    def forward(self, vid_feats, fg_mask =None):
        """
        vid_feats: (bs, 16, 4096)
        fg_mask: (bs, 16,) bool mask
        """
        
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats_trans = self.vid2hid(vid_feats.view(-1, self.dim_vid))
        vid_feats = vid_feats_trans.view(batch_size, seq_len, -1)

        if fg_mask is not None:
            fg_mask_ = fg_mask.clone()
            # stack left 
            temp=vid_feats.new_zeros(vid_feats.size())
            for i, (vid_feat_i, fg_mask_i)  in enumerate(zip(vid_feats, fg_mask)):
                fg_len_i=fg_mask_i.sum(-1)
                # if no fg frame, manualy set allframe to be fg
                if fg_len_i == 0:
                    fg_len_i = fg_mask_i.size(0)
                    fg_mask_i = fg_mask_i.new_ones(fg_mask_i.size()) 
                    fg_mask_[i,:]=fg_mask_i
                temp[i, :fg_len_i, :] = vid_feat_i[fg_mask_i, :] # assemble value to left, [1,0,2,0,3]-->[1,2,3,0,0]
            vid_feats = pack_padded_sequence(temp, fg_mask_.cpu().sum(-1), batch_first=True, enforce_sorted=False)

        # self.rnn.flatten_parameters() # for parallel
        foutput, fhidden = self.rnn(vid_feats)

        if fg_mask is not None:
            foutput, _ = pad_packed_sequence(foutput, batch_first=True)

        # fhidden = torch.squeeze(fhidden)
        fhidden = fhidden.reshape(fhidden.size()[1], -1)
        foutput = self.v_input_ln(foutput) # bs,16,hidden_dim

        if fg_mask is not None:
            return foutput, fhidden, fg_mask_.to(foutput.dtype)
        else:
            return foutput, fhidden
        # return foutput, fhidden


class HGA(nn.Module):
    def __init__(self, vocab_num, hidden_dim = 512,  word_dim = 512, input_dropout_p=0.5, tau=1, num_layers=1):
        """
        Reasoning with Heterogeneous Graph Alignment for Video Question Answering (AAAI2020)
        """
        super(HGA, self).__init__()
        vid_dim = 2048 + 2048
        self.tau=tau
        self.vid_encoder = EncoderVidHGA(vid_dim, hidden_dim, input_dropout_p=input_dropout_p,bidirectional=True, rnn_cell='gru')
        self.qns_encoder = EncoderQns(word_dim, hidden_dim, n_layers=1,rnn_dropout_p=0, input_dropout_p=input_dropout_p, bidirectional=True, rnn_cell='gru')

        hidden_size = self.vid_encoder.dim_hidden*2
        input_dropout_p = self.vid_encoder.input_dropout_p

        self.fg_att = AttentionScore(hidden_size)
        self.bg_att = AttentionScore(hidden_size)

        self.mem_swap = MemBank()
        
        self.adj_learner = AdjLearner(
            hidden_size, hidden_size, dropout=input_dropout_p)

        self.gcn = GCN(
            hidden_size,
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=input_dropout_p)

        self.atten_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=-2))

        self.global_fusion = fusions.Block([hidden_size, hidden_size], hidden_size, dropout_input=input_dropout_p)
        self.fusion = fusions.Block([hidden_size, hidden_size], hidden_size)

        self.decoder=nn.Linear(hidden_size, vocab_num+1) # ans_num+<unk>

    def forward(self, vid_feats, qns, qns_lengths, vid_idx):
        """

        :param vid_feats:[bs, 16, 4096]
        :param qns: [bs, 20, 768]
        :param qns_lengths:[bs,]
        :return:
        """

        ## encode q,v
        q_local, q_global = self.qns_encoder(qns, qns_lengths)
        v_local, _ = self.vid_encoder(vid_feats)
        
        ## fg/bg att
        fg_mask, bg_mask =self.frame_att(q_global, v_local) #[bs, 16]

        ## bg branch
        v_local_b, v_global_b, bg_mask = self.vid_encoder(vid_feats, bg_mask.bool())
        out_b = self.fusion_predict(q_global, v_global_b, q_local, v_local_b, qns_lengths, v_len=bg_mask.sum(-1))


        ## fg branch
        # v_local_f, v_global_f = self.vid_encoder(vid_feats, fg_mask.bool())
        v_local_f, v_global_f, fg_mask = self.vid_encoder(vid_feats, fg_mask.bool())
        bg_mask=1-fg_mask
        out_f = self.fusion_predict(q_global, v_global_f, q_local, v_local_f, qns_lengths, v_len=fg_mask.sum(-1))

        ## mem branch
        vid_feats_m = self.mem_swap(bg_mask, vid_feats, vid_idx)
        v_local_m, v_global_m = self.vid_encoder(vid_feats_m)
        # print(v_local_m.shape, v_global_m.shape)
        out_m = self.fusion_predict(q_global, v_global_m, q_local, v_local_m, qns_lengths)


        return out_f, out_m, out_b



    def frame_att(self, q_global, v_local):

        fg_score = self.fg_att(q_global.unsqueeze(1), v_local) #[bs, 1, 16]
        # bg_score = 1-fg_score
        bg_score = self.bg_att(q_global.unsqueeze(1), v_local)

        # gumbel_softmax, try tau 1-10
        score=torch.cat((fg_score,bg_score),1)#[bs, 2, 16]
        score=F.gumbel_softmax(score, tau=self.tau, hard=True, dim=1) #[bs, 2, 16]

        fg_mask=score[:,0,:]#[bs, 16]
        bg_mask=score[:,1,:]#[bs, 16]

        return fg_mask, bg_mask



    def fusion_predict(self, q_global, v_global, q_local, v_local, q_len, v_len=None, **kwargs):
        '''kwargs: coatt, gcn
            v_len: [bs,] number of fg frame in each sample'''

        ### integrate q,v: GCN/cat
        # ## gcn
        adj = self.adj_learner(q_local, v_local)
        q_v_inputs = torch.cat((q_local, v_local), dim=1)
        q_v_local=self.gcn(q_v_inputs, adj)

        ## attention pool with mask (if applicable)
        local_attn = self.atten_pool(q_v_local) # [bs, 23, 1]
        q_mask = length_to_mask(q_len, q_local.size(1))
        if v_len is not None: # both qv need mask
            v_mask = length_to_mask(v_len, v_local.size(1))
            pool_mask = torch.cat((q_mask, v_mask), dim=-1).unsqueeze(-1) # bs,20+16,1
        else: # only q need mask
            pool_mask=torch.cat((q_mask, v_local.new_ones(v_local.size()[:2])), dim=-1).unsqueeze(-1) # bs,len_q+16,1
        local_out = torch.sum(q_v_local * pool_mask * local_attn, dim=1) # bs, hidden

        ## fusion
        global_out = self.global_fusion((q_global, v_global))
        out = self.fusion((global_out, local_out)).squeeze() # bs x hidden

        ## decoder 
        out=self.decoder(out)

        return out


if __name__ == "__main__":
    videos=torch.rand(3,16,4096)
    qas=torch.cat((torch.rand(3,7,768),torch.zeros(3,13,768)), dim=1)
    qas_lengths=torch.tensor([7,10,10],dtype=torch.int64) #torch.randint(5, 20, (32,))
    answers=None
    qns_keys=None
    vid_idx=torch.tensor([7,7,7],dtype=torch.int64)

    model=HGA(5000)
    out_f, out_m, out_b = model(videos, qas, qas_lengths, vid_idx)
    print(out_f.shape, out_m.shape, out_b.shape)
    