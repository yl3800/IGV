import torch
import torch.nn as nn
import torch.nn.functional as F


def padding_mask_k(seq_q, seq_k):
    """ To mask invaild k(all dim are 0), and assign -inf in softmax, seq_k of shape (batch, k_len, k_feat) and seq_q (batch, q_len, q_feat). q and k are padded with 0. pad_mask is (batch, q_len, k_len).
    In batch 0:
    [[x x x 0]     [[0 0 0 1]
     [x x x 0]->    [0 0 0 1]
     [x x x 0]]     [0 0 0 1]] uint8
    """
    fake_q = torch.ones_like(seq_q)
    pad_mask = torch.bmm(fake_q, seq_k.transpose(1, 2))
    pad_mask = pad_mask.eq(0)
    # pad_mask = pad_mask.lt(1e-3)
    return pad_mask


def padding_mask_q(seq_q, seq_k):
    """ To mask invalid q(all dim are 0), seq_k of shape (batch, k_len, k_feat) and seq_q (batch, q_len, q_feat). q and k are padded with 0. pad_mask is (batch, q_len, k_len).
    In batch 0:
    [[x x x x]      [[0 0 0 0]
     [x x x x]  ->   [0 0 0 0]
     [0 0 0 0]]      [1 1 1 1]] uint8
    """
    fake_k = torch.ones_like(seq_k)
    pad_mask = torch.bmm(seq_q, fake_k.transpose(1, 2))
    pad_mask = pad_mask.eq(0)
    # pad_mask = pad_mask.lt(1e-3)
    return pad_mask


@classmethod
def get_u_tile(cls, s, s2): 
    """
    attended vectors of s2 for each word in s1,
    signify which words in s2 are most relevant to words in s1
    """
    a_weight = F.softmax(s, dim=2)  # [B, l1, l2]
    
    a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)
    # [B, l1, l2] * [B, l2, D] -> [B, l1, D]
    u_tile = torch.bmm(a_weight, s2)
    return u_tile, a_weight


def forward(self, s1, l1, s2, l2):
    s = self.similarity(s1, l1, s2, l2)
    u_tile, a_weight = self.get_u_tile(s, s2)
    
    return u_tile, a_weight
            

class AttentionScore(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)

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

        if attn_mask is None: 
            attn_mask = padding_mask_k(q, k)
        if softmax_mask is None:
            softmax_mask = padding_mask_q(q, k)

        # linear projection
        q = self.linear_q(q)
        k = self.linear_k(k)

        scale = q.size(-1)**-0.5

        attention = torch.bmm(q, k.transpose(-2, -1))
        
        if scale is not None:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask.bool(), -float("inf"))
        attention = self.softmax(attention)
        attention = attention.masked_fill(softmax_mask, 0.)
 
        return attention


class Grounding_Indicator(nn.Module):

    def __init__(self, hidden_size, tau=1, is_hard=False, dropout_p=0):
        super().__init__()
        self.tau = tau
        self.is_hard = is_hard
        self.fg_att = AttentionScore(hidden_size, dropout_p)
        self.bg_att = AttentionScore(hidden_size, dropout_p)

    def forward(self, q_global, v_local, attn_mask=None):
        """
        q_global: bs,d
        v_local: bs, L, d
        """
        fg_score = self.fg_att(q_global.unsqueeze(1), v_local, attn_mask = attn_mask) #[bs, 1, 16]
        bg_score = self.bg_att(q_global.unsqueeze(1), v_local, attn_mask = attn_mask)
        score=torch.cat((fg_score,bg_score),1)#[bs, 2, 16]
        score=F.gumbel_softmax(score, tau=self.tau, hard=self.is_hard, dim=1) #[bs, 2, 16]

        fg_mask=score[:,0,:]#[bs, 16]
        bg_mask=score[:,1,:]#[bs, 16]
        
        # # if sampled all fg/bg, then manully set :-1 to be fg.
        # fg_len = fg_mask.sum(-1)
        # bg_len = bg_mask.sum(-1)
        # invalid = (fg_len==0) + (bg_len==0)
        
        # if invalid.any():
        #     fg_mask[invalid, :-1] = 1
        #     fg_mask[invalid,  -1] = 0

        #     bg_mask[invalid, :-1] = 0
        #     bg_mask[invalid,  -1] = 1            
            
        return fg_mask, bg_mask
