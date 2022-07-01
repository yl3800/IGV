import torch 

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    lens = [3, 5, 4]
    mask = [[1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0]]
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def intervene(mem_bank, bg_mask, vid_feats, vid_idx):
    '''
    input:
        bg_mask: bs,16 float
        vid_feats: bs,16,4096
        vid_idx: bs,

    output: 
        new_vid_feats: bs,16,4096 fix with bg/fg, as indicate by bg_mask
    '''

    mem_bank = mem_bank.type_as(vid_feats).to(vid_feats.device)
    bs, v_len, hid_dim= vid_feats.size()
    ## for each video, get a vid_feats that the composed by bg (random select from other video)
    weight =  vid_feats.new_ones(bs, mem_bank.size(0))
    # exlude video of this sample from sample pool
    weight[torch.arange(bs), vid_idx] = 0
    weight = weight.unsqueeze(-1).expand(-1, -1, v_len) # bs,6513,16
    weight = weight.reshape(bs, -1) 
    sample_idx = torch.multinomial(weight, num_samples=v_len, replacement=True)
    # get bg:bs,16,4096       
    sampled_bg = mem_bank.view(-1, hid_dim).unsqueeze(0).expand(bs,-1,-1)[torch.arange(bs).unsqueeze(-1), sample_idx.long()] # bs,16, 4096, all bg
    
    # mix fg/bg, indicated by bg_mask
    vid_feats_new = vid_feats*((1-bg_mask).unsqueeze(-1)) + sampled_bg*(bg_mask.unsqueeze(-1)) 

    return vid_feats_new