from torch._C import device
from utils.util import EarlyStopping, save_file, set_gpu_devices, pause
import os
from utils.logger import logger
import time
import logging
import argparse
import os.path as osp
import numpy as np    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from networks.hga import HGA
from dataloader.dataset import VidQADataset 
# from torch.utils.tensorboard import SummaryWriter
torch.set_printoptions(linewidth=200)
np.set_printoptions(edgeitems=30, linewidth=30, formatter=dict(float=lambda x: "%.3g" % x))

def predict(model,test_loader, device):
    """
    predict the answer with the trained model
    :param model_file:
    :return:
    """

    model.eval()
    results = {}
    prediction_list = []
    answer_list = []
    with torch.no_grad():
        for iter, inputs in enumerate(test_loader):
            videos, qas, qas_lengths, answers, qns_keys,vid_idx = inputs
            video_inputs = videos.to(device)
            qas_inputs = qas.to(device)
            qas_lengths = qas_lengths.to(device)
            vid_idx = vid_idx.to(device)
            out, out_kl = model(video_inputs, qas_inputs, qas_lengths, vid_idx)
            prediction=out.max(-1)[1] # bs,
            prediction_list.append(prediction)
            answer_list.append(answers)
            # pause()

            for qid, pred, ans in zip(qns_keys, prediction.data.cpu().numpy(), answers.numpy()):
                results[qid] = {'prediction': int(pred), 'answer': int(ans)}
    
    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()

    return results, acc_num*100.0 / len(ref_answers)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path='/storage_fast/ycli/vqa/causal_qa/gird_msrvtt/models/best_model-ONLY_grid_A10_at_11.5_21.6.2.ckpt'
    video_feature_path = '/raid/ycli/vqa/qa_feat/msrvtt'
    sample_list_path = '/raid/ycli/vqa/qa_dataset/msr-vtt/MSRVTT-QA/from_jb'
    
    #data loader
    test_dataset=VidQADataset(video_feature_path, sample_list_path, 'test')
    test_loader = DataLoader(dataset=test_dataset,batch_size=256,shuffle=False,num_workers=0)

    # predicate
    model = HGA(5000).to(device)
    model.load_state_dict(torch.load(best_model_path))
    results, test_acc=predict(model,test_loader, device)
    print(test_acc)