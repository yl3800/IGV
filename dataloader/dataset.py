import torch
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('..')
from utils.util import load_file, pause
import os.path as osp
import numpy as np
import nltk
import pandas as pd
import json
import string
import h5py
import pickle as pkl


class VidQADataset(Dataset):
    """load the dataset in dataloader
    app+mot_feat:
                [ids]:vid     (6513,)
                [feat]:frame_feature (6513,16,16,2048)
                [feat]:mot_feature (6513,16,2048)
    qas_bert_feat:
                [feat]:feature (158581, 20, 768)
    """

    def __init__(self, video_feature_path, sample_list_path, mode, ans_vocab_num):
        self.mode = mode
        self.video_feature_path = video_feature_path
        sample_list_file = osp.join(sample_list_path, '{}.csv'.format(mode))
        self.sample_list = load_file(sample_list_file)
        print('dataset len' , len(self.sample_list))
        ans=load_file('../qa_dataset/msr-vtt/MSRVTT-QA/stats/ans_word_{}.json'.format(ans_vocab_num))
        self.ans2idx={a:idx for idx,a in enumerate(ans)}
        self.bert_file = osp.join(video_feature_path, 'qas_bert/bert_ft_{}.h5'.format(mode))
        frame_feat_file = osp.join(video_feature_path, 'frame_feat/app_feat_{}.h5'.format(mode))
        mot_feat_file = osp.join(video_feature_path, 'mot_feat/mot_feat_{}.h5'.format(mode))

        print('Load {}...'.format(frame_feat_file))
        print('Load {}...'.format(mot_feat_file))
        self.frame_feats = {}
        self.mot_feats = {}
        self.vid2idx={}
    
        # get frame feat
        with h5py.File(frame_feat_file, 'r') as fp:
            vids = fp['ids']
            feats = fp['resnet_features'] 
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                self.frame_feats[str(vid)] = feat[:, 8, :]  # (16, 2048) get mid frame in each seg
                self.vid2idx[str(vid)] = id

        # get mot feat
        with h5py.File(mot_feat_file, 'r') as fp:
                vids = fp['ids']
                feats = fp['resnext_features'] 
                for id, (vid, feat) in enumerate(zip(vids, feats)):        
                    self.mot_feats[str(vid)] = feat  # (16, 2048)


    def __len__(self):
        return len(self.sample_list)


    def get_video_feature(self, video_name):
        """
        :param video_name:
        :return:
        """
        app_feat = self.frame_feats[video_name]
        mot_feat = self.mot_feats[video_name]
        video_feature = np.concatenate((app_feat, mot_feat), axis=1) #(16, 4096)

        return torch.from_numpy(video_feature).type(torch.float32)


    def __getitem__(self, idx):
        # print(idx)
        cur_sample = self.sample_list.iloc[idx]
        video_name, qns, ans_str, qns_id, category_id = str(cur_sample['video']), str(cur_sample['question']),\
                                    str(cur_sample['answer']), str(cur_sample['qid']), str(cur_sample['type'])

        if ans_str in self.ans2idx.keys():
            ans=self.ans2idx[ans_str]
        else:
            if self.mode =='train':
                ans=len(self.ans2idx)
            else:
                ans=-1

        # index to embedding
        with h5py.File(self.bert_file, 'r') as fp:
            temp_feat = fp['feat'][idx]
            qns = torch.from_numpy(temp_feat).type(torch.float32) # (20,768)
            q_lengths=((qns.sum(-1))!=0.0).sum(-1)

        video_feature = self.get_video_feature(video_name)

        # get video idx in vid_feat.h5
        vid_idx=self.vid2idx[video_name]
        return video_feature, qns, q_lengths, ans , qns_id, vid_idx

if __name__ == "__main__":

    video_feature_path = '../vqa/qa_feat/msrvtt'
    sample_list_path = '../vqa/qa_dataset/msr-vtt/MSRVTT-QA'
    train_dataset=VidQADataset(video_feature_path, sample_list_path, 'train',1000)
    # val_dataset=VidQADataset(video_feature_path, sample_list_path, 'val',1000)

    train_loader = DataLoader(dataset=train_dataset,batch_size=8,shuffle=False,num_workers=0)
    # val_loader = DataLoader(dataset=val_dataset,batch_size=args.bs,shuffle=False,num_workers=8)
    # test_loader = DataLoader(dataset=test_dataset,batch_size=args.bs,shuffle=False,num_workers=8)
    for sample in train_loader:
        video_feature, qns, q_lengths, ans , qns_id, vid_idx=sample
        print(q_lengths.shape)
        print(vid_idx.shape)
        print(vid_idx)
        pause()