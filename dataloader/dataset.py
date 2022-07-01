import sys
sys.path.append('..')
import torch
import h5py
import os.path as osp
import numpy as np
from utils.util import load_file, pause
from torch.utils.data import Dataset, DataLoader


class VideoQADataset(Dataset):
    def __init__(self, sample_list_path, feat_path, split, args):
        super(VideoQADataset, self).__init__()
        print(feat_path)
        app_feat_path = osp.join(feat_path,'{}/frame_feat/{}_appearance_{}_feat_{}.h5'.format(args.dataset,args.dataset, args.app_feat, split))
        mot_feat_path = osp.join(feat_path,'{}/mot_feat/{}_motion_{}_feat_{}.h5'.format(args.dataset,args.dataset, args.mot_feat, split))
        self.split = split
        
        self.sample_list = load_file(osp.join(sample_list_path, '{}/{}/stats/{}.csv'.format(args.dataset, args.dataset, split)))
        print('dataset len' , len(self.sample_list))

        # self.bert_file = osp.join(video_feature_path, 'qas_bert/bert_ft_{}.h5'.format(mode))
        self.bert_file = osp.join(feat_path, '{}/roberta/roberta_21_meanpool_{}.h5'.format(args.dataset, split))
        ans=load_file(osp.join(sample_list_path,'{}/{}/stats/ans_word_{}.json'.format(args.dataset,args.dataset,args.ans_num)))
        self.ans2idx={a:idx for idx,a in enumerate(ans)}
        self.sample_list['answer_id'] = self.sample_list['answer'].map(self.ans2idx)
        if split=='train':
            self.sample_list['answer_id'].fillna(len(self.ans2idx), inplace=True)
            self.ans_group=self.sample_list.groupby("answer_id")
        else:
            self.sample_list['answer_id'].fillna(-1, inplace=True)

        print('Load {}...'.format(app_feat_path))
        print('Load {}...'.format(mot_feat_path))
        self.frame_feats = {}
        self.mot_feats = {}
        self.vid2idx={}
    
        # get frame feat
        with h5py.File(app_feat_path, 'r') as fp:
            vids = fp['ids'][:]
            self.app = fp[str(args.app_feat)+'_features'][:]
            for id, (vid, feat) in enumerate(zip(vids, self.app)):
                self.frame_feats[str(vid)] = feat                                                                                   # (16, 2048) get mid frame 
                self.vid2idx[str(vid)] = id 

        # get mot feat
        with h5py.File(mot_feat_path, 'r') as fp:
                vids = fp['ids'][:]
                self.mot = fp[str(args.mot_feat)+'_features'][:]
                for id, (vid, feat) in enumerate(zip(vids, self.mot)):      
                    self.mot_feats[str(vid)] = feat                                                                                 # (16, 2048)

        # bert feat
        with h5py.File(self.bert_file, 'r') as fp:
            self.bert_feat = fp['feat'][:]
        

    def get_video_feature(self, video_id):
        """
        :param video_id:
        :return:
        """
        app_feat = self.frame_feats[str(video_id)]
        mot_feat = self.mot_feats[str(video_id)]
        video_feature = np.concatenate((app_feat, mot_feat), axis=-1)                                                               #(16, 4096)
        return torch.from_numpy(video_feature).type(torch.float32)
        

    def __getitem__(self, idx):
        cur_sample = self.sample_list.iloc[idx]
        video_id, ans_id, qns_id = str(cur_sample['video_id']), int(cur_sample['answer_id']), int(cur_sample['id'])

        # qst feat
        qns = torch.from_numpy(self.bert_feat[idx]).type(torch.float32)                                                             # (20,768)
        q_lengths=((qns.sum(-1))!=0.0).sum(-1)

        # vid feat
        vis_feat = self.get_video_feature(video_id)
        vid_idx=self.vid2idx[str(video_id)]

        return vis_feat, qns, q_lengths, ans_id, qns_id, vid_idx


    def __len__(self):
        return len(self.sample_list)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MSPAN logger")
    parser.add_argument('-dataset', default='msvd-qa',choices=['msrvtt-qa', 'msvd-qa'], type=str)
    parser.add_argument('-app_feat', default='resnet', choices=['resnet', 'res152'], type=str)
    parser.add_argument('-mot_feat', default='resnext', choices=['resnext', '3dres152'], type=str)
    parser.add_argument('-ans_num', default=1852, type=int)
    args = parser.parse_args()
    sample_list_path = '/storage_fast/ycli/vqa/qa_dataset'
    feat_path= '/storage_fast/ycli/vqa/qa_feat'
    qst_feat_path = '/storage_fast/ycli/vqa/qa_feat'

    train_data = VideoQADataset(sample_list_path, feat_path, 'train', args)

    train_loader = DataLoader(dataset=train_data,batch_size=8,shuffle=False,num_workers=0)
    for sample in train_loader:
        vis_feat, qns, q_lengths, ans, qns_id,vid_idx=sample
        print(vis_feat.shape)
        print(qns.shape)
        print(q_lengths.shape)
        print(vid_idx.shape)
        pause()