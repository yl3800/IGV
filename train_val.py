from utils.util import EarlyStopping, save_file, set_gpu_devices, pause
import os
from utils.logger import logger
import time
import logging
import argparse
import os.path as osp
import numpy as np


parser = argparse.ArgumentParser(description="GCN train parameter")
parser.add_argument("-v", type=str, required=True, help="version")
parser.add_argument("-bs", type=int, action="store", help="BATCH_SIZE", default=256)
parser.add_argument("-lr", type=float, action="store", help="learning rate", default=1e-4)
parser.add_argument("-epoch", type=int, action="store", help="epoch for train", default=40)
parser.add_argument("-nfs", action="store_true", help="use local ssd")
parser.add_argument("-gpu", type=int, help="set gpu id", default=0)    
parser.add_argument("-ans_num", type=int, help="ans vocab num", default=5000)  
parser.add_argument("-es", action="store_true", help="early_stopping")
parser.add_argument("-hd", type=int, help="hidden dim of vq encoder", default=512) 
parser.add_argument("-wd", type=int, help="word dim of q encoder", default=512)   
parser.add_argument("-drop", type=float, help="dropout rate", default=0.56) 
parser.add_argument("-tau", type=float, help="gumbel tamper", default=0.9)
parser.add_argument("-ln", type=int, help="number of layers", default=1) 
parser.add_argument("-pa", type=int, help="patience of ReduceonPleatu", default=5)  
parser.add_argument("-a", type=float, help="ratio on L2", default=1) 
parser.add_argument("-b", type=float, help="ratio on L3", default=1) 

args = parser.parse_args()
set_gpu_devices(args.gpu)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from networks.embed_loss import MultipleChoiceLoss
from networks.hga import HGA
from dataloader.dataset import VidQADataset 

seed = 999

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# from torch.utils.tensorboard import SummaryWriter
torch.set_printoptions(linewidth=200)
np.set_printoptions(edgeitems=30, linewidth=30, formatter=dict(float=lambda x: "%.3g" % x))
# torch.autograd.set_detect_anomaly(True)


def train(model, epoch, epoch_num, optimizer, train_loader, ce, kl_mb, kl_b, device):
    model.train()
    total_step = len(train_loader)
    epoch_loss = 0.0
    epoch_ce_loss = 0.0
    epoch_kl_loss = 0.0
    epoch_klb_loss = 0.0
    prediction_list = []
    answer_list = []
    for iter, inputs in enumerate(train_loader):
        videos, qas, qas_lengths, answers, _, vid_idx = inputs
        video_inputs = videos.to(device)
        qas_inputs = qas.to(device)
        ans_targets = answers.to(device)
        qas_lengths = qas_lengths.to(device)
        vid_idx = vid_idx.to(device)
        out_f, out_m,out_b = model(video_inputs, qas_inputs, qas_lengths, vid_idx)
        model.zero_grad()
        ce_loss = ce(out_f, ans_targets)
        kl_loss = kl_mb(F.log_softmax(out_m, dim=1), F.softmax(out_f, dim=1))
        klb_loss = kl_b(F.log_softmax(out_b, dim=1), out_b.new_ones(out_b.size())/5001)
 
        # kl_loss = sum([kl(F.log_softmax(out_m[:,:,i], dim=1), F.softmax(out_f, dim=1)) for i in range(3)])
        # kl_loss=kl_loss/3

        # var_loss = torch.var(out_m, dim=-1).mean(-1).mean(-1)

        # kl_loss = kl(F.log_softmax(out_m, dim=1), ans_targets)
        # print(kl_loss, ce_loss)
        # pause()
        loss = ce_loss + args.a*kl_loss + args.b*klb_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_ce_loss += ce_loss.item()
        epoch_kl_loss += args.a*kl_loss.item()
        epoch_klb_loss += args.b*klb_loss.item()
        prediction=out_f.max(-1)[1] # bs,
        prediction_list.append(prediction)
        answer_list.append(answers)

        # # print(out_m)
        # if iter%200==0:
        #     print(ce_loss, kl_loss, 'iter{}/{}'.format(iter,total_step))
        # # pause()

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()
    # print(epoch,'train ansewer')
    # print(predict_answers[:1000])
    # print('*'*50)
    # print(ref_answers[:1000])
    
    return epoch_loss / total_step, epoch_ce_loss/ total_step, epoch_kl_loss/ total_step,epoch_klb_loss/total_step, acc_num*100.0 / len(ref_answers)
    

def eval(model, epoch, epoch_num, val_loader, device):
    model.eval()
    prediction_list = []
    answer_list = []
    with torch.no_grad():
        for iter, inputs in enumerate(val_loader):
            videos, qas, qas_lengths, answers, _, vid_idx = inputs
            video_inputs = videos.to(device)
            qas_inputs = qas.to(device)
            qas_lengths = qas_lengths.to(device)
            vid_idx = vid_idx.to(device)
            out, _, _ = model(video_inputs, qas_inputs, qas_lengths,vid_idx)
            prediction=out.max(-1)[1] # bs,            
            prediction_list.append(prediction)
            answer_list.append(answers)

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()
    # print(epoch,'val ansewer')
    # print(predict_answers)
    # print('*'*50)
    # print(ref_answers)
    return acc_num*100.0 / len(ref_answers)

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
            out, _, _ = model(video_inputs, qas_inputs, qas_lengths, vid_idx)
            prediction=out.max(-1)[1] # bs,
            prediction_list.append(prediction)
            answer_list.append(answers)

            for qid, pred, ans in zip(qns_keys, prediction.data.cpu().numpy(), answers.numpy()):
                results[qid] = {'prediction': int(pred), 'answer': int(ans)}
    
    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()

    return results, acc_num*100.0 / len(ref_answers)


if __name__ == "__main__":

    # writer = SummaryWriter('./log/tensorboard')
    logger, sign =logger(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data set&Loader
    if not args.nfs:
        video_feature_path = '../vqa/qa_feat/msrvtt' # .h5 file
        sample_list_path = '../vqa/qa_dataset/msr-vtt/MSRVTT-QA' # .csv file
    else:
        video_feature_path = '../qa_feat/msrvtt' # .h5 file
        sample_list_path = '../vqa/qa_dataset/msr-vtt/MSRVTT-QA' # .csv file

    train_dataset=VidQADataset(video_feature_path, sample_list_path, 'train',args.ans_num)
    val_dataset=VidQADataset(video_feature_path, sample_list_path, 'val',args.ans_num)
    test_dataset=VidQADataset(video_feature_path, sample_list_path, 'test',args.ans_num)

    train_loader = DataLoader(dataset=train_dataset,batch_size=args.bs,shuffle=True,num_workers=8)
    val_loader = DataLoader(dataset=val_dataset,batch_size=args.bs,shuffle=False,num_workers=8)
    test_loader = DataLoader(dataset=test_dataset,batch_size=args.bs,shuffle=False,num_workers=8)

    # hyper setting
    lr_rate = args.lr
    epoch_num = args.epoch
    model = HGA(args.ans_num, args.hd,  args.wd, args.drop, args.tau, args.ln)
    optimizer = torch.optim.Adam(params = [{'params':model.parameters()}], lr=lr_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=args.pa, verbose=True)
    model.to(device)
    ce = nn.CrossEntropyLoss().to(device)
    kl_mb = nn.KLDivLoss(reduction='batchmean').to(device)
    kl_b = nn.KLDivLoss(reduction='batchmean').to(device)

    if args.es:
        early_stopping = EarlyStopping(patience=7, min_delta=0)

    # train & val
    best_eval_score = 0.0
    best_epoch=1
    for epoch in range(1, epoch_num+1):
        train_loss, ce_loss, kl_loss, klb_loss,train_acc = train(model, epoch, epoch_num, optimizer, train_loader, ce, kl_mb, kl_b, device)
        # print(ce_loss)
        eval_score = eval(model, epoch, epoch_num, val_loader, device)
        scheduler.step(eval_score)
        if eval_score > best_eval_score :
            best_eval_score = eval_score
            best_epoch = epoch 
            best_model_path='./models/best_model-{}.ckpt'.format(sign)
            torch.save(model.state_dict(), best_model_path)

        # after 20epoch print both val/test acc
        if epoch<=20:
            logger.debug("==>Epoch:[{}/{}][LR{}][Train Loss: {:.4f} CE Loss: {:.4f} KL Loss: {:.4f} KLB Loss: {:.4f} Train acc: {:.2f} Val acc: {:.2f}".
            format(epoch, epoch_num, optimizer.param_groups[0]['lr'], train_loss, ce_loss, kl_loss, klb_loss, train_acc, eval_score))
        else:
            test_score = eval(model, epoch, epoch_num, test_loader, device)
            logger.debug("==>Epoch:[{}/{}][LR{}][Train Loss: {:.4f} CE Loss: {:.4f} KL Loss: {:.4f} KLB Loss: {:.4f} Train acc: {:.2f} Val acc: {:.2f} Test acc: {:.2f}".
            format(epoch, epoch_num, optimizer.param_groups[0]['lr'], train_loss, ce_loss, kl_loss, klb_loss, train_acc, eval_score, test_score))

        # check if need to early stop
        if args.es:
            early_stopping(eval_score)
            if early_stopping.early_stop:
                break

    logger.debug("Epoch {} Best Val acc{:.2f}".format(best_epoch, best_eval_score))

    # predict with best model
    model.load_state_dict(torch.load(best_model_path))
    results, test_acc=predict(model,test_loader, device)
    logger.debug("Test acc{:.2f} on {} epoch".format(test_acc, best_epoch))

    result_path= './prediction/{}-{}-{:.2f}.json'.format(sign, best_epoch, best_eval_score)
    save_file(results, result_path)
