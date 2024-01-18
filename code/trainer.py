# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse
from collections import namedtuple, defaultdict, Counter
import numpy as np
import os
import sys

from copy import copy

from utils import *
import metrics

import pickle

try:
    from tqdm import tqdm
    import ipdb
except:
    pass

def Trans_train(args, dataloader, model, opt):
    model.train()
    kgdataset = dataloader
    kgloader = DataLoader(kgdataset,batch_size=2048, drop_last=True)
    trans_loss = 0.
    for data in tqdm(kgloader, total=len(kgloader), disable=True):
        heads = data[0].to(args.device)
        relations = data[1].to(args.device)
        pos_tails = data[2].to(args.device)
        neg_tails = data[3].to(args.device)
        if args.trans == 'seek':
            kg_batch_loss = model.calc_kg_loss_SEEK(heads, relations, pos_tails, neg_tails)
        if args.trans == 'transr':
            kg_batch_loss = model.calc_kg_loss_transR(heads, relations, pos_tails, neg_tails)
        if args.trans == 'transe':
            kg_batch_loss = model.calc_kg_loss_transE(heads, relations, pos_tails, neg_tails)
        trans_loss += kg_batch_loss / len(kgloader)
        opt.zero_grad()
        kg_batch_loss.backward()
        opt.step()
    return trans_loss.cpu().item()

def train_single_phase(model, train_loader, valid_loader, args, logger, kg=None):
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    stopping_dict = defaultdict(float)
    flag = True

    for e in range(args.epoch):
        # pre-training
        if args.kg and args.train_trans:
            print("[KG_Trans]")
            trans_loss = Trans_train(args, kg, model, optimizer)
            print(f"trans Loss: {trans_loss:.3f}")

        model.train() # train mode
        model.eval_p = None
        model.eval_r = None
        model.eval_p_big = None
        model.eval_r_big = None
        loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
        for b, (uid, o_ck, d_ck, o_rg, d_rg) in tqdm(enumerate(train_loader), total=len(train_loader)):
            uid = uid.to(args.device)
            o_ck = o_ck.to(args.device)
            d_ck = d_ck.to(args.device)
            o_rg = o_rg.to(args.device)
            d_rg = d_rg.to(args.device)
            optimizer.zero_grad()
            loss, crf_output = model(uid, o_ck, d_ck, o_rg, d_rg)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            # torch.cuda.empty_cache()
        scheduler.step()
        logger.log("Epoch %d/%d : Train Loss %.10f" % (e, args.epoch - 1, loss_sum / (b + 1)))
        if e % args.save_step == 0 and not args.best_save:
            save_model(model, e, args.save_path, optimizer, scheduler)
        model.eval()

        if flag:
            poi_rec_10 = 0.
            poi_precision_10 = 0.
            poi_f1_10 = 0.
            poi_ndcg_10 = 0.

            for b, (uid, o_ck, d_ck, o_rg, d_rg) in enumerate(valid_loader):
                uid = uid.to(args.device)
                o_ck = o_ck.to(args.device)
                d_ck = d_ck.to(args.device)
                o_rg = o_rg.to(args.device)
                d_rg = d_rg.to(args.device)
                
                score, alias_poi = model.rank(uid, o_ck, d_ck, o_rg, d_rg)
                _, alias_top = score.topk(k=30,dim=1,largest=False)
                real_top = torch.gather(alias_poi, 1, alias_top)
                real_top = real_top.cpu().detach().numpy()
                label = d_ck.cpu().detach().numpy()

                poi_rec_10 += metrics.p_rec(real_top, label, k=10)
                poi_precision_10 += metrics.p_precision(real_top, label, k=10)
                poi_f1_10 += metrics.p_f1(real_top, label, k=10)
                poi_ndcg_10 += metrics.p_ndcg(real_top, label, k=10)
                # torch.cuda.empty_cache()
            
            # independent poi-level metrics
            poi_rec_10 /= len(valid_loader.dataset)
            poi_precision_10 /= len(valid_loader.dataset)
            poi_f1_10 /= len(valid_loader.dataset)
            poi_ndcg_10 /= len(valid_loader.dataset)

            logger.log("[val] Epoch {}/{} P-Rec@10: {:5.4f} P-Precision@10: {:5.4f} P-F1@10: {:5.4f} P-NDCG@10: {:5.4f}" \
            .format(e, args.epoch - 1, poi_rec_10, poi_precision_10, poi_f1_10, poi_ndcg_10))


        # early stop
        if flag:
            if poi_f1_10 > stopping_dict['best_f1']:
                stopping_dict['best_f1'] = poi_f1_10
                stopping_dict['f1_epoch'] = 0
                stopping_dict['best_epoch'] = e
                if args.best_save: save_model(model, "best", args.save_path, optimizer, scheduler)
            else:
                stopping_dict['f1_epoch'] += 1

            if poi_ndcg_10 > stopping_dict['best_ndcg']:
                stopping_dict['best_ndcg'] = poi_ndcg_10
                stopping_dict['ndcg_epoch'] = 0
            else:
                stopping_dict['ndcg_epoch'] += 1

            # torch.cuda.empty_cache()
            logger.log("early stop: {}|{}".format(stopping_dict['f1_epoch'], stopping_dict["ndcg_epoch"]))

            if stopping_dict['f1_epoch'] >= args.stop_epoch or stopping_dict['ndcg_epoch'] >= args.stop_epoch:
                flag = False
                logger.log("early stopped! best epoch: {}".format(stopping_dict['best_epoch']))

                best_return = stopping_dict['best_epoch']
        if not flag:
            if args.best_save:
                return "best"
            else:
                return best_return

def test(model, model_path, test_loader, args, logger, n_region):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(args.device)
    model.eval()
    model.eval_p = None
    model.eval_r = None
    model.eval_p_big = None
    model.eval_r_big = None

    poi_rec_5 = 0.
    poi_precision_5 = 0.
    poi_ndcg_5 = 0.

    poi_rec_10 = 0.
    poi_precision_10 = 0.
    poi_ndcg_10 = 0.

    poi_rec_15 = 0.
    poi_precision_15 = 0.
    poi_ndcg_15 = 0.

    real_top_list = []
    real_top_score = []

    for b, (uid, o_ck, d_ck, o_rg, d_rg) in tqdm(enumerate(test_loader), total=len(test_loader.dataset) / args.test_batch):
        uid = uid.to(args.device)
        o_ck = o_ck.to(args.device)
        d_ck = d_ck.to(args.device)
        o_rg = o_rg.to(args.device)
        d_rg = d_rg.to(args.device)

        score, alias_poi = model.rank(uid, o_ck, d_ck, o_rg, d_rg)
        alias_score, alias_top = score.topk(k=100,dim=1,largest=False)
        real_top = torch.gather(alias_poi, 1, alias_top)
        real_top = real_top.cpu().detach().numpy() # B x k

        real_top_list.append(real_top)
        real_top_score.append(alias_score.cpu().detach().numpy())

        label = d_ck.cpu().detach().numpy()

        poi_rec_5 += metrics.p_rec(real_top, label, k=5)
        poi_precision_5 += metrics.p_precision(real_top, label, k=5)
        poi_ndcg_5 += metrics.p_ndcg(real_top, label, k=5)

        poi_rec_10 += metrics.p_rec(real_top, label, k=10)
        poi_precision_10 += metrics.p_precision(real_top, label, k=10)
        poi_ndcg_10 += metrics.p_ndcg(real_top, label, k=10)

        poi_rec_15 += metrics.p_rec(real_top, label, k=15)
        poi_precision_15 += metrics.p_precision(real_top, label, k=15)
        poi_ndcg_15 += metrics.p_ndcg(real_top, label, k=15)

        # torch.cuda.empty_cache()

    poi_rec_5 /= len(test_loader.dataset)
    poi_precision_5 /= len(test_loader.dataset)
    poi_ndcg_5 /= len(test_loader.dataset)

    poi_rec_10 /= len(test_loader.dataset)
    poi_precision_10 /= len(test_loader.dataset)
    poi_ndcg_10 /= len(test_loader.dataset)

    poi_rec_15 /= len(test_loader.dataset)
    poi_precision_15 /= len(test_loader.dataset)
    poi_ndcg_15 /= len(test_loader.dataset)

    logger.log("[test-general] HR@5: {:5.4f} Pre@5: {:5.4f} NDCG@5: {:5.4f}, HR@10: {:5.4f} Pre@10: {:5.4f} NDCG@10: {:5.4f}, HR@15: {:5.4f} Pre@15: {:5.4f} NDCG@15: {:5.4f}, " \
    .format(poi_rec_5, poi_precision_5, poi_ndcg_5, poi_rec_10, poi_precision_10, poi_ndcg_10, poi_rec_15, poi_precision_15, poi_ndcg_15))