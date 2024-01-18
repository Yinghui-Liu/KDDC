# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse
from collections import namedtuple, defaultdict
import numpy as np
import os
import sys
from copy import copy
import warnings
warnings.filterwarnings('ignore')

try:
    import ipdb
except:
    pass

from utils import *
from data import TravelDataset, random_split, KGDataset
from core import KDDC

import metrics
from trainer import *

import pickle

def main():
    parser = argparse.ArgumentParser()
    # Foursquare arguments
    parser.add_argument('--ori_data', type=str, default='../Foursquare/home.txt')
    parser.add_argument('--dst_data', type=str, default='../Foursquare/oot.txt')
    parser.add_argument('--trans_data', type=str, default='../Foursquare/travel.txt')
    parser.add_argument('--save_path', type=str, default='./model_save')
    parser.add_argument("--best_save", action="store_false")
    parser.add_argument("--pp_graph_path", type=str, default="../Foursquare/pp_adj.npz")
    parser.add_argument("--kg_path", type=str, default="../Foursquare/kg.txt")
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--data_split_path', type=str, default='../Foursquare/data_split.pkl')
    # training configurations
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--train_batch', type=int, default=64)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--test_batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument('--margin', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--lr_dc', type=float, default=0.2)
    parser.add_argument('--lr_dc_step', type=int, default=4)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=9911)
    parser.add_argument('--log_path', type=str, default='./')
    parser.add_argument('--log', action="store_false")
    parser.add_argument('--name', type=str, default="default")
    parser.add_argument('--model', type=str, default="base")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument("--stop_epoch", type=int, default=8) # early stopping
    parser.add_argument("--fine_stop", type=int, default=12)
    parser.add_argument("--k", type=int, default=50) # k in kNN

    # kg arguments
    parser.add_argument("--segments", type=int, default=8) # yelp 4
    parser.add_argument("--kg", action="store_true")
    parser.add_argument("--entity_num_per_poi", type=int, default=2) # yelp 13
    parser.add_argument("--train_trans", action="store_true")
    parser.add_argument('--trans', type=str, default="seek")
    parser.add_argument("--contrast", action="store_true")
    parser.add_argument("--kgcn", type=str, default="RGAT")
    parser.add_argument("--kg_p_drop", type=float, default=0.5)
    parser.add_argument("--ui_p_drop", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.2)

    # crf arguments
    parser.add_argument("--crf", action="store_true")
    parser.add_argument("--crf_layer", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)

    # infer arguments
    parser.add_argument("--dc", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--dm", action="store_true")
    parser.add_argument("--pop_path", type=str, default='../Foursquare/poi_pop.pkl')
    parser.add_argument("--pop_coeff", type=float, default=1.0)

    args = parser.parse_args()
    set_seeds(args.seed)
    args.save_path = os.path.join(args.save_path, args.name)
    path_exist(args.save_path)

    logger = Logger(args.log_path, args.name, args.seed, args.log)
    logger.log(str(args))
    logger.log("Experiment name: %s" % args.name)

    data = TravelDataset(args, args.ori_data, args.dst_data, args.trans_data)
    if args.kg:
        kg_data = KGDataset(args)
    else:
        kg_data = None
    train_data, valid_data, test_data = random_split(data, split_path=args.data_split_path)

    train_loader = DataLoader(train_data, args.train_batch, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, args.test_batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, args.test_batch, shuffle=False, collate_fn=collate_fn)

    n_region = len(data.region_idx)
    pp_adj = data.pp_adj

    model = KDDC(args, len(data.poi_idx) + 1, len(data.region_idx), data.region_poi, pp_adj, kg_data)

    model = model.to(args.device)

    if args.mode == 'train':
        best = train_single_phase(model, train_loader, valid_loader, args, logger, kg_data)

        test(model, os.path.join(args.save_path, "model_{}.xhr".format(best)), test_loader, args, logger, n_region)
        print("################## current exp done ##################")
    elif args.mode == 'test':
        test(model, os.path.join(args.save_path, "model_best.xhr"), test_loader, args, logger,
             n_region)

    logger.close_log()
    
if __name__ == "__main__":
    main()