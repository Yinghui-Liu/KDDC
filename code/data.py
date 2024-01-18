from random import shuffle, choice
import numpy as np
import scipy.sparse as sp
from copy import copy
from collections import defaultdict
from torch.utils.data import Dataset, Subset
import pandas as pd
import collections
from os.path import join
import torch
import json
# import dgl
import os
import pickle

try:
    import ipdb
except:
    pass

from utils import *

class KGDataset(Dataset):
    def __init__(self, args):
        kg_data = pd.read_csv(args.kg_path, sep='\t', names=['h', 'r', 't'], engine='python')
        self.kg_data = kg_data.drop_duplicates()
        self.kg_dict, self.heads = self.generate_kg_data(kg_data=self.kg_data)
        self.args = args

    @property
    def entity_count(self):
        # start from one
        return self.kg_data['t'].max() + 2

    @property
    def relation_count(self):
        return self.kg_data['r'].max()+2

    def get_kg_dict(self, poi_num):
        entity_num = self.args.entity_num_per_poi # 2
        p2es = dict()
        p2rs = dict()
        for poi in range(poi_num):
            rts = self.kg_dict.get(poi, False)
            if rts:
                tails = list(map(lambda x:x[1], rts))
                relations = list(map(lambda x:x[0], rts))
                if(len(tails) >= entity_num):
                    p2es[poi] = torch.IntTensor(tails).to(self.args.device)[:entity_num]
                    p2rs[poi] = torch.IntTensor(relations).to(self.args.device)[:entity_num]
                else:
                    # last embedding pos as padding idx
                    tails.extend([self.entity_count]*(entity_num-len(tails)))
                    relations.extend([self.relation_count]*(entity_num-len(relations)))
                    p2es[poi] = torch.IntTensor(tails).to(self.args.device)
                    p2rs[poi] = torch.IntTensor(relations).to(self.args.device)
            else:
                p2es[poi] = torch.IntTensor([self.entity_count]*entity_num).to(self.args.device)
                p2rs[poi] = torch.IntTensor([self.relation_count]*entity_num).to(self.args.device)
        return p2es, p2rs


    def generate_kg_data(self, kg_data):
        # construct kg dict
        kg_dict = collections.defaultdict(list)
        for row in kg_data.iterrows():
            h, r, t = row[1]
            kg_dict[h].append((r, t))
        heads = list(kg_dict.keys())
        return kg_dict, heads

    def __len__(self):
        return len(self.kg_dict)

    def __getitem__(self, index):
        head = self.heads[index]
        relation, pos_tail = random.choice(self.kg_dict[head])
        while True:
            neg_head = random.choice(self.heads)
            neg_tail = random.choice(self.kg_dict[neg_head])[1]
            if (relation, neg_tail) in self.kg_dict[head]:
                continue
            else:
                break
        return head, relation, pos_tail, neg_tail

class TravelDataset(Dataset):
    def __init__(self, args, ori_data_path, dst_data_path, trans_data_path):
        ori_raw = list(map(lambda x: x.strip().split('\t'), open(ori_data_path, 'r')))
        dst_raw = list(map(lambda x: x.strip().split('\t'), open(dst_data_path, 'r')))
        trans_raw = list(map(lambda x: x.strip().split('\t'), open(trans_data_path, 'r')))
        self.args = args

        
        self.poi_idx = {}
        self.region_idx = {}
        self.tag_idx = {}
        self.region_poi = defaultdict(set)

        self.trans = []
        self.feats = []
        self.uids = []


        for i in trans_raw:
            uid, cuid, ori_region, dst_region = i
            if ori_region not in self.region_idx: 
                self.region_idx[ori_region] = len(self.region_idx) 
            if dst_region not in self.region_idx:
                self.region_idx[dst_region] = len(self.region_idx)
            self.trans.append((self.region_idx[ori_region], self.region_idx[dst_region]))
            self.uids.append(int(uid))

        for i in ori_raw + dst_raw:
            uid, cuid, _, bid, timestamp, std_tag = i
            if bid not in self.poi_idx:
                self.poi_idx[bid] = len(self.poi_idx) + 1
            if std_tag not in self.tag_idx:
                self.tag_idx[std_tag] = len(self.tag_idx)
        self.oris = []
        self.dsts = []

        ori_buffer = []
        train_buffer = []
        dst_buffer = []

        last_uid = '0'
        for i in ori_raw:
            uid, cuid, _, bid, timestamp, std_tag = i
            self.region_poi[self.trans[int(uid)][0]].add(self.poi_idx[bid])
            if uid != last_uid:
                self.oris.append(ori_buffer)
                ori_buffer = []
                train_buffer = []
                last_uid = uid
            ori_buffer.append((self.poi_idx[bid], self.tag_idx[std_tag], timestamp))
            train_buffer.append(self.poi_idx[bid])
        self.oris.append(ori_buffer)

        last_uid = '0'
        for i in dst_raw:
            uid, cuid, _, bid, timestamp, std_tag = i
            self.region_poi[self.trans[int(uid)][1]].add(self.poi_idx[bid])
            if uid != last_uid:
                self.dsts.append(dst_buffer)
                dst_buffer = []
                last_uid = uid
            dst_buffer.append((self.poi_idx[bid], self.tag_idx[std_tag], timestamp))
        self.dsts.append(dst_buffer)

        # CRF data
        if args.crf:
            try:
                self.pp_adj = sp.load_npz(args.pp_graph_path)
                self.pp_adj = dgl.from_scipy(self.pp_adj)
                self.pp_adj = self.pp_adj.to(args.device)
            except:
                n_poi = len(self.poi_idx) + 1
                pp_graph = np.zeros((n_poi, n_poi))
                row, col = [0], [0]
                for items in self.region_poi.values():
                    for i in items:
                        row.extend([i] * len(items))
                        col.extend(items)
                data = [1] * len(row)
                self.pp_adj = sp.coo_matrix((data, (row, col)), shape=(n_poi, n_poi))
                self.pp_adj = normalize_gat(self.pp_adj)
                self.pp_adj = torch.FloatTensor(np.array(self.pp_adj.todense()))
                self.pp_adj = self.pp_adj.to(args.device)
        else:
            self.pp_adj = None

        
    def __getitem__(self, index):
        uid = self.uids[index]
        o = self.oris[index]
        d = self.dsts[index]
        t = self.trans[index]
        ori_ck = torch.LongTensor(list(map(lambda y: y[0], o)))
        dst_ck = torch.LongTensor(list(map(lambda y: y[0], d))).unique()
        ori_rg = t[0]
        dst_rg = t[1]
        
        return uid, ori_ck, dst_ck, ori_rg, dst_rg
    
    def __len__(self):
        return len(self.trans)


def random_split(dataset, split_path, ratios=[0.8, 0.1, 0.1]):
    trans = dataset.trans
    trans_by_pair = defaultdict(list)
    for u, t in enumerate(trans):
        trans_by_pair[t].append(u)
    
    train_indice, valid_indice, test_indice = [], [], []

    if os.path.exists(split_path):
        train_indice, valid_indice, test_indice = np.load(split_path, allow_pickle=True)
    else:
        for t, us in trans_by_pair.items():
            us_shuf = copy(us)
            np.random.shuffle(us_shuf)
            us_len = len(us)

            train_offset = int(us_len * ratios[0])
            valid_offset = int(us_len * (ratios[0] + ratios[1]))

            train_indice.extend(us_shuf[:train_offset])
            valid_indice.extend(us_shuf[train_offset:valid_offset])
            test_indice.extend(us_shuf[valid_offset:])

        with open('../Foursquare/data_split.pkl', 'wb') as file:
            pickle.dump([train_indice, valid_indice, test_indice], file)

    # if os.path.exists("../Foursquare/poi_pop.pkl"):
    #     print('pop exists!')
    # else:
    #     train_list = []
    #     for i in train_indice:
    #         train_list.append(dataset.oris[i])
    #         train_list.append(dataset.dsts[i])
    #     train_poi_list = [t[0] for sub_list in train_list for t in sub_list]
    #     pop_dict = {}
    #     for poi in train_poi_list:
    #         if poi in pop_dict:
    #             pop_dict[poi] += 1
    #         else:
    #             pop_dict[poi] = 1
    #     with open("../Foursquare/poi_pop.pkl", "wb") as file:
    #         pickle.dump(pop_dict, file)

    return Subset(dataset, train_indice), Subset(dataset, valid_indice), Subset(dataset, test_indice) # train_indices 是训练数据的索引列表
