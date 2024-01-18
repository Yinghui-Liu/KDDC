import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter, MultiheadAttention
import torch.nn.functional as F
import random
from utils import cuda, subgraph, eval_sampling
import sys
# import dgl
# import ipdb
import pickle
from utils import _L2_loss_mean
from GAT import GAT

try:
    import ipdb
    import tqdm
except:
    pass



class CRFLayer(nn.Module):
    """
    A Conditional Random Field (CRF) layer with attention mechanism for graph neural networks.
    Attributes:
        gamma: Parameter for the LeakyReLU activation function.
        alpha, beta: Weights for combining node embeddings and CRF outputs.
        hidden_size: Size of the hidden layer.
        W_fc: Fully connected layer for node feature transformation.
        attn_fc: Fully connected layer for calculating attention weights.
        leakyrelu: LeakyReLU activation function.
    """
    def __init__(self, hidden_size, alpha, beta, gamma=0.2, dropout=0.6):
        """
        Sets up the CRF layer with parameters for node feature transformation and edge attention
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        self.hidden_size = hidden_size

        self.W_fc = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attn_fc = nn.Linear(2 * hidden_size, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.gamma) # Activation function: f(x) = { x (if x > 0), gamma * x (otherwise) }

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['emb_attn'], edges.dst['emb_attn']], dim=1) # N x 2h
        a = self.attn_fc(z2) # N x 1
        return {'e': self.leakyrelu(a)} # N x 1s

    def message_func(self, edges):
        """
        Defines how messages are passed between nodes during the graph update.
        It uses the transformed node features and the computed attention weights.
        message UDF for equation (3) & (4)
        Returns:
            dict
        """
        return {'z': edges.src['emb_crf'], 'e': edges.data['e']}
    
    def reduce_func(self, nodes):
        alpha = torch.softmax(nodes.mailbox['e'], dim=1) # N x 1
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1) # N x h -> 1 x h
        return {'h': h}

    def forward(self, embedding_input, h_input, graph):
        """
        Transforms node features and applies the CRF mechanism to the graph.
        It combines the original node embeddings with the CRF outputs,
        weighted by the given alpha and beta parameters
        Returns:
            Tensor
        """
        z = self.W_fc(h_input)
        graph.ndata['emb_crf'] = h_input  # graph.ndata is the node feature dictionary.
        graph.ndata['emb_attn'] = z
        graph.apply_edges(self.edge_attention) # Used to transform the edges of the graph.
        # It accepts a function as a parameter, which is used to transform each edge.
        # The input to the function is the source node, target node, and edge feature data of the edge, and the output is the new feature data for each edge.
        graph.update_all(self.message_func, self.reduce_func) # Used to update all the nodes and edges on the graph.
        # This function will traverse the entire graph, executing the specified message_func and reduce_func for each node and edge.

        crf_output = graph.ndata.pop('h') # graph.ndata.pop('h') removes the element with the key 'h' from the node feature dictionary and returns its value (if it exists).
        output = (self.alpha * embedding_input + self.beta * crf_output) / (self.alpha + self.beta)

        return output


class CRF(nn.Module):
    """
    Conditional Random Field (CRF) model with an attention mechanism for graph neural networks.
    Attributes:
        alpha: Weight for the embedding input in the final output.
        beta: Weight for the CRF output in the final output.
        n_layer: Number of CRF layers.
        layer: Single instance of the CRFLayer class.
    """
    def __init__(self, args, alpha, beta, n_layer):
        """
        Sets up the CRF module with a specified number of CRF layers, and weights for combining node embeddings and CRF outputs
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_layer = n_layer

        # self.layers = nn.ModuleList([CRFLayer(args.hidden_size, alpha, beta) for _ in range(self.n_layer)])
        self.layer = CRFLayer(args.hidden_size, alpha, beta) # alpha

    def message_func(self, edges):
        """
        Message function for graph updates.
        Returns:
            dict
        """
        return {'z': edges.src['emb_crf'], 'e': edges.data['e'], 't': edges.dst['emb_crf']}
    
    def reduce_func(self, nodes):
        """
        Reduce function for aggregating messages at nodes.
        Starts by taking batches from nodes without edges, and updates nodes with the same number of target nodes together, forming a batch.
        Returns:
            dict
        """
        alpha = torch.softmax(nodes.mailbox['e'], dim=1)
        # p=2 specifies that the function should calculate the L2 norm
        h = torch.sum(alpha.squeeze(2) * torch.norm(nodes.mailbox['t'] - nodes.mailbox['z'], p=2, dim=-1), dim=1)
        return {'loss': h}
    
    def forward(self, embedding_input, graph, is_train=True):
        """
        Applies the CRF layers to update node embeddings based on the graph structure and calculates the loss during training
        Returns:
            tuple
        """
        for n in range(self.n_layer):
            if n == 0:
                h_input = embedding_input
            h_input = self.layer(embedding_input, h_input, graph) # embedding_input: initial embeddings
        
        if is_train:
            loss_a = torch.norm(h_input - embedding_input, 2, -1) ** 2

            graph.ndata['emb_crf'] = h_input
            graph.update_all(self.message_func, self.reduce_func)
            loss_b = graph.ndata.pop('loss')

            loss = torch.mean(self.alpha * loss_a + self.beta * loss_b)
        else:
            loss = None
        return h_input, loss
        

class KDDC(nn.Module):
    """
    KDDC (Knowledge-Driven Disentangled Causal) is a neural network model for processing travel data with knowledge graphs.
    """
    def __init__(self, args, n_poi, n_region, region_poi, pp_adj, kg_dataset=None):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.projection_dim = args.projection_dim
        self.n_poi = n_poi
        self.n_region = n_region
        self.poi_embedding = nn.Embedding(self.n_poi, self.hidden_size)

        if self.args.kg:
            self.kg_dataset = kg_dataset
            self.n_entities = self.kg_dataset.entity_count
            self.n_relations = self.kg_dataset.relation_count
            self.entity_embedding = nn.Embedding(self.n_entities + 1, self.hidden_size)
            self.relations_embedding = nn.Embedding(self.n_relations + 1, self.hidden_size)
            self.kg_dict, self.poi2relations = self.kg_dataset.get_kg_dict(self.n_poi)

        self.margin = args.margin
        self.pp_adj = pp_adj
        self.region_poi = region_poi

        if self.args.trans == 'transr':
            self.projection_matrix = nn.Linear(self.hidden_size, self.projection_dim)
        self.region_embedding = nn.Embedding(self.n_region, self.hidden_size)

        self.head_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.tail_linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.crf = CRF(args, args.alpha, args.beta, args.crf_layer)
        self.tau = args.tau

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.SiLU()
        )

        self.gat = GAT(self.hidden_size, self.hidden_size, dropout=0.4, alpha=0.2).train()

        if self.args.mode == 'test':
            self.user_mem_emb_clct = []
            self.user_ht_emb_clct = []
            self.user_oft_emb_clct = []
            self.user_ht_ck_clct = []
            self.user_oft_ck_clct = []
            self.user_neg_ck_clct = []
            self.user_slot_attn_clct = []
            self.user_reg_attn_clct = []

        if self.args.dc:
            with open(self.args.pop_path, "rb") as pkl_file:
                poi_pop_dict = pickle.load(pkl_file)
                poi_pop_size = len(poi_pop_dict)
                poi_pop_list = [poi_pop_dict.get(p, 0.0) for p in range(self.n_poi)]
                poi_pop_max = max(poi_pop_list)
                poi_pop_min = min(poi_pop_list)
                self.poi_pop_tensor = torch.tensor(poi_pop_list, dtype=torch.float32)
                self.poi_pop_norm_tensor = torch.tensor([e / poi_pop_max for e in poi_pop_list], dtype=torch.float32).to(self.args.device)

        self.cont_emb_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.pop_emb_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.conf_infer_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU()
        )
        self.int_infer_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU()
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes or resets the parameters of the model uniformly.
        This method is typically called to ensure consistent initial weights
        across different runs or before training starts.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    def _avg_pooling(self, ck, emb):
        """
        Applies average pooling to embeddings.
        Returns:
            Tensor
        """
        emb_sum = torch.sum(emb, axis=1)
        row_count = torch.sum(ck != 0, axis=-1)
        emb_agg = emb_sum / row_count.unsqueeze(1).expand_as(emb_sum)
        return emb_agg


    def neg_sampling(self, k, d_ck, is_mask):
        """
        Generates negative samples for each data point in a batch.
        Returns:
            Tensor
        """
        neg_samples = []
        for idx, t in enumerate(d_ck):
            t = t.cpu().numpy()
            # np.setdiff1d is used to compute the difference between two one-dimensional arrays.
            # It returns the values that appear in the first array but not in the second array.
            neg_sample = np.random.choice(np.setdiff1d(np.arange(self.n_poi), t), k)
            # Replace positions in tensor t that are equal to 0 with 0, and positions that are not equal to 0 with neg_sample.
            if is_mask: neg_sample = np.where(t == 0, 0, neg_sample)
            neg_samples.append(torch.LongTensor(neg_sample))
        neg_tensor = torch.stack(neg_samples, dim=0).to(self.args.device)
        return neg_tensor

    def _evaluate(self, o_emb, d_ck, d_rg, r, rel_embs, poi_embs, neg_sample, subgraph_alias):
        """
        Evaluates the model on a given batch of data.
        Returns:
            score
            eval_sample
        """
        # o_emb_dup: b x l x h
        # d_ck: b x l
        neg_sample_poi = neg_sample
        eval_sample = torch.cat([d_ck, neg_sample_poi], dim=-1)
        eval_sample_re = subgraph_alias[eval_sample]

        eval_sample_emb_ori = poi_embs[eval_sample_re] # b x l x h
        if self.args.dc:
            eval_sample_pop_emb = self.pop_emb_layer(eval_sample_emb_ori)
            if self.args.dm:
                eval_sample_pop_emb = self._relation(eval_sample_pop_emb, r, rel_embs, 'transd')
            eval_sample_cont_emb = self.cont_emb_layer(eval_sample_emb_ori)
            if self.args.dm:
                eval_sample_cont_emb = self._relation(eval_sample_cont_emb, r, rel_embs, 'transd')
            eval_sample_emb = torch.cat([eval_sample_pop_emb, eval_sample_cont_emb], dim=-1)
        else:
            eval_sample_emb = eval_sample_emb_ori
        o_emb_dup = o_emb.unsqueeze(1).expand_as(eval_sample_emb) # b x l x h、
        score = torch.norm(o_emb_dup - eval_sample_emb, p=2, dim=-1) # b x l
        score = torch.where(eval_sample == 0, np.inf, score.double()).float() # mask
        return score, eval_sample

    def calc_kg_loss_transE(self, h, r, pos_t, neg_t):
        """
        Calculates the loss for the model using the TransE approach.
        Args:
            h:      (kg_batch_size)
            r:      (kg_batch_size)
            pos_t:  (kg_batch_size)
            neg_t:  (kg_batch_size)
        Returns:
            loss
        """
        # (kg_batch_size, relation_dim) There are a total of 44 relation types, each corresponding to a 64-dimensional embedding vector.
        # Each sample corresponds to an index of a relation type, and embedding_relation converts the index of each relation type into the corresponding embedding vector.
        r_embed = self.relations_embedding(r)
        h_embed = self.poi_embedding(h) # (kg_batch_size, entity_dim)
        pos_t_embed = self.entity_embedding(pos_t) # (kg_batch_size, entity_dim)
        neg_t_embed = self.entity_embedding(neg_t) # (kg_batch_size, entity_dim)
        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1) # (kg_batch_size) As per the formula f_d in the paper.
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1) # (kg_batch_size)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        # This value can be considered as the "energy" of the input samples.
        # This code is typically used for calculating regularization terms in the loss function.
        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        # # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        return loss

    def calc_kg_loss_transR(self, h, r, pos_t, neg_t):
        """
        Calculates the loss for the model using the TransR approach.
        Args:
            h:      (kg_batch_size)
            r:      (kg_batch_size)
            pos_t:  (kg_batch_size)
            neg_t:  (kg_batch_size)
        Returns:
            loss
        """
        r_embed = self.projection_matrix(self.relations_embedding(r))
        h_embed = self.projection_matrix(self.poi_embedding(h))
        pos_t_embed = self.projection_matrix(self.entity_embedding(pos_t))
        neg_t_embed = self.projection_matrix(self.entity_embedding(neg_t))
        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        # # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        return loss

    def calc_kg_loss_SEEK(self, h, r, pos_t, neg_t):
        """
        Calculates the loss using the SEEK approach for knowledge graph embeddings.
        Args:
            h:      (kg_batch_size)
            r:      (kg_batch_size)
            pos_t:  (kg_batch_size)
            neg_t:  (kg_batch_size)
        Returns:
            loss
        """
        # The 44 here indicates that there are a total of 44 types of relations, each corresponding to a 64-dimensional embedding vector.
        # Each sample corresponds to an index of a relation type, and the embedding_relation converts the index of each relation type into the corresponding embedding vector.
        r_embed = self.relations_embedding(r)        # (kg_batch_size, relation_dim)
        h_embed = self.poi_embedding(h)               # (kg_batch_size, entity_dim)
        pos_t_embed = self.entity_embedding(pos_t)      # (kg_batch_size, entity_dim)
        neg_t_embed = self.entity_embedding(neg_t)      # (kg_batch_size, entity_dim)

        k_num = self.args.segments
        rank = int(self.hidden_size / k_num)
        h = [h_embed[i * rank : (i + 1) * rank] for i in range(k_num)]
        h = tuple(h)
        r = [r_embed[i * rank : (i + 1) * rank] for i in range(k_num)]
        r = tuple(r)
        pos_t = [pos_t_embed[i * rank : (i + 1) * rank] for i in range(k_num)]
        pos_t = tuple(pos_t)
        neg_t = [neg_t_embed[i * rank : (i + 1) * rank] for i in range(k_num)]
        neg_t = tuple(neg_t)
        pos_tmp = 0
        neg_tmp = 0

        for x in range(k_num):
            for y in range(k_num):
                s = -1 if x % 2 != 0 and x + y >= k_num else 1
                w = y if x % 2 == 0 else (x + y) % k_num
                pos_tmp += s * r[x] * h[y] * pos_t[w]
                neg_tmp += s * r[x] * h[y] * neg_t[w]
        pos_score = torch.sum(pos_tmp, 1)
        neg_score = torch.sum(neg_tmp, 1)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        # This value can be considered as the "energy" of the input samples.
        # This code is typically used for calculating regularization terms in the loss function.
        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        # # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        # loss = kg_loss
        return loss

    def _relation(self, emb, r, rel_embs, mode='transr'):
        """
        Applies a relation transformation to the embeddings, Dynamic Mapping.
        Args:
            o_emb: b x h
            d_emb: b x l x h
        Returns:
            Tensor
        """
        if mode == 'transr':
            relation = rel_embs[r].view(-1, self.hidden_size, self.hidden_size) # b x (h x h)
            if len(emb.shape) == 2:
                emb_r = torch.bmm(emb.unsqueeze(1), relation).squeeze(1)
                return emb_r
            elif len(emb.shape) == 3:
                emb_r = torch.matmul(emb.unsqueeze(2), relation.unsqueeze(1).expand(-1, emb.size(1), -1, -1)).squeeze(2)
                return emb_r
        if mode == 'transd':
            relation = rel_embs[r] # b x h Embeddings of 64 regions (cities) visited by users
            if len(emb.shape) == 2:
                # b x h x h matrix multiplication
                # equivalent to the embedding weights of the region (city) multiplied by the node embedding that has passed through a linear layer.
                trans_mat = torch.matmul(relation.unsqueeze(2), self.head_linear(emb).unsqueeze(1)) # b x h x h
                # torch.bmm() might be faster, but both are matrix-level multiplication
                emb_r = torch.bmm(emb.unsqueeze(1), trans_mat).squeeze(1)
                return emb_r
            elif len(emb.shape) == 3:
                # b x h x h (64, 13, 128, 1) * (64, 13, 1, 128)
                trans_mat = torch.matmul(relation.view(relation.size(0), 1, -1, 1).expand(-1, emb.size(1), -1, -1), self.tail_linear(emb).unsqueeze(2)) # b x h x h
                emb_r = torch.matmul(emb.unsqueeze(2), trans_mat).squeeze(2)
                return emb_r
        if mode == 'transe':
            return emb

    def drop_edge_random(self, poi2entities, p_drop, padding):
        """
        Randomly drops edges from the POI to entity mappings.
        Returns:
            dict
        """
        res = dict()
        for item, es in poi2entities.items():
            new_es = list()
            for e in es.tolist():
                if (random.random() > p_drop):
                    new_es.append(e)
                else:
                    new_es.append(padding)
            res[item] = torch.IntTensor(new_es).to(self.args.device)
        return res

    def get_kg_views(self):
        """
        Generates two views of the knowledge graph by randomly dropping edges.
        Returns:
            tuple
        """
        kg = self.kg_dict
        view1 = self.drop_edge_random(kg, self.args.kg_p_drop, self.n_entities)
        view2 = self.drop_edge_random(kg, self.args.kg_p_drop, self.n_entities)
        return view1, view2

    def cal_poi_embedding_mean(self, kg: dict):
        """
        Calculates the mean embeddings of POIs based on their associated entities.
        Returns:
            Tensor
        """
        poi_embs = self.poi_embedding(torch.IntTensor(list(kg.keys())).to(self.args.device)) #poi_num, emb_dim
        poi_entities = torch.stack(list(kg.values())) # poi_num, entity_num_each
        entity_embs = self.entity_embedding(poi_entities) # poi_num, entity_num_each, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(poi_entities!=self.n_entities, torch.ones_like(poi_entities), torch.zeros_like(poi_entities)).float()
        # padding is zero
        entity_embs = entity_embs * padding_mask.unsqueeze(-1).expand(entity_embs.size())
        # poi_num, emb_dim
        entity_embs_sum = entity_embs.sum(1)
        entity_embs_mean = entity_embs_sum / padding_mask.sum(-1).unsqueeze(-1).expand(entity_embs_sum.size())
        # replace nan with zeros
        entity_embs_mean = torch.nan_to_num(entity_embs_mean)
        # poi_num, emb_dim
        return poi_embs+entity_embs_mean

    def cal_poi_embedding_gat(self, kg:dict):
        """
        Calculates the POI embeddings using a Graph Attention Network (GAT) based on the associated entities.
        Returns:
            Tensor
        """
        poi_embs = self.poi_embedding(torch.IntTensor(list(kg.keys())).to(self.args.device)) #poi_num, emb_dim
        poi_entities = torch.stack(list(kg.values())) # poi_num, entity_num_each
        entity_embs = self.entity_embedding(poi_entities) # poi_num, entity_num_each, emb_dim
        # poi_num, entity_num_each
        padding_mask = torch.where(poi_entities!=self.n_entities, torch.ones_like(poi_entities), torch.zeros_like(poi_entities)).float()
        return self.gat(poi_embs, entity_embs, padding_mask)

    def cal_poi_embedding_rgat(self, kg:dict):
        """
        Calculates POI embeddings using a Relational Graph Attention Network (RGAT).
        Returns:
            Tensor
        """
        poi_embs = self.poi_embedding(torch.IntTensor(list(kg.keys())).to(self.args.device)) #poi_num, emb_dim
        poi_entities = torch.stack(list(kg.values())) # poi_num, entity_num_each
        poi_relations = torch.stack(list(self.poi2relations.values()))
        entity_embs = self.entity_embedding(poi_entities) # poi_num, entity_num_each, emb_dim
        relation_embs = self.relations_embedding(poi_relations) # poi_num, entity_num_each, emb_dim
        padding_mask = torch.where(poi_entities!=self.n_entities, torch.ones_like(poi_entities), torch.zeros_like(poi_entities)).float()
        return self.gat.forward_relation(poi_embs, entity_embs, relation_embs, padding_mask)

    def cal_poi_embedding_from_kg(self, kg: dict):
        """
        Calculates POI embeddings based on the specified knowledge graph convolution method.
        Returns:
            Tensor
        """
        if kg is None:
            kg = self.kg_dict

        if(self.args.kgcn=="GAT"):
            return self.cal_poi_embedding_gat(kg)
        elif self.args.kgcn=="RGAT":
            return self.cal_poi_embedding_rgat(kg)
        elif(self.args.kgcn=="MEAN"):
            return self.cal_poi_embedding_mean(kg)
        elif(self.args.kgcn=="NO"):
            return self.poi_embedding.weight

    def get_ui_views_weighted(self, poi_stabilities, stab_weight):
        """
        Calculates weighted POI views based on stability scores.
        Returns:
            Tensor
        """
        # kg probability of keep
        poi_stabilities = torch.exp(poi_stabilities)
        kg_weights = (poi_stabilities - poi_stabilities.min()) / (poi_stabilities.max() - poi_stabilities.min())
        # Replace elements in kg_weights less than or equal to 0.3 with 0.3, keep elements greater than 0.3 unchanged.
        kg_weights = kg_weights.where(kg_weights > 0.3, torch.ones_like(kg_weights) * 0.3)
        weights = (1-self.args.ui_p_drop)/torch.mean(stab_weight*kg_weights)*(stab_weight*kg_weights)
        # weights = weights.where(weights>0.3, torch.ones_like(weights) * 0.3)
        # Replace elements in weights greater than or equal to 0.95 with 0.95, keep elements less than 0.95 unchanged.
        weights = weights.where(weights<0.95, torch.ones_like(weights) * 0.95)
        # Perform Bernoulli sampling to get a mask tensor poi_mask of the same dimension as weights,
        # where the probability of an element being True is the corresponding value in weights.
        # Values are chosen as 1 or 0 with probabilities p and 1-p, respectively.
        poi_mask = torch.bernoulli(weights).to(torch.bool)
        # drop
        poi_mask.requires_grad = False
        return poi_mask

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        Calculates the similarity between two tensors.
        Returns:
            Tensor
        """
        if z1.size()[0] == z2.size()[0]:
            return F.cosine_similarity(z1,z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

    def poi_kg_stability(self, view1, view2):
        """
        Computes the stability of POI embeddings across two views of the knowledge graph.
        Returns:
            Tuple
        """
        kgv1_ro = self.cal_poi_embedding_from_kg(view1)
        kgv2_ro = self.cal_poi_embedding_from_kg(view2)
        sim = self.sim(kgv1_ro, kgv2_ro)
        return kgv1_ro, kgv2_ro, sim


    def get_views(self, aug_side="both"):
        """
        Generates augmented views for contrastive learning.
        Returns:
            Dict
        """
        # drop (epoch based)
        # kg drop -> 2 views -> view similarity for item
        # Randomly remove tail entities and fill in the removed parts.
        kgv1, kgv2 = self.get_kg_views()
        # [item_num]
        kgv1, kgv2, stability = self.poi_kg_stability(kgv1, kgv2)  # Calculate consistency
        kgv1 = kgv1.to(self.args.device)
        kgv2 = kgv2.to(self.args.device)
        stability = stability.to(self.args.device)
        # item drop -> 2 views
        # Delete the user-item interaction edges (deleting edges with item nodes as index) from the interaction graph.
        v1_mask = self.get_ui_views_weighted(stability, 1)
        # uiv2 = self.ui_drop_random(world.ui_p_drop)
        v2_mask = self.get_ui_views_weighted(stability, 1)

        contrast_views = {
            "kgv1": kgv1,
            "kgv2": kgv2,
            "uiv1": v1_mask,
            "uiv2": v2_mask
        }
        return contrast_views

    def info_nce_loss_overall(self, z1, z2):
        """
        Calculates the InfoNCE loss, a contrastive loss used for learning efficient embeddings.
        Returns:
            torch.Tensor
        """
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        batch_size, d_model = z1.shape
        features = torch.cat([z1, z2], dim=0)  # (batch_size * 2, d_model)

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [batch_size * 2, 1]

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [batch_size * 2, 2N-2]

        logits = torch.cat([positives, negatives], dim=1)  # (batch_size * 2, batch_size * 2 - 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)  # (batch_size * 2, 1)
        logits = logits / self.tau

        loss = criterion(logits, labels)
        return loss

    def pop_func(self, pop_tensor, pop_coeff):
        """
        Adjusts the popularity tensor using an exponential decay function.
        Returns:
            torch.Tensor
        """
        pop_tensor = torch.mul(pop_tensor, pop_coeff)
        pop_tensor = torch.where(pop_tensor >= 1.0, torch.ones_like(pop_tensor), pop_tensor)
        pop_curve = torch.exp(-pop_tensor)
        return pop_curve

    def _alias(self, bids, n_poi):
        """
        Creates an alias tensor mapping original POI indices to a new set of indices.
        Returns:
            torch.Tensor
        """
        alias = torch.zeros(n_poi).long()
        for idx, b in enumerate(bids):
            alias[b] = idx
        alias = alias.to(self.args.device)
        return alias
    
    def _eval_sample(self, region_poi, o_rg, d_ck, batch_size, device, k=100):
        """
        Generates evaluation samples for each user-region pair.
        Returns:
            Tensor
        """
        poi_samples = []
        # If d_ck and o_rg are respectively [1, 0, 1] and [0, 1, 1], then the result of zip(d_ck, o_rg) would be [(1, 0), (0, 1), (1, 1)].
        # Each tuple can be seen as the information of a user at a certain moment, which includes whether the user appears at each POI and the region to which the user belongs.
        for t_p, t_or in zip(d_ck, o_rg):
            # Randomly select k elements from the given array to generate a NumPy array of shape (1, k), and assign it to variable s.
            s = np.random.choice(np.setdiff1d(list(range(1, self.n_poi)), t_p.cpu().tolist() + list(region_poi[t_or.item()])), (1, k))
            poi_samples.append(torch.from_numpy(s))
        poi_samples = torch.cat(poi_samples, dim=0).to(device)
        return poi_samples

    def forward(self, uid, o_ck, d_ck, o_rg, d_rg):
        """
        Forward pass of the KDDC model.
        Returns:
            tuple
        """
        # Generate negative samples for each data point in a batch
        d_neg_items = self.neg_sampling(d_ck.size(1), d_ck, is_mask=True) # b x l
        # Concatenate all items involved in the batch and find the unique items
        item_involve = torch.cat([o_ck.flatten(), d_ck.flatten(), d_neg_items.flatten()])
        item_involve = torch.unique(item_involve)
        # Retrieve the region embeddings
        region_repr = self.region_embedding.weight
        # Process for knowledge graph and contrastive learning, if enabled
        # KGCL
        if self.args.kg and self.args.contrast:
            contrast_views = self.get_views()
            kgv1 = contrast_views["kgv1"]
            kgv2 = contrast_views["kgv2"]
            mask1 = contrast_views["uiv1"]
            mask2 = contrast_views["uiv2"]
            ck_involve = torch.unique(torch.cat([o_ck.flatten(), d_ck.flatten()]))
            p_loss = self.info_nce_loss_overall(kgv1[ck_involve], kgv2[ck_involve])
            ck = torch.cat([o_ck, d_ck], dim=-1)
            o_ckv1 = torch.where(mask1[ck[:, ]], ck[:, ], torch.zeros_like(ck[:, ]))
            o_ckv2 = torch.where(mask2[ck[:, ]], ck[:, ], torch.zeros_like(ck[:, ]))
            o_emb1 = kgv1[o_ckv1]
            o_emb2 = kgv2[o_ckv2]
            o_emb1 = self._avg_pooling(ck, o_emb1)
            o_emb2 = self._avg_pooling(ck, o_emb2)
            up_loss = self.info_nce_loss_overall(o_emb1, o_emb2)
            c_loss = p_loss + up_loss
            # c_loss = p_loss
        # Create subgraph for CRF processing, if CRF is enabled.
        if self.args.crf:
            sub_pp_adj = self.pp_adj.subgraph(item_involve)
        # If CRF is enabled, process the node embeddings with CRF, otherwise, use the node embeddings as is

        # POI Semantic Knowledge Aggregation
        if self.args.kg:
            node_embed = self.cal_poi_embedding_from_kg(self.kg_dict)
            # As long as the dimension is less than the maximum dimension, it is fine, no need to be equal.
            node_repr = node_embed[item_involve]
        else:
            node_repr = self.poi_embedding(item_involve)

        # If CRF is enabled, process the node embeddings with CRF, otherwise, use the node embeddings as is
        if self.args.crf:
            poi_repr, crf_loss = self.crf(node_repr, sub_pp_adj)
        else:
            poi_repr = node_repr

        # Reindex nodes according to the subgraph alias
        # Map the index of POIs in the subgraph into the tensor of all POIs, that is,
        # the values at the corresponding positions of all POIs are set to the index in the subgraph.
        subgraph_alias = self._alias(item_involve, self.n_poi)
        # The value is the POI's number in the subgraph
        o_ck_re = subgraph_alias[o_ck]
        d_ck_re = subgraph_alias[d_ck]
        # The negative samples are added when constructing the item_involve (subgraph).
        d_neg_re = subgraph_alias[d_neg_items]

        # Retrieve embeddings for origin check-ins, destination check-ins, and negative samples.
        o_items_emb = poi_repr[o_ck_re] #self.poi_embedding(o_ck)

        d_neg_emb = poi_repr[d_neg_re] #self.poi_embedding(d_neg_items) # b x l x h
        # The out-of-town embedding before average pooling
        d_target_emb = poi_repr[d_ck_re] #self.poi_embedding(d_ck) # b x l x h

        # Process embeddings with popularity and content layers,
        # if Disentangled Causal Metric Learning (DC) is enabled
        if self.args.dc:
            o_pop_emb = self.pop_emb_layer(o_items_emb)  # b * l * h
            if self.args.dm:
                o_pop_emb = self._relation(o_pop_emb, o_rg, region_repr, 'transd')
            o_cont_emb = self.cont_emb_layer(o_items_emb)  # b * l * h
            if self.args.dm:
                o_cont_emb = self._relation(o_cont_emb, o_rg, region_repr, 'transd')

            o_emb = torch.cat([o_pop_emb, o_cont_emb], dim=-1)
            u_o_emb = self._avg_pooling(o_ck, o_emb)


            d_pop_neg_emb = self.pop_emb_layer(d_neg_emb)
            if self.args.dm:
                d_pop_neg_emb = self._relation(d_pop_neg_emb, o_rg, region_repr, 'transd')
            d_cont_neg_emb = self.cont_emb_layer(d_neg_emb)
            if self.args.dm:
                d_cont_neg_emb = self._relation(d_cont_neg_emb, o_rg, region_repr, 'transd')
            neg_emb = torch.cat([d_pop_neg_emb, d_cont_neg_emb], dim=-1)

            d_pop_target_emb = self.pop_emb_layer(d_target_emb)
            if self.args.dm:
                d_pop_target_emb = self._relation(d_pop_target_emb, o_rg, region_repr, 'transd')
            d_cont_target_emb = self.cont_emb_layer(d_target_emb)
            if self.args.dm:
                d_cont_target_emb = self._relation(d_cont_target_emb, o_rg, region_repr, 'transd')
            target_emb = torch.cat([d_pop_target_emb, d_cont_target_emb], dim=-1)

            if self.args.infer:
                # users' conf preferences
                u_o_conf_emb = self._avg_pooling(o_ck, o_pop_emb)  # b * h
                infer_conf_emb = self.conf_infer_mlp(u_o_conf_emb)
                o_pop_emb_dup = infer_conf_emb.unsqueeze(1).expand_as(d_pop_target_emb)  # b * l * h
                # users' int preferences
                u_o_int_emb = self._avg_pooling(o_ck, o_cont_emb)  # b * h
                infer_int_emb = self.int_infer_mlp(u_o_int_emb)
                o_cont_emb_dup = infer_int_emb.unsqueeze(1).expand_as(d_cont_target_emb)  # b * l * h

                # eq.(12)
                poi_pop_tensor = self.poi_pop_norm_tensor[d_ck.flatten()].view(d_ck.size())
                mask_poi_cont = self.pop_func(poi_pop_tensor, self.args.pop_coeff)
                mask_poi_pop = torch.ones_like(mask_poi_cont) - mask_poi_cont
                infer_loss = mask_poi_cont * torch.norm(o_cont_emb_dup - d_cont_target_emb, p=2, dim=-1) + \
                             mask_poi_pop * torch.norm(o_pop_emb_dup - d_pop_target_emb, p=2, dim=-1)
                infer_intend = torch.cat([infer_conf_emb, infer_int_emb], dim=-1)

                fusion_emb = self.fusion_mlp(torch.cat([u_o_emb, infer_intend], dim=-1))
                o_emb_dup = fusion_emb.unsqueeze(1).expand_as(target_emb)
            else:
                fusion_emb = u_o_emb
                o_emb_dup = fusion_emb.unsqueeze(1).expand_as(target_emb)

            s_pos = torch.norm(o_emb_dup - target_emb, p=2, dim=-1) ** 2  # b x l
            s_neg = torch.norm(o_emb_dup - neg_emb, p=2, dim=-1) ** 2  # b x l
            if self.args.infer:
                loss = 1 * torch.relu(s_pos - s_neg + self.args.margin).mean() + 1 * infer_loss.mean()
            else:
                loss = torch.relu(s_pos - s_neg + self.args.margin).mean()
        else:
            o_emb = self._avg_pooling(o_ck, o_items_emb)
            o_emb_dup = o_emb.unsqueeze(1).expand_as(d_target_emb)

            s_pos = torch.norm(o_emb_dup - d_target_emb, p=2, dim=-1) ** 2 # b x l
            s_neg = torch.norm(o_emb_dup - d_neg_emb, p=2, dim=-1) ** 2 # b x l

            loss = torch.relu(s_pos - s_neg + self.args.margin).mean()
        if self.args.crf: loss += crf_loss
        if self.args.kg and self.args.contrast: loss += c_loss
        
        return loss, poi_repr

    def rank(self, uid, o_ck, d_ck, o_rg, d_rg):
        """
        Ranking method for the KDDC model, used for evaluation.
        Returns:
            tuple
        """
        # Negative sampling of POIs from other regions (cities), excluding POIs from the user's own hometown (used here in o_rg).
        eval_p = self._eval_sample(self.region_poi, o_rg, d_ck, uid.size(0), self.args.device)
        item_involve = torch.cat([o_ck.flatten(), d_ck.flatten(), eval_p.flatten()])
        item_involve = torch.unique(item_involve)

        if self.args.crf: sub_pp_adj = self.pp_adj.subgraph(item_involve)

        if self.args.kg:
            node_embed = self.cal_poi_embedding_from_kg(self.kg_dict)
            node_repr = node_embed[item_involve]
        else:
            node_repr = self.poi_embedding(item_involve)
        # generic graph
        if self.args.crf: 
            poi_repr, _ = self.crf(node_repr, sub_pp_adj)
        else:
            poi_repr = node_repr
        region_repr = self.region_embedding.weight

        # reindex by alias
        subgraph_alias = self._alias(item_involve, self.n_poi)
        o_ck_re = subgraph_alias[o_ck]
        # home town avg pooling
        o_items_emb = poi_repr[o_ck_re] #self.poi_embedding(o_ck)

        if self.args.dc:

            o_pop_emb = self.pop_emb_layer(o_items_emb)
            if self.args.dm:
                o_pop_emb = self._relation(o_pop_emb, o_rg, region_repr, 'transd')

            o_cont_emb = self.cont_emb_layer(o_items_emb)
            if self.args.dm:
                o_cont_emb = self._relation(o_cont_emb, o_rg, region_repr, 'transd')

            o_emb = torch.cat([o_pop_emb, o_cont_emb], dim=-1)
            u_o_emb = self._avg_pooling(o_ck, o_emb)

            if self.args.infer:
                u_o_conf_emb = self._avg_pooling(o_ck, o_pop_emb)  # b * h
                infer_conf_emb = self.conf_infer_mlp(u_o_conf_emb)
                u_o_int_emb = self._avg_pooling(o_ck, o_cont_emb)  # b * h
                infer_int_emb = self.int_infer_mlp(u_o_int_emb)

                infer_intend = torch.cat([infer_conf_emb, infer_int_emb], dim=-1)
                fusion_emb = self.fusion_mlp(torch.cat([u_o_emb, infer_intend], dim=-1))
            else:
                fusion_emb = u_o_emb
        else:
            o_emb = self._avg_pooling(o_ck, o_items_emb)  # b x h
            fusion_emb = o_emb
        # (32, 113) The first 13 are positive samples, and the last 100 are negative samples.
        score, alias_poi = self._evaluate(fusion_emb, d_ck, d_rg, o_rg, region_repr, poi_repr, eval_p, subgraph_alias)
        return score, alias_poi
