import os
import pickle

import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import RGCNConv, GCNConv, HypergraphConv
from utils.Constants import *
from utils.GraphBuilder import LoadHeteStaticGraph, LoadHyperGraph


# basic class for graph embedding generater
class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.1):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)

        self.gnn1 = GCNConv(ninp, ninp * 2)
        self.gnn2 = GCNConv(ninp * 2, ninp)

        self.gnn3 = HypergraphConv(ninp, ninp * 2)
        self.gnn4 = HypergraphConv(ninp, ninp)

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)
        # init.xavier_normal_(self.gnn1.weight)
        # init.xavier_normal_(self.gnn2.weight)

    def forward(self, heter_graph, hyper_graph):
        heter_graph_edge_index = heter_graph.edge_index.cuda()
        heter_graph_edge_type = heter_graph.edge_type.cuda()
        # print (graph_x_embeddings.shape)
        heter_graph_x_embeddings = self.gnn1(self.embedding.weight, heter_graph_edge_index)
        heter_graph_x_embeddings = self.dropout(heter_graph_x_embeddings)
        heter_graph_output = self.gnn2(heter_graph_x_embeddings, heter_graph_edge_index)

        hyper_graph_edge_index = hyper_graph.edge_index.cuda()
        # print (graph_x_embeddings.shape)
        #hyper_graph_x_embeddings = self.gnn3(heter_graph_output.data.clone(), hyper_graph_edge_index)
        #hyper_graph_x_embeddings = self.dropout(hyper_graph_x_embeddings)
        hyper_graph_output = self.gnn4(heter_graph_output.data.clone(), hyper_graph_edge_index)

        return heter_graph_output.cuda(), hyper_graph_output.cuda()


class GraphEncoder(nn.Module):
    def __init__(self, opt, Type="", dropout=0.15):
        super(GraphEncoder, self).__init__()
        self.dropedge=opt.dropout

        self.ntoken = opt.ntoken
        self.ninp = opt.d_word_vec
        self.output_dim=opt.d_word_vec

        self.heter_graph = LoadHeteStaticGraph(opt.data_path,Type)
        self.hyper_graph = LoadHyperGraph(opt.data_path,Type)

        self.gnn_layer = GraphNN(self.ntoken, self.ninp)
        #self.hyper_layer = HyperGraphNN(self.ntoken, self.ninp)

    def forward(self,input,input_timestamp,train=True):

        batch_size, max_len = input.size()

        # if train:
        #     graph=DropEdge(self.edges_list,self.edges_type_list,self.dropedge)
        # else :

        user_social_embedding_lookup, user_hyper_embedding_lookup = self.gnn_layer(self.heter_graph, self.hyper_graph)
        #user_hyper_embedding_lookup = self.hyper_layer(self.hyper_graph).cuda()  # [user_size, user_embedding]

        user_input=input.contiguous().view(batch_size*max_len,1).cuda()
        user_social_embedding_one_hot=torch.zeros(batch_size*max_len, self.ntoken).cuda()
        user_social_embedding_one_hot=user_social_embedding_one_hot.scatter_(1, user_input, 1)

        user_social_embedding=torch.einsum("bt,td->bd",user_social_embedding_one_hot,user_social_embedding_lookup).view(batch_size,max_len,self.ninp).cuda()
        user_hyper_embedding = torch.einsum("bt,td->bd", user_social_embedding_one_hot,
                                             user_hyper_embedding_lookup).view(batch_size, max_len, self.ninp).cuda()

        return user_social_embedding.cuda(), user_hyper_embedding.cuda()


