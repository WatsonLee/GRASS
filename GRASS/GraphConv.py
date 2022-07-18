import numpy as np
import time
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from torch_geometric.nn import GCNConv, SAGEConv,GATConv,AGNNConv
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)

        self.gnn1 = GCNConv(ninp, ninp * 2)
        self.gnn2 = GCNConv(ninp * 2, ninp)

        self.dropout = dropout
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)
        #init.xavier_normal_(self.gnn1.weight)
        #init.xavier_normal_(self.gnn2.weight)

    def forward(self, graph):
        graph_edge_index = graph.edge_index.cuda()
        # print (graph_x_embeddings.shape)
        graph_x_embeddings = F.relu(self.gnn1(self.embedding.weight, graph_edge_index))
        graph_x_embeddings = F.dropout(graph_x_embeddings, p=self.dropout, training=self.training)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        return graph_output.cuda()