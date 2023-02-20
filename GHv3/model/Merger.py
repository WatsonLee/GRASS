import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn import functional as F
from utils.Constants import *

from model.TransformerBlock import TransformerBlock


# 输入输出相同
class Merger_None(nn.Module):
    def __init__(self, opt):
        super(Merger_None, self).__init__()
        self.final_dim = opt.d_word_vec

    def forward(self, a):
        return a.cuda()

# 输入输出相同
class Merger_None_2(nn.Module):
    def __init__(self, opt):
        super(Merger_None_2, self).__init__()
        self.final_dim = opt.d_word_vec

    def forward(self, a, b):
        return a.cuda(), b.cuda()

# 直接拼接之后输出
class Merger_cat(nn.Module):
    def __init__(self, opt):
        super(Merger_cat, self).__init__()
        self.final_dim = opt.d_word_vec * 2

    def forward(self, user_dyemb, user_user_dyemb):
        dyemb = torch.cat([user_dyemb, user_user_dyemb],
                          dim=-1).cuda()  # dynamic_node_emb
        return dyemb.cuda()

# 求和方式进行筛选
class Merger_add(nn.Module):
    def __init__(self, opt):
        super(Merger_add, self).__init__()
        self.final_dim = opt.d_word_vec

    def forward(self, a, b):
        dyemb = a + b
        return dyemb.cuda()

# 求差方式进行筛选
class Merger_minus(nn.Module):
    def __init__(self, opt):
        super(Merger_minus, self).__init__()
        self.final_dim = opt.d_word_vec

    def forward(self, a, b):
        dyemb = a - b
        return dyemb.cuda()

# 求乘方式进行筛选
class Merger_mul(nn.Module):
    def __init__(self, opt):
        super(Merger_mul, self).__init__()
        self.final_dim = opt.d_word_vec

    def forward(self, a, b):
        dyemb = a * b
        return dyemb.cuda()

# 一层MLP特征筛选
class Merger_MLP_Elu(nn.Module):
    def __init__(self, opt, dropout=0.1, input_dim=128):
        super(Merger_MLP_Elu, self).__init__()

        self.final_dim = opt.d_word_vec

        self.linear = nn.Linear(input_dim, self.final_dim)
        init.xavier_normal_(self.linear.weight)

        self.elu = nn.ELU()

    def forward(self, a, b):
        ui_dyemb = torch.cat([a, b], dim=-1).cuda()
        dyemb = self.elu(self.linear(ui_dyemb)).cuda()
        return dyemb.cuda()
    
# 一层MLP特征筛选
class Merger_MLP_Tanh(nn.Module):
    def __init__(self, opt, dropout=0.1, input_dim=160):
        super(Merger_MLP_Tanh, self).__init__()

        self.final_dim = opt.transformer_dim

        self.linear = nn.Linear(input_dim, self.final_dim)
        init.xavier_normal_(self.linear.weight)

        self.norm = nn.LayerNorm(normalized_shape=self.final_dim)

        #self.tanh = torch.tanh()

    def forward(self, a, b):
        ui_dyemb = torch.cat([a, b], dim=-1).cuda()
        dyemb = torch.relu(self.linear(ui_dyemb))
        return dyemb.cuda()

# DyHGCN方式进行筛选
class Merger_DyHGCN(nn.Module):
    def __init__(self, opt):
        super(Merger_DyHGCN, self).__init__()

        self.input_dim = opt.d_word_vec
        self.final_dim = opt.d_word_vec

        self.linear_dyemb = nn.Linear(self.input_dim*4, self.final_dim)
        init.xavier_normal_(self.linear_dyemb.weight)

        self.elu = F.elu()

    def forward(self, a, b):
        # dynamic_node_emb
        dyemb = self.elu(torch.cat([a, b, a * b, a - b], dim=-1)).cuda()
        dyemb = self.elu(self.linear_dyemb(dyemb)).cuda()
        return dyemb.cuda()

class UserMerger(nn.Module):
    def __init__(self, opt):
        super(UserMerger, self).__init__()

        data_name = "./data/"+opt.data
        options = Options(data_name)

        self.input_dim = opt.d_word_vec
        self.output_dim = opt.d_word_vec
        self.linear_1 = nn.Linear(self.input_dim * 2, self.output_dim)
        init.xavier_normal_(self.linear_1.weight)
        self.linear_2 = nn.Linear(self.input_dim * 2, self.output_dim)
        init.xavier_normal_(self.linear_2.weight)
        # self.tanh=F.tanh()

    def forward(self, user_embedding, hyper_embedding = None, train = None):

        index = torch.cat([user_embedding, hyper_embedding], dim=-1)

        batch_size, max_len, dim = user_embedding.size()
        Q = (self.linear_1(index).cuda())
        K = (self.linear_2(index).cuda())
        Q_K = torch.einsum("bld,bmd->bml", Q, K).cuda()
        temperature = dim ** 0.5
        episilon = 1e-6
        Q_K = Q_K / (temperature + episilon)
        mask = torch.zeros([max_len, max_len]).cuda()
        mask += -2**32+1
        mask = torch.triu(mask, diagonal=0).cuda()

        b_mask = torch.zeros_like(Q_K).cuda()
        b_mask[:, :, :] = mask[:, :]

        Q_K += b_mask
        score = F.softmax(Q_K, dim=-1).cuda()
        output = torch.einsum("bll,bmd->bld", score, user_embedding)

        return output

class HiddenMerger(nn.Module):
    def __init__(self, opt):
        super(HiddenMerger, self).__init__()

        data_name = "./data/"+opt.data
        options = Options(data_name)

        self.input_dim = opt.transformer_dim
        self.output_dim = opt.transformer_dim
        self.linear_1 = nn.Linear(self.input_dim, self.output_dim)
        init.xavier_normal_(self.linear_1.weight)
        self.linear_2 = nn.Linear(self.input_dim, self.output_dim)
        init.xavier_normal_(self.linear_2.weight)
        # self.tanh=F.tanh()

    def forward(self, hidden, hyper_embedding = None, train = None):

        #index = torch.cat([user_embedding, hyper_embedding], dim=-1)

        batch_size, max_len, dim = hidden.size()
        Q = (self.linear_1(hyper_embedding).cuda())
        K = (self.linear_2(hyper_embedding).cuda())
        Q_K = torch.einsum("bld,bmd->bml", Q, K).cuda()
        temperature = dim ** 0.5
        episilon = 1e-6
        Q_K = Q_K / (temperature + episilon)
        mask = torch.zeros([max_len, max_len]).cuda()
        mask += -2**32+1
        mask = torch.triu(mask, diagonal=0).cuda()

        b_mask = torch.zeros_like(Q_K).cuda()
        b_mask[:, :, :] = mask[:, :]

        Q_K += b_mask
        score = F.softmax(Q_K, dim=-1).cuda()
        output = torch.einsum("bll,bmd->bld", score, hidden)

        return output
# Fusion gate
class Fusion(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.1):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out


