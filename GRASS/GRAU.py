import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


class GRAUCell_withSS(nn.Module):
    """自定义GRUCell"""
    def __init__(self, input_size, hidden_size, use_network = 1):
        super(GRAUCell_withSS, self).__init__()
        # 输入变量的线性变换过程是 x @ W.T + b (@代表矩阵乘法， .T代表矩阵转置)
        # in2hid_w 的原始形状应是 (hidden_size, input_size), 为了编程的方便, 这里改成(input_size, hidden_size)
        lb, ub = -math.sqrt(1/hidden_size), math.sqrt(1/hidden_size)
        self.in2hid_w = nn.ParameterList([self.__init(lb, ub, input_size, hidden_size) for _ in range(2)])
        self.hid2hid_w = nn.ParameterList([self.__init(lb, ub, hidden_size, hidden_size) for _ in range(4)])
        self.in2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(2)])
        self.hid2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(4)])

        if use_network == 1:
            self.struct2hid_w = nn.ParameterList([self.__init(lb, ub, hidden_size * 2, hidden_size) for _ in range(1)])
        else:
            self.struct2hid_w = nn.ParameterList([self.__init(lb, ub, hidden_size, hidden_size) for _ in range(1)])
        self.struct2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(1)])

    @staticmethod
    def __init(low, upper, dim1, dim2=None):
        if dim2 is None:
            return nn.Parameter(torch.rand(dim1) * (upper - low) + low )  # 按照官方的初始化方法来初始化网络参数
        else:
            return nn.Parameter(torch.rand(dim1, dim2) * (upper - low) + low)



    def forward(self, hid, user_emb, user_candi, struct_candi, pos_emb):

        r = (torch.sigmoid(torch.mm(torch.cat((user_emb, pos_emb),dim=-1), self.in2hid_w[0]) + self.in2hid_b[0] +
                           torch.mm(hid, self.hid2hid_w[0]) + self.hid2hid_b[0]))
        z = (torch.sigmoid(torch.mm(torch.cat((user_emb, pos_emb),dim=-1), self.in2hid_w[1]) + self.in2hid_b[1] +
                          torch.mm(hid, self.hid2hid_w[1]) + self.hid2hid_b[1]))


        tmp_hidden = torch.mul(user_candi, torch.mm(struct_candi, self.struct2hid_w[0]) + self.struct2hid_b[0])


        n = (torch.tanh((tmp_hidden + (torch.mul(r, (torch.mm(hid, self.hid2hid_w[3]) + self.hid2hid_b[3]))))))
        next_hidden = (torch.mul((1 - z), n) + torch.mul(z, hid))

        #struct_bias =(torch.mm(struct_candi, self.hid2hid_w_s[0]) + self.hid2hid_b_s[0])

        return next_hidden#,struct_bias

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
