
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import copy
import math

def MultileayerModule(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerBlock(nn.Module):

    def __init__(self, d_hidden = 72, d_q = 72, d_k=72, d_v=72, n_heads=8, is_layer_norm=3, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_hidden = d_hidden

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=d_hidden)

        # self.pos_encoding = PositionalEncoding(d_model=input_size, dropout=0.5)
        self.W_q = nn.Parameter(torch.Tensor(d_q, n_heads * d_hidden))
        self.W_k = nn.Parameter(torch.Tensor(d_k, n_heads * d_hidden))
        self.W_v = nn.Parameter(torch.Tensor(d_v, n_heads * d_hidden))

        self.W_o = nn.Parameter(torch.Tensor(d_hidden*n_heads, d_hidden))
        self.linear1 = nn.Linear(d_hidden, d_hidden).cuda()
        self.linear2 = nn.Linear(d_hidden, d_hidden).cuda()

        self.dropout = attn_dropout
        self.__init_weights__()
        #print(self)

    def __init_weights__(self):
        init.xavier_uniform_(self.W_q)
        init.xavier_uniform_(self.W_k)
        init.xavier_uniform_(self.W_v)
        init.xavier_uniform_(self.W_o)

        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output

    def scaled_dot_product_attention(self, Q, K, V, mask, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, n_heads, input_size)
        :param K: (*, max_k_words, n_heads, input_size)
        :param V: (*, max_v_words, n_heads, input_size)
        :param mask: (*, max_q_words)
        :param episilon:
        :return:
        '''
        temperature = self.d_hidden ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
            mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().cuda()
            mask_ = mask + pad_mask
            Q_K = Q_K.masked_fill(mask_, -2**32+1)

        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_q_words, max_k_words)
        Q_K_score = F.dropout(Q_K_score, p=self.dropout, training=self.training)

        V_att = Q_K_score.bmm(V)  # (*, max_q_words, input_size)
        return V_att


    def multi_head_attention(self, Q, K, V, mask):
        '''
        :param Q:
        :param K:
        :param V:
        :param mask: (bsz, max_q_words)
        :return:
        '''
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_hidden)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_hidden)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_hidden)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_hidden)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_hidden)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_hidden)

        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)  # For head axis broadcasting.
            mask = mask.reshape(-1, mask.size(-1))

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_hidden)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_hidden)

        output = F.dropout(V_att.matmul(self.W_o), p=self.dropout, training=self.training) # (batch_size, max_q_words, input_size)
        return output


    def forward(self, Q, K, V, mask=None):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        V_att = self.multi_head_attention(Q, K, V, mask)

        if self.is_layer_norm == 1:
            X = self.layer_morm(Q[:,:,:self.d_hidden] + V_att)  # (batch_size, max_r_words, embedding_dim)
            return self.layer_morm(self.FFN(X) + X)
        elif self.is_layer_norm ==2:
            X = Q[:,:,:self.d_hidden] + V_att
            return self.FFN(X) + X
        else:
            return self.layer_morm(V_att)
