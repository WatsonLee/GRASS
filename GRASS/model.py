import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from GRAU import GRAUCell_withSS
from GraphConv import GraphNN
import Constants
from TransformerBlock import TransformerBlock


def get_previous_user_mask(seq, user_size):
    ''' Mask previous activated users.'''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1,1,seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    #print(previous_mask)
    #print(seqs)
    masked_seq = previous_mask * seqs.data.float()
    #print(masked_seq.size())

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq,PAD_tmp],dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2,masked_seq.long(),float('-inf'))
    
    return masked_seq




class GRASS(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, opt, dropout=0.1, tie_weights=False):
        super(GRASS, self).__init__()
        ntoken = opt.user_size
        ninp = opt.d_word_vec
        nhid = opt.d_inner_hid
        # network part
        self.use_network = opt.network
        self.dropout = opt.dropout

        self.gnn_cas_emb_layer = GraphNN(ntoken, nhid, dropout=self.dropout).cuda()

        if opt.network:
            self.gnn_topo_emb_layer = GraphNN(ntoken, nhid, dropout=self.dropout).cuda()




        self.encoder = nn.Embedding(ntoken, ninp).cuda()
        self.pos_emb = opt.pos_emb

        self.pos_dim = 8
        self.pos_embedding = nn.Embedding(1000, self.pos_dim).cuda()

        self.user_hidden_encoder = TransformerBlock(d_hidden=ninp*1 + self.pos_dim*0, d_q = (ninp*1 + self.pos_dim),
                                                   d_k = (ninp*1 + self.pos_dim), d_v = (ninp*1 + self.pos_dim),
                                                    n_heads=opt.head_num, is_layer_norm=1).cuda()

        if self.use_network:
            self.rnn = GRAUCell_withSS(ninp * 1 + self.pos_dim * 1, nhid * 1 + self.pos_dim * 0, use_network=1).cuda()
            self.struct_hidden_encoder = TransformerBlock(d_hidden=ninp * 2 + self.pos_dim * 0,
                                                        d_q=(ninp * 2 + self.pos_dim),
                                                        d_k=(ninp * 2 + self.pos_dim), d_v=(ninp * 2 + self.pos_dim),
                                                        n_heads=opt.head_num, is_layer_norm=1).cuda()
            self.decoder = nn.Linear(nhid * 3 + self.pos_dim*0, ntoken).cuda()  #
        else:
            self.rnn = GRAUCell_withSS(ninp * 1 + self.pos_dim * 1, nhid * 1 + self.pos_dim * 0, use_network=0).cuda()
            self.struct_hidden_encoder = TransformerBlock(d_hidden=ninp * 1 + self.pos_dim * 0,
                                                        d_q=(ninp * 1 + self.pos_dim),
                                                        d_k=(ninp * 1 + self.pos_dim), d_v=(ninp * 1 + self.pos_dim),
                                                        n_heads=opt.head_num, is_layer_norm=1).cuda()
            self.decoder = nn.Linear(nhid * 2 + self.pos_dim*0, ntoken).cuda()  #



        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.user_size = ntoken


    def init_weights(self):

        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.pos_embedding.weight.data.uniform_(-initrange, initrange)



    def forward(self, input, cas_graph, topo_graph, generate=False):
        if not generate:
            input = input[:, :-1]
        # network part
        input_mask = (input==Constants.PAD).cuda()
        #input_mask = torch.cat([torch.ones(input.size(0)).bool().cuda().unsqueeze(1), input_mask], dim=1)

        batch_size = input.size(0)
        max_len = input.size(1)

        pos_emb_vec = F.dropout(self.pos_embedding(
            torch.Tensor([i for i in range(max_len)]).unsqueeze(0).expand(batch_size, -1).long().cuda()), p=self.dropout, training=self.training)

        # embedding
        emb = F.dropout(self.encoder(input), p=self.dropout, training=self.training)

        cas_emb = self.gnn_cas_emb_layer(cas_graph)[input.reshape(-1), :].view(input.size(0), input.size(1), self.nhid).cuda()


        if self.use_network:
            net_emb = self.gnn_topo_emb_layer(topo_graph)[input.reshape(-1),:].view(input.size(0), input.size(1),self.nhid).cuda()
            outputs = Variable(torch.zeros(max_len, batch_size, self.nhid * 3 + self.pos_dim*0)).cuda()
            struct_hidden = self.struct_hidden_encoder(torch.cat((cas_emb, net_emb, pos_emb_vec), dim=-1),
                                                   torch.cat((cas_emb, net_emb, pos_emb_vec), dim=-1),
                                                   torch.cat((cas_emb, net_emb, pos_emb_vec), dim=-1), mask=input_mask)
        else:
            outputs = Variable(torch.zeros(max_len, batch_size, self.nhid * 2 + self.pos_dim*0)).cuda()
            struct_hidden = self.struct_hidden_encoder(torch.cat((cas_emb, pos_emb_vec), dim=-1),
                                                   torch.cat((cas_emb, pos_emb_vec), dim=-1),
                                                   torch.cat((cas_emb, pos_emb_vec), dim=-1), mask=input_mask)

        user_hidden = self.user_hidden_encoder(torch.cat((emb, pos_emb_vec), dim=-1),
                                                       torch.cat((emb, pos_emb_vec), dim=-1),
                                                       torch.cat((emb, pos_emb_vec), dim=-1), mask=input_mask)


        hidden = Variable(torch.zeros(batch_size, (self.nhid*1+self.pos_dim*0))).cuda()





        for t in range(0, max_len):
            # GRU

            hidden = self.rnn(hidden, emb[:, t, :], user_hidden[:, t, :], struct_hidden[:,t,:], pos_emb_vec[:, t, :])

            if self.use_network:
                outputs[t] = torch.cat((hidden, (torch.cat((cas_emb[:,t,:], net_emb[:,t,:]), dim=-1) + struct_hidden[:,t,:])/2), dim=1)
            else:
                outputs[t] = torch.cat((hidden, (cas_emb[:,t,:] + struct_hidden[:,t,:])/2), dim=1)


            
        outputs = outputs.transpose(0,1).contiguous()#b*l*v
        outputs = F.dropout(outputs, p=self.dropout, training=self.training)

        decoded = self.decoder(outputs.view(outputs.size(0)*outputs.size(1), outputs.size(2)))
        result = decoded.view(outputs.size(0), outputs.size(1), decoded.size(1)) + torch.autograd.Variable(get_previous_user_mask(input, self.user_size),requires_grad=False)

        
        return result.view(-1,decoded.size(1)), hidden
