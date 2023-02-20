from torch.nn.init import zeros_
import utils.Constants as Constants
import torch
import numpy as np
from torch.autograd import Variable

from model.GraphEncoder import GraphEncoder
from model.GRAU import GRAU
from model.TimeEncoder import *
from model.Merger import *
from model.TransformerBlock import TransformerBlock


class GRASS(nn.Module):
    def __init__(self,opt):
        super(GRASS, self).__init__()

        # hypers 
        self.opt=opt
        dropout=opt.dropout
        self.ntoken = opt.ntoken
        self.user_size = opt.user_size
        self.ninp = opt.d_word_vec
        self.transformer_dim = opt.transformer_dim  # 最终的维度
        self.pos_dim = opt.pos_dim
        self.__name__ = "GRU"

        # module control 
        self.graph_type = opt.graph_type
        self.time_encoder_type = opt.time_encoder_type

        # dropout module
        self.dropout = nn.Dropout(dropout)
        #self.drop_timestamp = nn.Dropout(dropout)

        # modules 
        self.user_encoder = GraphEncoder(opt,self.graph_type)
        self.time_encoder = TimeEncoder(opt)
        self.pos_embedding = nn.Embedding(1000, self.pos_dim)
        #self.decoder = TransformerBlock(input_size=opt.transformer_dim, n_heads=8)

        self.user_hidden_fusion = TransformerBlock(d_q=opt.transformer_dim + self.ninp, d_k=opt.transformer_dim + self.ninp, d_v =opt.transformer_dim, n_heads=1)



        self.cas_encoder = GRAU(opt.transformer_dim, opt.transformer_dim, opt=opt)

        self.decoder = Decoder(input_size=opt.transformer_dim, user_size=self.user_size, opt=opt)


        self.init_weights()
        #print(self)

    def init_weights(self):
        init.xavier_normal_(self.pos_embedding.weight)


    def forward(self, input, input_timestamp, input_id,train=True):
        input = input[:, :-1]  # [bsz,max_len]   input_id:[batch_size]

        # graph encoder
        user_embedding, hyper_embedding = self.user_encoder(input, input_timestamp,train)

        #hyper_embedding = self.fusion_gate(user_embedding, hyper_embedding)


        user_embedding = self.dropout(user_embedding)
        hyper_embedding = self.dropout(hyper_embedding)




        # timeencoder
        time_embedding,input_timestamp = self.time_encoder(input, input_timestamp,train)
        # item_embedding=self.item_encoder(input,input_timestamp,train)

        # time difference embedding
        #user_embedding=torch.cat([user_embedding,time_embedding],dim=-1)

        # positional embedding
        mask = (input == Constants.PAD).cuda()
        batch_t = torch.arange(input.size(1)).expand(input.size()).cuda()
        order_embed = self.dropout(self.pos_embedding(batch_t))
        final_input = torch.cat([user_embedding,time_embedding, order_embed], dim=-1).cuda()  # dynamic_node_emb
        #hyper_input = torch.cat([hyper_embedding, time_embedding, order_embed], dim=-1).cuda()  # dynamic_node_emb

        hyper_user_input = torch.cat([user_embedding, time_embedding, order_embed, hyper_embedding], dim=-1).cuda()

        batch_size = input.size(0)
        max_len = input.size(1)

        outputs = Variable(torch.zeros(max_len, batch_size, self.transformer_dim)).cuda()
        #outputs2 = Variable(torch.zeros(max_len, batch_size, self.transformer_dim)).cuda()
        hidden = Variable(torch.zeros(batch_size, self.transformer_dim)).cuda()

        #struct_hidden = (self.struct_hidden_fusion(final_input, final_input, hyper_input, mask=mask))

        user_hidden = (self.user_hidden_fusion(hyper_user_input, hyper_user_input, final_input, mask=mask))


        #user_hidden = torch.cat([torch.zeros(batch_size, self.transformer_dim).unsqueeze(1), user_hidden], dim=-1)


        for t in range(0, max_len):
            hidden = self.cas_encoder(hidden, final_input[:,t,:], user_hidden[:,t,:])

            outputs[t] = hidden


        outputs = outputs.transpose(0,1).contiguous()


        # Output prediction
        pred = self.decoder(outputs)  # (bsz, max_len, |U|)
        mask = self.get_previous_user_mask(input.cuda(), self.user_size)
        output = pred.cuda() + mask.cuda()
        user_predction = output.view(-1, output.size(-1))  # (bsz*max_len, |U|)

        return user_predction



    def get_previous_user_mask(self, seq, user_size):
        ''' Mask previous activated users.'''
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask)
        if seq.is_cuda:
            previous_mask = previous_mask.cuda()

        masked_seq = previous_mask * seqs.data.float()
        # print(masked_seq.size())

        # force the 0th dimension (PAD) to be masked
        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
        if seq.is_cuda:
            PAD_tmp = PAD_tmp.cuda()
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
        if seq.is_cuda:
            ans_tmp = ans_tmp.cuda()
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
        masked_seq = Variable(masked_seq, requires_grad=False)
        return masked_seq




