import torch
from torch.autograd import Variable
from model.GraphEncoder import GraphEncoder
from model.Merger import *
from model.TransformerBlock import TransformerBlock


def GetPasstime(data_name):
        options = Options(data_name)
        max_time = 0
        min_time = 1000000000000
        with open(options.train_data) as file:
            lines = [line.split("\n")[0].strip() for line in file.readlines()]
            for line in lines:
                times = [item.split(",")[1] for item in line.split()]
                for time in times:
                    max_len = 10
                    if data_name.find("memetracker") != -1: max_len = 12
                    if len(time) != max_len:
                        int_time = int(time.ljust(max_len, '0'))
                        if int_time > max_time:
                            max_time = int_time
                        if int_time < min_time:
                            min_time = int_time
                    else:
                        int_time = int(time)
                        if int_time > max_time:
                            max_time = int_time
                        if int_time < min_time:
                            min_time = int_time

        with open(options.valid_data) as file:
            lines = [line.split("\n")[0].strip() for line in file.readlines()]
            for line in lines:
                times = [item.split(",")[1] for item in line.split()]
                for time in times:
                    max_len = 10
                    if data_name.find("memetracker") != -1: max_len = 12
                    if len(time) != max_len:
                        int_time = int(time.ljust(max_len, '0'))
                        if int_time > max_time:
                            max_time = int_time
                        if int_time < min_time:
                            min_time = int_time
                    else:
                        int_time = int(time)
                        if int_time > max_time:
                            max_time = int_time
                        if int_time < min_time:
                            min_time = int_time

        with open(options.test_data) as file:
            lines = [line.split("\n")[0].strip() for line in file.readlines()]
            for line in lines:
                times = [item.split(",")[1] for item in line.split()]
                for time in times:
                    max_len = 10
                    if data_name.find("memetracker") != -1: max_len = 12
                    if len(time) != max_len:
                        int_time = int(time.ljust(max_len, '0'))
                        if int_time > max_time:
                            max_time = int_time
                        if int_time < min_time:
                            min_time = int_time
                    else:
                        int_time = int(time)
                        if int_time > max_time:
                            max_time = int_time
                        if int_time < min_time:
                            min_time = int_time

        # print(max_time)
        # print(min_time)
        # print(max_time - min_time)
        return max_time-min_time, min_time

# Time embedding selection
class TimeEncoder(nn.Module):
    def __init__(self,opt):
        super(TimeEncoder, self).__init__()
        self.encoder_type = opt.time_encoder_type
        if opt.time_encoder_type == "Interval":
            self.time_encoder= IntervalTimeEncoder(opt)
        # elif opt.time_encoder_type == "Neural":
        #     self.time_encoder= NeuralTimeEncoder(opt)
        elif opt.time_encoder_type == "Linear":
            self.time_encoder= LinearTimeEncoder(opt)
        elif opt.time_encoder_type == "Slide":
            self.time_encoder= SlideTimeEncoder(opt)
        elif opt.time_encoder_type == "None":
            self.time_encoder= NoneTimeEncoder(opt)
        else :
            raise  Exception("Undefined time encoder") 

        self.output_dim = self.time_encoder.output_dim

    def forward(self,input,timestamp,train):
        
        return self.time_encoder(input,timestamp,train)

# Time embedding with time pass encoding (current to last user)
class IntervalTimeEncoder(nn.Module):
    def __init__(self,opt):
        super(IntervalTimeEncoder, self).__init__()

        data_name=opt.data
        self.pass_time, _=GetPasstime(data_name)
        self.n_time_interval = opt.time_interval
        self.per_time = self.pass_time/self.n_time_interval
        self.output_dim=opt.time_dim
        self.linear_1= nn.Linear(self.n_time_interval, self.output_dim, bias=True).cuda()
        init.xavier_normal_(self.linear_1.weight)
        self.relu=nn.ReLU()



    def forward(self,input,timestamp,train):
        batch_size,max_len=input.size()

        t_1=timestamp[:,1:]
        t_2=timestamp[:,:-1]
        pass_time=t_1-t_2
        pass_time=pass_time / self.per_time
        pass_time=self.relu(pass_time.floor_().contiguous().view(batch_size*max_len,1).int()).cuda()

        pass_time=pass_time.long().cuda()

        time_embedding_one_hot=torch.zeros(batch_size*max_len, self.n_time_interval).cuda()
        time_embedding_one_hot=time_embedding_one_hot.scatter_(1, pass_time, 1).cuda()

        time_embedding = self.linear_1(time_embedding_one_hot)

        time_embedding=time_embedding.view(batch_size, max_len, self.output_dim).cuda()

        return time_embedding.cuda(),timestamp[:, :-1]


# Time embedding with Position Embedding
class LinearTimeEncoder(nn.Module):
    def __init__(self,opt):
        super(LinearTimeEncoder, self).__init__()
        data_name="./data/"+opt.data

        # self.n_time_interval = opt.time_interval
        # self.per_time = self.pass_time/self.n_time_interval
        self.output_dim=opt.time_dim
        # self.linear_1= nn.Linear(self.n_time_interval, self.output_dim, bias=True).cuda()
        # init.xavier_normal_(self.linear_1.weight)
        # self.relu=nn.ReLU()

    def forward(self,input,timestamp,train):
        batch_size,max_len=input.size()

        # t_1=timestamp[:,1:]
        # t_2=timestamp[:,:-1]
        # pass_time=t_1-t_2
        # pass_time=pass_time / self.per_time
        # pass_time=self.relu(pass_time.floor_().contiguous().view(batch_size*max_len,1).int())

        # pass_time=pass_time.long()


        # time_embedding_one_hot=time_embedding_one_hot.scatter_(1, pass_time, 1).cuda()

        # time_embedding = self.linear_1(time_embedding_one_hot)

        time_embedding =torch.zeros(batch_size, max_len, self.output_dim).cuda()
        one_time_embedding = self.BuildRelativePositionEmbedding(max_len,self.output_dim)
        time_embedding[:,:,:]=one_time_embedding[:,:]
        # time_embedding=time_embedding.view(
        return time_embedding.cuda(),timestamp[:, :-1]
    
    def BuildRelativePositionEmbedding(self,max_len,d_model):
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

# Time embedding with time pass encoding (current to  first user)
class SlideTimeEncoder(nn.Module):
    def __init__(self,opt):
        super(SlideTimeEncoder, self).__init__()

        data_name=opt.data

        self.pass_time,self.min_time=GetPasstime(data_name)
        self.n_time_interval = 10000
        self.per_time = self.pass_time/self.n_time_interval
        self.output_dim=opt.time_dim
        self.linear_1= nn.Linear(self.n_time_interval, self.output_dim, bias=True).cuda()
        init.xavier_normal_(self.linear_1.weight)
        self.relu=nn.ReLU()


    def forward(self,input,timestamp,train):
        batch_size,max_len=input.size()

        # t_1=timestamp[:,1:]
        # t_2=timestamp[:,:-1]
        # pass_time=t_1-t_2
        timestamp=timestamp[:, :-1]
        pass_time=(timestamp-timestamp[:,0].unsqueeze(1).repeat(1,max_len))/self.per_time
        
        pass_time=self.relu(pass_time.floor_().contiguous().view(batch_size*max_len,1).int()).long()
 
 
        time_embedding_one_hot=torch.zeros(batch_size*max_len, self.n_time_interval)
        time_embedding_one_hot=time_embedding_one_hot.scatter_(1, pass_time, 1).cuda()

        time_embedding = self.linear_1(time_embedding_one_hot)


        time_embedding=time_embedding.view(batch_size, max_len, self.output_dim).cuda()

        return time_embedding.cuda(),timestamp


# Time embedding with all zero
class NoneTimeEncoder(nn.Module):
    def __init__(self,opt):
        super(NoneTimeEncoder, self).__init__()

        data_name="./data/"+opt.data

        # self.pass_time=self.GetPasstime(data_name)
        # self.n_time_interval = 100000
        # self.per_time = self.pass_time/self.n_time_interval
        self.output_dim=opt.time_dim
        # self.linear_1= nn.Linear(self.n_time_interval, self.output_dim, bias=True).cuda()
        # init.xavier_normal_(self.linear_1.weight)
        # self.relu=nn.ReLU()

    def forward(self,input,timestamp,train):
        batch_size,max_len=input.size()

        time_embedding=torch.zeros(batch_size,max_len, self.output_dim).cuda()

        return time_embedding.cuda(),timestamp[:, :-1]


class Decoder(nn.Module):
    def __init__(self, input_size, user_size, opt):
        super(Decoder, self).__init__()
        if opt.norm == True:
            self.decoder = Decoder2L(input_size, user_size, opt.dropout)
        else:
            self.decoder = Decoder1L(input_size, user_size, opt.dropout)

    def forward(self, outputs):
        return self.decoder(outputs)



class Decoder2L(nn.Module):
    def __init__(self, input_size, user_size, dropout=0.1):
        super(Decoder2L, self).__init__()

        self.linear2 = nn.Linear(input_size, input_size * 2)
        self.linear1 = nn.Linear(input_size * 2, user_size)
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, outputs):

        return self.linear1(self.dropout(torch.tanh(self.linear2(outputs))))

class Decoder1L(nn.Module):
    def __init__(self, input_size, user_size, dropout = 0.1):
        super(Decoder1L, self).__init__()

        self.linear1 = nn.Linear(input_size, user_size)
        init.xavier_normal_(self.linear1.weight)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, outputs):
        return self.linear1(self.dropout(outputs))
