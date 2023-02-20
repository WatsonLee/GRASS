from utils.Constants import Logger, Options
from model.GRASS import GRASS
from utils.Optim import ScheduledOptim
from utils.Metrics import Metrics
from utils.DataConstruct import DataConstruct
import utils.Constants as Constants
import torch.nn as nn
import torch
import argparse
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm


np.set_printoptions(threshold=np.inf)


def best_result(scores, k_list=[10,50,100], weights=[1,1,1,1,1,1]):
    result = 0.0
    i=0
    for k in k_list:
        #s = scores["hits@"+str(k)]
        #w = weights[i]
        result += weights[i]*scores["hits@"+str(k)]
        result += weights[i+1] * scores["map@" + str(k)]
        i+=2
    return result


def SeedEverything():
    SEED = opt.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)
    print(f"Init_every_seed with {SEED}")


root_path = './'
parser = argparse.ArgumentParser()
## Training Hypers
parser.add_argument('--epoch', type=int, default=70)
parser.add_argument("--data", type=str, default="weibo23", metavar='dataname', help='dataset name')
parser.add_argument('--batch_size', type=int, default=16,metavar='BS', help='batch size')
parser.add_argument('--dropout', type=float, default=0.15, metavar='dropout',help='dropout rate')
parser.add_argument('--smooth', type=float, default=0.1, help='Lable Smooth rate for the model')
parser.add_argument('--seed', type=int, default=2023, help='random state seed')
parser.add_argument('--warmup', type=int, default=10)  # warmup epochs
parser.add_argument('--n_warmup_steps', type=int, default=1000, metavar='LR',
                    help='the warmup steps in the model')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2',
                    help='L2 regularization weight')
parser.add_argument('--norm', type=bool, default=True, metavar='Norm',
                    help='Need norm')

## Module Hypers
# time representation
parser.add_argument('--time_encoder_type', type=str, default="Interval",
                    choices=['Interval', 'Neural', 'Slide', 'None'],
                    help='The decoder of the model') # time Encoder

parser.add_argument('--time_encoder_type_gru', type=str, default="Interval",
                    choices=['Interval', 'Neural', 'Slide', 'None'],
                    help='The decoder of the model') # time Encoder

## Model Hypers 
parser.add_argument('--d_model', type=int, default=64, metavar='inputF',
                    help='dimension of initial features.')
parser.add_argument('--time_dim', type=int, default=8, metavar='time', help='The dim of the time encoder')
parser.add_argument('--pos_dim', type=int, default=8, metavar='pos', help='The dim of the positional embedding')
parser.add_argument('--heads', type=int, default=1,help='number of heads in transformer')
parser.add_argument('--time_interval', type=int, default=10000,
                    help='the time interval for each time slice')
parser.add_argument('--graph_type', default="social+diffusion+item",
                    choices=['social', 'diffusion', 'item', 'social+diffusion', 'social+item', 'diffusion+item', 'social+diffusion+item'],
                    help='set the edges in the heterogenerous graph type.')
# parser.add_argument('--graph_type', default="diffusion+item", help='set the edges in the heterogenerous graph type.')

# Logging Hypers
parser.add_argument('--save_path', default=None)
parser.add_argument('--save_mode', type=str,
                    choices=['all', 'best'], default='best')
parser.add_argument('--notes', default="",help='lets take some notes for the model.')
parser.add_argument('--export_log', type=bool, default=True,help='copy the log to a certain file.')


opt = parser.parse_args()
opt.d_word_vec = opt.d_model
opt.transformer_dim = opt.d_model + opt.time_dim + opt.pos_dim
opt.notes = "GRASS"
if opt.save_path is None:
    opt.save_path = root_path + f"checkpoints/GRASS_{opt.data}_{int(time.time())}.pt"
opt.save_path = root_path + f"checkpoints/GRASS_weibo23.pt"
print(opt)
SeedEverything()



metric = Metrics()
data_path = opt.data

def get_performance(crit, pred, gold):
    ''' Apply label smoothing 
    if needed '''
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    # print ("get performance, ", gold.data, pred.data)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()

    true_set = set()
    for items in gold.cpu().numpy().tolist():
        true_set.add(items)
    pre_set = set()
    for item in pred.cpu().numpy().tolist():
        if item in true_set:
            pre_set.add(item)

    return loss, n_correct, len(pre_set), len(true_set)



def train_epoch(model, training_data, loss_func, optimizer,epoch):
    ''' Epoch operation in training phase'''
    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    total_same_user = 0.0
    n_total_uniq_user = 0.0
    batch_num = 0.0

    #for i, batch in tqdm(
    #        enumerate(training_data), mininterval=2,
    #        desc='  - (Training)   ', leave=False):

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
    #for i, batch in enumerate(
            #training_data):  # tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # prepare data
        tgt, tgt_timestamp, tgt_id = batch
        tgt.cuda()
        tgt_timestamp.cuda()
        user_gold = tgt[:, 1:].cuda()
        time_gold = tgt_timestamp[:, 1:].cuda()

        # start_time = time.time()

        n_words = user_gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words
        batch_num += tgt.size(0)

        optimizer.zero_grad()

        user_pred = model(tgt, tgt_timestamp, tgt_id,train=True)

        # backward
        loss, n_correct, same_user, input_users = get_performance(loss_func, user_pred, user_gold)
        loss.backward(retain_graph=True)

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate(epoch)

        # note keeping
        n_total_correct += n_correct
        total_loss = total_loss + loss.item()

        total_same_user += same_user
        n_total_uniq_user += input_users

        #print("Training batch ", i, " loss: ", loss.item(), " acc:", (n_correct.item() / len(user_pred)), f"\t\toutput_users:{(same_user)}/{(input_users)}={same_user / input_users}", )

    return total_loss / n_total_words, n_total_correct / n_total_words, total_same_user / n_total_uniq_user


def test_epoch(model, validation_data, k_list=[10, 50, 100]):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    for batch in tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
        #print("Validation batch ", i)
        # prepare data
        # print(batch)

        tgt, tgt_timestamp, tgt_id = batch
        tgt.cuda()
        tgt_timestamp.cuda()
        user_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

        user_pred = model(tgt, tgt_timestamp, tgt_id,train=False)

        user_pred = user_pred.detach().cpu().numpy()

        scores_batch, scores_len,MRR = metric.compute_metric(user_pred, user_gold, k_list)

        n_total_words += scores_len
        for k in k_list:
            scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
            scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    return scores, MRR


def train_model(data_path):
    # ========= Preparing DataLoader =========#

    train_data = DataConstruct(data_path, data=0, load_dict=True, batch_size=opt.batch_size, cuda=False, seed=opt.seed)
    valid_data = DataConstruct(data_path, data=1, batch_size=opt.batch_size, cuda=False, seed=opt.seed)  # torch.cuda.is_available()
    test_data = DataConstruct(data_path, data=2, batch_size=opt.batch_size, cuda=False, seed=opt.seed)

    opt.user_size = train_data.user_size
    opt.ntoken = train_data.ntoken

    # ========= Preparing Model =========#
    opt.data_path = data_path
    opt.norm = train_data.need_norm
    model = GRASS(opt)

    #print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

   
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizerAdam = torch.optim.Adam(params, betas=(
        0.9, 0.98), eps=1e-09, weight_decay=opt.l2)  # weight_decay is l2 regularization
    optimizer = ScheduledOptim(
        optimizerAdam, opt.d_model, opt.n_warmup_steps, data_path)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizerAdam, 'min', factor=0.4, patience=7, verbose=True)

    loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)

    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()

    validation_history = 0.0
    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_accu, train_pred = train_epoch(model, train_data, loss_func, optimizer,epoch_i)
        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, predected:{pred:3.3f} %' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu, pred=100 * train_pred,
            elapse=(time.time() - start) / 60), flush=True)

        if epoch_i >= 0:
            start = time.time()
            scores,MRR = test_epoch(model, valid_data)
            print('  - ( Validation )) ')
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            print("Validation use time: ", (time.time() - start) / 60, "min")
            print(f"MRR: {MRR}")

            print('  - (Test) ')
            scores,MRR = test_epoch(model, test_data)
            for metric in scores.keys():
                print(metric + ' ' + str(scores[metric]))
            if validation_history <= best_result(scores):
                print("Best Validation at Epoch:{}".format(epoch_i))
                validation_history = best_result(scores)
                #print("Save best model!!!")
                torch.save(model.state_dict(), opt.save_path)
                print(f"MRR: {MRR}")
            
        scheduler.step(validation_history)

def test_model(data_path):
    train_data = DataConstruct(data_path, data=0, load_dict=True, batch_size=opt.batch_size, cuda=False)
    test_data = DataConstruct(data_path, data=2, batch_size=opt.batch_size, cuda=torch.cuda.is_available())
    opt.user_size = train_data.user_size
    opt.ntoken = train_data.ntoken
    opt.data_path = data_path
    opt.norm = train_data.need_norm

    model = GRASS(opt)
    model.load_state_dict(torch.load(opt.save_path))
    model.cuda()

    scores,MRR= test_epoch(model, test_data)
    print('  - (Test) ')
    for metric in scores.keys():
        print(metric + ' ' + str(scores[metric]))
    print(f"MRR: {MRR}")

if __name__ == "__main__":

    #train_model(data_path)
    test_model(data_path)