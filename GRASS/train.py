'''
This script handling the training process.
'''

import argparse
import math
import time

import MyDataLoader
import metrics
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import Constants
from model import GRASS
from Optim import ScheduledOptim
from MyDataLoader import DataLoader
import numpy as np

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

def best_result(scores, k_list=[1,5,10,20,50,100], weights=[0,0,0,0,1,1,0,0,1,1,1,1]):
    result = 0.0
    i=0
    for k in k_list:
        #s = scores["hits@"+str(k)]
        #w = weights[i]
        result += weights[i]*scores["hits@"+str(k)]
        result += weights[i+1] * scores["map@" + str(k)]
        i+=2
    return result




def get_performance(crit, pred, gold, smoothing=False, num_class=None):
    ''' Apply label smoothing if needed '''

    # TODO: Add smoothing
    if smoothing:
        assert bool(num_class)
        eps = 0.1
        gold = gold * (1 - eps) + (1 - gold) * eps / num_class
        raise NotImplementedError

    loss = crit(pred, gold.contiguous().view(-1))


    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()

    return loss, n_correct



def train_epoch(model, training_data, cas_graph, topo_graph, crit, optimizer,):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0

    batch_num = 0.0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        tgt = batch
        gold = tgt[:, 1:]
        n_words = gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words

        batch_num += tgt.size(0)

        # forward
        optimizer.zero_grad()
        pred, *_ = model(tgt, cas_graph, topo_graph)

        # backward
        loss, n_correct = get_performance(crit, pred, gold)



        loss.backward()


        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()

        # note keeping
        n_total_correct += n_correct
        total_loss += loss.data.item()

    return total_loss/n_total_words, n_total_correct/n_total_words

def eval_epoch(model, validation_data, cas_graph, topo_graph, crit):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0


    for batch in tqdm(
            validation_data, mininterval=2,
            desc='  - (Validation) ', leave=False):

        # prepare data
        tgt = batch
        gold = tgt[:, 1:]

        # forward
        pred, *_ = model(tgt, cas_graph, topo_graph)
        loss, n_correct = get_performance(crit, pred, gold)

        # note keeping
        n_words = gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.data.item()

    return total_loss/n_total_words, n_total_correct/n_total_words

def test_epoch(model, test_data, cas_graph, topo_graph, k_list=[1,5,10,20,50,100]):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0
    n_total_words = 0

    for batch in tqdm(
            test_data, mininterval=2,
            desc='  - (Test) ', leave=False):

        # prepare data
        tgt = batch
        gold = tgt[:, 1:]

        # forward
        pred, *_ = model(tgt, cas_graph, topo_graph)
        scores_batch, scores_len = metrics.portfolio(pred.detach().cpu().numpy(), gold.contiguous().view(-1).detach().cpu().numpy(), k_list)
        n_total_words += scores_len
        for k in k_list:
            scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
            scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len



    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words
    return scores

def train(model, training_data, validation_data, test_data, cas_graph, topo_graph, crit, optimizer, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')


    expect = 0
    validation_history = 0.0
    for epoch_i in range(opt.epoch):
        print("GPU Memory Allocated: %d" %(torch.cuda.memory_allocated(device=None)))
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, cas_graph, topo_graph, crit, optimizer)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))
        # validation
        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, cas_graph, topo_graph, crit)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        #valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}


        scores = test_epoch(model, test_data, cas_graph, topo_graph)
        print('  - (Test) ')
        for metric in scores.keys():
            print(metric+' '+str(scores[metric]), flush=True)

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                valid_now = best_result(scores)
                if valid_now >= validation_history:
                    validation_history = valid_now
                    torch.save(checkpoint, model_name)
                    print('    - [Info-BEST] The checkpoint file has been updated.')



        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    #torch.set_num_threads(4)
    ''' Main function'''
    parser = argparse.ArgumentParser()

    #parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=70)
    parser.add_argument('-batch_size', type=int, default=16) #16

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=64) #64
    parser.add_argument('-d_inner_hid', type=int, default=64) #64

    parser.add_argument('-n_warmup_steps', type=int, default=1000)

    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default="None")
    parser.add_argument('-save_model', default='model')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-head_num', type=int, default=1)

    parser.add_argument('-network', type=int, default=1)
    parser.add_argument('-pos_emb', type=int, default=1)
    parser.add_argument('-warmup', type=int, default=10) # warmup epochs
    parser.add_argument('-notes', default='')
    parser.add_argument('-data_name', default='douban')


    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    if opt.network==1:
        opt.network = True
    else:
        opt.network = False
    if opt.pos_emb==1:
        opt.pos_emb = True
    else:
        opt.pos_emb = False
    print(opt.notes)
    

    #========= Preparing DataLoader =========#
    train_data = DataLoader(opt.data_name, data=0, load_dict=True, batch_size=opt.batch_size, cuda=opt.cuda, loadNE=opt.network)
    valid_data = DataLoader(opt.data_name, data=1, batch_size=opt.batch_size, cuda=opt.cuda, loadNE=False)
    test_data = DataLoader(opt.data_name, data=2, batch_size=opt.batch_size, cuda=opt.cuda, loadNE=False)

    opt.user_size = train_data.user_size

    CascadePredictor = GRASS(opt)


    optimizer = ScheduledOptim(
        optim.Adam(
            CascadePredictor.parameters(),
            lr= 0.001,
            betas=(0.9, 0.999), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)


    def get_criterion(user_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(user_size)
        weight[Constants.PAD] = 0
        weight[Constants.EOS] = 1
        return nn.CrossEntropyLoss(weight, size_average=False, ignore_index=Constants.PAD)

    crit = get_criterion(train_data.user_size).cuda()

    if opt.network:
        cas_graph = MyDataLoader.LoadHeteGraph(opt.data_name).cuda()
        topo_graph = MyDataLoader.LoadRelationGraph(opt.data_name).cuda()
    else:
        cas_graph = MyDataLoader.LoadSocialGraph(opt.data_name).cuda()
        topo_graph = None

    train(CascadePredictor, train_data, valid_data, test_data, cas_graph, topo_graph, crit, optimizer, opt)


if __name__ == '__main__':
    main()

