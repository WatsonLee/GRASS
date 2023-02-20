


import torch
device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

PAD = 0
UNK = 2
BOS = 3
EOS = 1

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
time_step_split = 5
n_heads = 14
step_len = 5
scale_time_loss = 100
MAX_CAS_LEN = 500

CAS_ID_INIT = 900000000


# data paths for the model 
class Options(object):
    def __init__(self, data_name='twitter'):
        # data options.
        # data_name = 'twitter'
        # train file path.
        self.train_data = '../data/' + data_name + '/cascade.txt'
        # valid file path.
        self.valid_data = '../data/' + data_name + '/cascadevalid.txt'
        # test file path.
        self.test_data = '../data/' + data_name + '/cascadetest.txt'

        # train file path.
        self.train_data_id = '../data/' + data_name + '/cascade_id.txt'
        # valid file path.
        self.valid_data_id = '../data/' + data_name + '/cascadevalid_id.txt'
        # test file path.
        self.test_data_id = '../data/' + data_name + '/cascadetest_id.txt'

        # user dict and array
        self.u2idx_dict = '../data/' + data_name + '/u2idx.pickle'
        self.idx2u_dict = '../data/' + data_name + '/idx2u.pickle'

        # user and item dict and array 
        self.ui2idx_dict = '../data/' + data_name + '/ui2idx.pickle'
        self.idx2ui_dict = '../data/' + data_name + '/idx2ui.pickle'

        # time dict and array (useless)
        self.t2idx_dict = '../data/' + data_name+ '/t2idx.pickle'
        self.idx2t_dict = '../data/' + data_name+ '/idx2t.pickle'


        # save path.
        self.save_path = ''

        # social network file 
        self.net_data = '../data/' + data_name + '/edges.txt'
        self.net_data_refined = '../data/' + data_name + '/edges_refined.txt'

        # enhancement edge file 
        self.item_net_data = '../data/' + data_name + "/edges_item.txt"

        # diffusion net file 
        self.repost_net_data = '../data/' + data_name + "/edges_reposts.txt"
        self.repost_time_data = '../data/' + data_name + "/edges_reposts_time.txt"  # timestamp for each diffusion edge

        # For model with edge time embedding
        self.item_net_data_with_time_weight = '../data/' + data_name + "/edges_item_weights.txt"
        self.repost_net_data_with_time_weight = '../data/' + data_name + "/edges_reposts_weights.txt"

        # 1. For model with user drop 
        # 2. For model with user drop + edge time embedding
        self.item_net_data_trimmed = '../data/' + data_name + "/edges_item_trimmed.txt"
        self.item_net_data_with_time_weight_trimmed = '../data/' + data_name + "/edges_item_trimmed_weights.txt"

        self.repost_net_data_trimmed = '../data/' + data_name + "/edges_reposts_trimmed.txt"
        self.repost_net_data_with_time_weight_trimmed = '../data/' + data_name + "/edges_reposts_trimmed_weights.txt"
        
        # item inner network file 
        self.item_inner_net_data='../data/' + data_name+"/edges_item_inner.txt"
        self.item_inner_net_data_refined='../data/' + data_name+"/edges_item_inner_refined.txt"

        # 用户之间的转发图
        self.user_repost_file = '../data/' + data_name+"/user_repost_list.pickle"
        
        # 用户之间的隐式连接 
        self.user_hidden_net = '../data/' + data_name+'/edges_user_hidden.txt'

        # 
        self.batch_size = 16
        self.embed_dim = 64





# redirect the stdout to filename
class Logger(object):
    def __init__(self, filename="Default.log"):
        import sys,os
        self.terminal = sys.stdout
        if os.path.exists("./log") is False: 
            os.mkdir("./log")
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass