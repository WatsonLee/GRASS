

import os
import pickle

import torch.nn as nn
from tqdm import tqdm, trange
from torch_geometric.data import Data
from utils.Constants import *



def BuildRepostGraph(data_name):
    options = Options(data_name)
    _u2idx = {}
    _idx2u = []

    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)

    train_data = open(options.train_data, "r");
    lines = train_data.readlines()

    with open(options.repost_net_data, "a") as file:
        for i in range(0, len(lines)):
            items = lines[i].split()
            for i in range(0, len(items) - 2):
                user1, _ = items[i].split(",")
                user2, _ = items[i + 1].split(",")
                file.write(f"{user1},{user2}\n")

    train_data.close()



def BuildItemGraph(data_name):
    options = Options(data_name)
    _ui2idx = {}
    _idx2ui = []

    with open(options.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    with open(options.idx2ui_dict, 'rb') as handle:
        _idx2ui = pickle.load(handle)

    train_data = open(options.train_data, "r");
    lines = train_data.readlines()
    train_data_id = open(options.train_data_id, "r");
    ids = [line.split("\n")[0] for line in train_data_id.readlines()]

    with open(options.item_net_data, "a") as file:
        for i in range(0, len(lines)):
            items = lines[i].split()
            for item in items:
                if item is not "\n":
                    user, _ = item.split(",")
                    file.write(f"{user},{ids[i]}\n")

    train_data.close()
    train_data_id.close()



def BuildSocialGraph(data_name):
    options = Options(data_name)

    import pickle, os
    import numpy as np
    with open(options.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    with open(options.idx2ui_dict, 'rb') as handle:
        _idx2ui = pickle.load(handle)
    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        _idx2u = pickle.load(handle)

    user_lists = [[0 for _ in range(len(_idx2ui))] for _ in range(len(_idx2ui))]
    with open(options.train_data) as cas_file:
        with open(options.train_data_id) as id_file:
            id_lines = [line.split("\n")[0] for line in id_file.readlines()]
            cas_lines = [line.split("\n")[0] for line in cas_file.readlines()]
            for i in range(len(cas_lines)):
                line = cas_lines[i]
                id = id_lines[i]
                users = [item.split(",")[0] for item in line.split()]
                for user in users:
                    user_lists[_ui2idx[user]][_ui2idx[id]] = 1
                    user_lists[_ui2idx[id]][_ui2idx[user]] = 1

    print("start counting")
    a = torch.tensor(user_lists, dtype=torch.float32).cuda()
    res = torch.matmul(a, a)
    res_list = res.detach().cpu().numpy().tolist()
    print("finish counting")

    total = 0
    if data_name == "./data/memetracker":
        if os.path.exists(options.net_data): os.remove(options.net_data)
        with open(options.net_data, "a") as file:
            for i in range(2, len(_idx2u) - 1):
                for j in range(i + 1, len(_idx2u)):
                    if res_list[i][j] > 10:
                        if int(_idx2u[i]) < CAS_ID_INIT:
                            total += 2
                            file.write(f"{_idx2u[i]},{_idx2u[j]}\n")
                            file.write(f"{_idx2u[j]},{_idx2u[i]}\n")

    print(total / 2)


def RefineSocialNetwork(data_name):
    options = Options(data_name)
    _ui2idx = {}
    _idx2ui = []

    social_user_dict = {}
    with open(options.net_data, "r") as file:
        for line in file.readlines():
            user_1, user_2 = line.split("\n")[0].split(",")
            if user_1 not in social_user_dict.keys():
                social_user_dict[user_1] = []

            social_user_dict[user_1].append(user_2)
            # print(social_user_dict[user_1])
    # print(len(social_user_pair))
    # print(social_user_pair)

    cas_user_dict = {}
    with open(options.train_data, "r") as file:
        for line in tqdm(file.readlines()):
            items = line.split("\n")[0].split(" ")
            user = [item.split(",")[0] for item in items]
            for i in range(0, len(user) - 1):
                if user[i] not in cas_user_dict.keys():
                    cas_user_dict[user[i]] = []
                cas_user_dict[user[i]] += user[i:]

    output_user_dict = {}
    for user in cas_user_dict.keys():
        output_user_dict[user] = []
        for u in cas_user_dict[user]:
            if user in social_user_dict.keys():
                if u in social_user_dict[user] and u not in output_user_dict[user]:
                    output_user_dict[user].append(u)

    if os.path.exists(options.net_data_refined): os.remove(options.net_data_refined)
    with open(options.net_data_refined, "a") as file:
        for user_1 in output_user_dict.keys():
            for user_2 in output_user_dict[user_1]:
                file.write(f"{user_1},{user_2}\n")


def BuildTrimedGraphs(data_name="./data/twitter", item_threshold=3, threshold=3):
    options = Options(data_name)

    train_user_dict = {}
    with open(options.train_data, "r") as file:
        for line in file.readlines():
            for item in line.split(" "):
                if item != "\n":
                    user, time = item.split(',')

                    if user not in train_user_dict.keys():
                        train_user_dict[user] = 1
                    else:
                        train_user_dict[user] += 1

    if os.path.exists(options.item_net_data_trimmed): os.remove(options.item_net_data_trimmed)
    with open(options.item_net_data, "r") as input_edge_file:
        with open(options.item_net_data_trimmed, "a") as output_edge_file:
            edge_lines = input_edge_file.readlines()
            for i in range(0, len(edge_lines)):
                line = edge_lines[i]

                u1, u2 = line.split("\n")[0].split(",")
                if train_user_dict[u1] > item_threshold:
                    output_edge_file.write(edge_lines[i])

    if os.path.exists(options.repost_net_data_trimmed): os.remove(options.repost_net_data_trimmed)
    with open(options.repost_net_data, "r") as input_edge_file:
        with open(options.repost_net_data_trimmed, "a") as output_edge_file:
            edge_lines = input_edge_file.readlines()
            for i in range(0, len(edge_lines)):
                line = edge_lines[i]
                u1, u2 = line.split("\n")[0].split(",")
                if train_user_dict[u1] > threshold and train_user_dict[u2] > threshold:
                    output_edge_file.write(edge_lines[i])


def PreprocessByType(data_name, Type):
    option = Options(data_name)

    if os.path.exists(option.net_data) is False and Type.find("social") != -1:
        print("There exists no social grpah!!")
        #BuildSocialGraph(data_name)

    if os.path.exists(option.item_net_data) is False:
        BuildItemGraph(data_name)

    if os.path.exists(option.repost_net_data) is False:
        BuildRepostGraph(data_name)

    if os.path.exists(option.net_data_refined) is False and Type.find(
            "social") != -1:  # os.remove(option.net_data_refined)
        if os.path.exists((option.net_data)) is True:
            RefineSocialNetwork(data_name)


    if os.path.exists(option.repost_net_data_trimmed) is False or os.path.exists(option.net_data_refined) is False:
        if data_name == "./data/twitter":
            BuildTrimedGraphs(data_name, item_threshold=1, threshold=1)
            # BuildUserHiddenGraph(data_name,threshold=0.85)
        elif data_name == "./data/douban":
            BuildTrimedGraphs(data_name, item_threshold=1, threshold=1)
            # BuildUserHiddenGraph(data_name,threshold=0.6)
        elif data_name == "./data/memetracker":
            BuildTrimedGraphs(data_name, item_threshold=1, threshold=1)
            # BuildUserHiddenGraph(data_name,threshold=0.8)



def LoadHeteStaticGraph(data_name, Type):
    options = Options(data_name)
    _ui2idx = {}
    _idx2ui = []

    with open(options.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    with open(options.idx2ui_dict, 'rb') as handle:
        _idx2ui = pickle.load(handle)

    PreprocessByType(data_name, Type)

    edges_list = []
    edges_type_list = []
    edges_weight_list = []

    if Type.find("item") != -1:
        with open(options.item_net_data, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]
            relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _ui2idx and edge[1] in _ui2idx]
            # print(relation_list)
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            temp_edges_type_list = [0] * len(relation_list_reverse)
            # print(relation_list_reverse)
            edges_list += relation_list_reverse
            edges_type_list += temp_edges_type_list
            edges_weight_list += [1.0] * len(relation_list_reverse)

    if Type.find("social") != -1 and os.path.exists(options.net_data_refined) is True:
        with open(options.net_data_refined, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]
            relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _ui2idx and edge[1] in _ui2idx]
            # print(relation_list)
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            temp_edges_type_list = [1] * len(relation_list_reverse)
            # print(relation_list_reverse)
            edges_list += relation_list_reverse
            edges_type_list += temp_edges_type_list
            edges_weight_list += [1.0] * len(relation_list_reverse)
        print("load graph!")

    if Type.find("diffusion") != -1:
        with open(options.repost_net_data, 'r') as handle:
            cas_list = handle.read().strip().split("\n")
            cas_list = [edge.split(',') for edge in cas_list]
            relation_list = [(_ui2idx[edge[0]], _ui2idx[edge[1]]) for edge in cas_list if
                             edge[0] in _ui2idx and edge[1] in _ui2idx]
            # print(relation_list)
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            temp_edges_type_list = [2] * len(relation_list_reverse)
            # print(relation_list_reverse)
            edges_list += relation_list_reverse
            edges_type_list += temp_edges_type_list
            edges_weight_list += [1.0] * len(relation_list_reverse)

    edges_list_tensor = torch.LongTensor(edges_list).t()
    edges_type = torch.LongTensor(edges_type_list)
    edges_weight = torch.FloatTensor(edges_weight_list)
    # edges_weight = torch.ones(edges_list_tensor.size(1)).float()

    graph = Data(edge_index=edges_list_tensor, edge_type=edges_type, edge_attr=edges_weight)

    return graph


def LoadHyperGraph(data_name, Type):
    options = Options(data_name)
    _ui2idx = {}
    _idx2ui = []

    with open(options.ui2idx_dict, 'rb') as handle:
        _ui2idx = pickle.load(handle)
    with open(options.idx2ui_dict, 'rb') as handle:
        _idx2ui = pickle.load(handle)
    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)
    # with open(options.idx2u_dict, 'rb') as handle:
    #    _idx2u = pickle.load(handle)

    cas_idx_init = len(_u2idx)

    PreprocessByType(data_name, Type)

    edges_list = []
    # edges_type_list = []
    edges_weight_list = []

    with open(options.item_net_data, "r") as handle:
        relation_list = handle.read().strip().split("\n")
        for relation in relation_list:
            chunks = relation.split(",")
            edges_list.append((_ui2idx[chunks[0]], _ui2idx[chunks[1]] - cas_idx_init))
            # edges_type_list.append(3)
            edges_weight_list.append(1.0)

    edges_list_tensor = torch.LongTensor(edges_list).t()
    edges_weight = torch.FloatTensor(edges_weight_list)


    graph = Data(edge_index=edges_list_tensor, edge_attr=edges_weight)

    return graph


