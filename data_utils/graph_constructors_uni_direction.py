import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from itertools import product
import torch


def normalise_amounts(inp_amt_arr):
    # output amounts in range [0,1] !!!!!!!!! should we use [-1,1] here?!!!!!!!!!!!!!!!!!

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(inp_amt_arr.reshape(-1, 1))
    return np.squeeze(scaler.transform(inp_amt_arr.reshape(-1, 1)))

    # return (np.max(inp_amt_arr) - inp_amt_arr) / (np.max(inp_amt_arr) - np.min(inp_amt_arr))



def find_edge_features(inp_day_arr, inp_norm_amt_arr, inp_group_arr, edge):
    one_hot_day_arr = np.zeros(91)
    edge_days_diff = int(inp_day_arr[edge[1]] - inp_day_arr[edge[0]])
    #!!!!!!!!!!we are encoding here that all date diffs beyong 90 days will look the same
    if edge_days_diff < 0:
        direction = -1
    elif edge_days_diff == 0:
        direction = 0
    else:
        direction = 1

    if np.abs(edge_days_diff) <= 89:
        one_hot_day_arr[edge_days_diff] = 1
    elif np.abs(edge_days_diff) > 89:
        one_hot_day_arr[90] = 1

    #print('edge', edge)

    edge_amt_diff_arr = np.abs(np.array([inp_norm_amt_arr[edge[1]] - inp_norm_amt_arr[edge[0]]]))

    #return np.concatenate((edge_amt_diff_arr, np.array([direction]), one_hot_day_arr), axis = None)
    return np.concatenate((edge_amt_diff_arr, one_hot_day_arr), axis = None)



def find_edge_labels(inp_day_arr, inp_norm_amt_arr, inp_group_arr, edge):
    if inp_group_arr[edge[0]] == inp_group_arr[edge[1]]:
        return 1
    else:
        return 0


def rev_edge(inp_edge):
    return [inp_edge[1], inp_edge[0]]


def find_nearest_in_interval(inp_day_arr, inp_norm_amt_arr, inp_group_arr, cur_edge_lst =[], cur_edge_features_lst = [], cur_edge_labels_lst = [], interval = [5,9]):
    
    # print('interval', interval)
    # print('inp_norm_amt_arr.shape', inp_norm_amt_arr.shape)
    #idx = np.arange(len(inp_day_arr))
    sort_idx = np.argsort(inp_day_arr)
    sorted_day_arr = inp_day_arr[sort_idx]

    out_edge_lst = cur_edge_lst
    out_edge_features_lst = cur_edge_features_lst
    out_edge_labels_lst = cur_edge_labels_lst

    # find the nearest transactions approx 7 days after given trans- make the relationships symmetric
    for i, d in enumerate(sorted_day_arr):
        # print('d', d)
        source_idx = sort_idx[i]
        #look for a target in the date range
        sep_by_int_mask = np.logical_and(interval[0] <= sorted_day_arr - d , sorted_day_arr - d <= interval[1])
        sep_by_int_idxs = sort_idx[sep_by_int_mask]
        ##!!!!!should the second product below have the terms reversed?
        sep_by_int_edges = list(set([x for x in product([source_idx],sep_by_int_idxs)] + [x for x in product([source_idx],sep_by_int_idxs)]))
        sep_by_int_edges = [list(x) for x in sep_by_int_edges]
        # print('sep_by_int_edges', sep_by_int_edges)

        #iterate through these edges to determine if they are new.  if they are then calc the features and label and add to the lists for output
        for e in sep_by_int_edges: 
            if e in cur_edge_lst or rev_edge(e) in out_edge_lst:
                continue
            else:
                out_edge_lst = out_edge_lst + [e]
                # print('e', e)
                out_edge_features_lst = out_edge_features_lst + [list(find_edge_features(inp_day_arr, inp_norm_amt_arr, inp_group_arr, e))]
                out_edge_labels_lst = out_edge_labels_lst + [find_edge_labels(inp_day_arr, inp_norm_amt_arr, inp_group_arr, e)]

    return out_edge_lst, out_edge_features_lst, out_edge_labels_lst


def find_nearest_amount(inp_day_arr, inp_norm_amt_arr, inp_group_arr, cur_edge_lst =[], cur_edge_features_lst = [], cur_edge_labels_lst = []):
    out_edge_lst = cur_edge_lst
    out_edge_features_lst = cur_edge_features_lst
    out_edge_labels_lst = cur_edge_labels_lst

    try:
        sort_idx = np.argsort(inp_norm_amt_arr)
        sorted_norm_amt_arr = inp_norm_amt_arr[sort_idx]
    except:
        print('inp_norm_amt_arr:', inp_norm_amt_arr)
        print('sort_idx:', sort_idx)          

    for i in range(len(sorted_norm_amt_arr)-1):

        source_idx = sort_idx[i]
        target_idx = sort_idx[i+1]
        e_for = [source_idx, target_idx]
        e_rev = [target_idx, source_idx]

        nearest_amt_edges = [e_for, e_rev]

         #iterate through these edges to determine if they are new.  if they are then calc the features and label and add to the lists for output
        for e in nearest_amt_edges:
            if e in cur_edge_lst or rev_edge(e) in out_edge_lst:
                continue
            else:
                out_edge_lst = out_edge_lst + [e]
                out_edge_features_lst = out_edge_features_lst + [list(find_edge_features(inp_day_arr, inp_norm_amt_arr, inp_group_arr, e))]
                out_edge_labels_lst = out_edge_labels_lst + [find_edge_labels(inp_day_arr, inp_norm_amt_arr, inp_group_arr, e)]

    return out_edge_lst, out_edge_features_lst, out_edge_labels_lst


def find_nearest_day(inp_day_arr, inp_norm_amt_arr, inp_group_arr, cur_edge_lst =[], cur_edge_features_lst = [], cur_edge_labels_lst = []):
    out_edge_lst = cur_edge_lst
    out_edge_features_lst = cur_edge_features_lst
    out_edge_labels_lst = cur_edge_labels_lst

    sort_idx = np.argsort(inp_day_arr)
    sorted_day_arr = inp_day_arr[sort_idx]

    for i in range(len(sorted_day_arr)-1):

        source_idx = sort_idx[i]
        target_idx = sort_idx[i+1]
        e_for = [source_idx, target_idx]
        e_rev = [target_idx, source_idx]

        nearest_day_edges = [e_for, e_rev]

         #iterate through these edges to determine if they are new.  if they are then calc the features and label and add to the lists for output
        for e in nearest_day_edges: 
            if e in cur_edge_lst or rev_edge(e) in out_edge_lst:
                continue
            else:
                out_edge_lst = out_edge_lst + [e]
                out_edge_features_lst = out_edge_features_lst + [list(find_edge_features(inp_day_arr, inp_norm_amt_arr, inp_group_arr, e))]
                out_edge_labels_lst = out_edge_labels_lst + [find_edge_labels(inp_day_arr, inp_norm_amt_arr, inp_group_arr, e)]

    return out_edge_lst, out_edge_features_lst, out_edge_labels_lst


#def find_nearest_same_amount(inp_day_arr, inp_norm_amt_arr, inp_group_arr):
    # find the nearest amount day after given trans where the amount is the same - make the relationships symetric
    # output four arrays: 
        # the indices of the source
        # the indices of the target
        # the edge features - one-hot encoding of day diffs and value of norm_amt diff
        # edge labels - {0,1} with 1 if transactions are in the same group and 0 if not


def make_pyg_graph(inp_day_arr, inp_amt_arr, inp_group_arr):
    #run the above and put into PyG Data object
    intervals = [[5,9], [11,17], [25,35]]
    out_edge_lst = []
    out_edge_features_lst = []
    out_edge_labels_lst = []

    inp_norm_amt_arr = normalise_amounts(inp_amt_arr)

    # for i in intervals:
    #     out_edge_lst, out_edge_features_lst, out_edge_labels_lst = find_nearest_in_interval(inp_day_arr, inp_norm_amt_arr, inp_group_arr, cur_edge_lst =out_edge_lst, cur_edge_features_lst = out_edge_features_lst, cur_edge_labels_lst = out_edge_labels_lst, interval = i)

    out_edge_lst, out_edge_features_lst, out_edge_labels_lst = find_nearest_amount(inp_day_arr, inp_norm_amt_arr, inp_group_arr, cur_edge_lst =out_edge_lst, cur_edge_features_lst = out_edge_features_lst, cur_edge_labels_lst = out_edge_labels_lst)
    out_edge_lst, out_edge_features_lst, out_edge_labels_lst = find_nearest_day(inp_day_arr, inp_norm_amt_arr, inp_group_arr, cur_edge_lst =out_edge_lst, cur_edge_features_lst = out_edge_features_lst, cur_edge_labels_lst = out_edge_labels_lst)

    out_edge_ten = torch.tensor(out_edge_lst)

    x = torch.tensor([0.0], dtype = torch.float).repeat(len(inp_day_arr)).unsqueeze_(-1)
    edge_index= out_edge_ten.t().contiguous()
    edge_attr = torch.tensor(out_edge_features_lst, dtype = torch.float)
    y = torch.tensor(out_edge_labels_lst, dtype = torch.long)
    pos = torch.tensor([x for x in zip(inp_day_arr, inp_amt_arr)])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, norm=None, face=None)