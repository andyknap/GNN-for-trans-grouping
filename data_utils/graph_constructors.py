import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data


def normalise_amounts(inp_amt_arr):
    # output amounts in range [0,1] !!!!!!!!! should we use [-1,1] here?!!!!!!!!!!!!!!!!!
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(inp_amt_arr)
    return scaler.transform(inp_amt_arr)


def find_edge_features(inp_day_arr, inp_norm_amt_arr, inp_group_arr, edge):
    one_hot_day_arr = np.zeros(91)
    edge_days_diff = int(inp_day_arr[edge[1]] - inp_day_arr[edge[0]])
    #!!!!!!!!!!we are encoding here that all date diffs beyong 90 days will look the same
    if edge_days_diff <= 89:
        one_hot_day_arr[edge_days_diff] = 1
    else:
        one_hot_day_arr[90] = 1
    edge_amt_diff_arr = np.array([inp_norm_amt_arr[edge[1]] - inp_norm_amt_arr[edge[0]]])
    return np.concatenate(edge_amt_diff_arr, one_hot_day_arr)


def find_edge_labels(inp_day_arr, inp_norm_amt_arr, inp_group_arr, edge):
    if inp_group_arr[edge[0]] == inp_group_arr[edge[1]]:
        return 1
     else:
        return 0


def find_nearest_in_interval(inp_day_arr, inp_norm_amt_arr, inp_group_arr, cur_edge_lst =[], cur_edge_features_lst = [], cur_edge_labels_lst = [], interval = [5,9]):
    idx = np.arange(len(inp_day_arr))
    sort_idx = np.argsort(inp_day_arr)
    sorted_day_arr = inp_day_arr[sort_idx]

    out_edge_lst = cur_edge_lst
    out_edge_features_lst = cur_edge_features_lst
    out_edge_labels_lst = cur_edge_labels_lst

    # find the nearest transactions approx 7 days after given trans- make the relationships symmetric
    for d in sorted_day_arr:
        source_idx = sort_idx[d]
        #look for a target in the date range
        sep_by_int_mask = np.logical_and(interval[0] <= sorted_day_arr - d , sorted_day_arr - d <= interval[1])
        sep_by_int_idxs = sort_idx[sep_by_int_mask]
        sep_by_int_edges = list(set([x for x in product([source_idx],sep_by_int_idxs)] + [x for x in product([source_idx],sep_by_int_idxs)]))

        #iterate through these edges to determine if they are new.  if they are then calc the features and label and add to the lists for output
        for e in sep_by_int_edges: 
            if e in cur_edge_lst:
                continue
            else:
                out_edge_lst = out_edge_lst + [e]
                out_edge_features_lst = out_edge_features_lst + [list(find_edge_features(inp_day_arr, inp_norm_amt_arr, inp_group_arr, e))]
                out_edge_labels_lst = out_edge_labels_lst + [find_edge_labels(inp_day_arr, inp_norm_amt_arr, inp_group_arr, e)]

    return out_edge_lst, out_edge_features_lst, out_edge_labels_lst


def find_nearest_amount(inp_day_arr, inp_norm_amt_arr, inp_group_arr):
    # find the nearest amount transaction with amt >= given trans- make the relationships symetric
    # output four arrays: 
        # the indices of the source
        # the indices of the target
        # the edge features - one-hot encoding of day diffs and value of norm_amt diff
        # edge labels - {0,1} with 1 if transactions are in the same group and 0 if not


def find_nearest_day(inp_day_arr, inp_norm_amt_arr, inp_group_arr):
    # find the nearest amount transaction with day >= given trans- make the relationships symetric
    # output four arrays: 
        # the indices of the source
        # the indices of the target
        # the edge features - one-hot encoding of day diffs and value of norm_amt diff
        # edge labels - {0,1} with 1 if transactions are in the same group and 0 if not


def find_nearest_same_amount(inp_day_arr, inp_norm_amt_arr, inp_group_arr):
    # find the nearest amount day after given trans where the amount is the same - make the relationships symetric
    # output four arrays: 
        # the indices of the source
        # the indices of the target
        # the edge features - one-hot encoding of day diffs and value of norm_amt diff
        # edge labels - {0,1} with 1 if transactions are in the same group and 0 if not


def make_graph_arrs_from_group(inp_day_arr, inp_amt_arr, inp_group_arr):
    # run the above and dedupe
    # output four arrays: 
        # the indices of the source
        # the indices of the target
        # the edge features - one-hot encoding of day diffs and value of norm_amt diff
        # edge labels - {0,1} with 1 if transactions are in the same group and 0 if not


def make_pyg_graph(inp_day_arr, inp_amt_arr, inp_group_arr):
    #run the above and put into PyG Data object
    return Data(x=None, edge_index=None, edge_attr=None, y=None, pos=None, norm=None, face=None)