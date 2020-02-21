import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from torch_geometric.data import Data
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

    edge_amt_diff_arr = np.array([inp_norm_amt_arr[edge[1]] - inp_norm_amt_arr[edge[0]]])

    return np.concatenate((edge_amt_diff_arr, np.array([direction]), one_hot_day_arr), axis = None)



def find_edge_labels(inp_day_arr, inp_norm_amt_arr, inp_group_arr, edge):
    if inp_group_arr[edge[0]] == inp_group_arr[edge[1]]:
        return 1
    else:
        return 0


def rev_edge(inp_edge):
    return [inp_edge[1], inp_edge[0]]



def find_nearest_in_interval(inp_day_arr, inp_norm_amt_arr, inp_group_arr, cur_edge_lst =[], cur_edge_features_lst = [], cur_edge_labels_lst = [], interval = [5,9]):
    #idx = np.arange(len(inp_day_arr))
    sort_idx = np.argsort(inp_day_arr)
    sorted_day_arr = inp_day_arr[sort_idx]

    out_edge_lst = cur_edge_lst
    out_edge_features_lst = cur_edge_features_lst
    out_edge_labels_lst = cur_edge_labels_lst

    # find the nearest transactions approx 7 days after given trans- make the relationships symmetric
    for i, d in enumerate(sorted_day_arr):
        source_idx = sort_idx[i]
        #look for a target in the date range
        sep_by_int_mask = np.logical_and(interval[0] <= sorted_day_arr - d , sorted_day_arr - d <= interval[1])
        sep_by_int_idxs = sort_idx[sep_by_int_mask]
        sep_by_int_edges = list(set([x for x in product([source_idx],sep_by_int_idxs)] + [x for x in product(sep_by_int_idxs, [source_idx])]))
        sep_by_int_edges = [list(x) for x in sep_by_int_edges]
        #iterate through these edges to determine if they are new.  if they are then calc the features and label and add to the lists for output
        for e in sep_by_int_edges: 
            #if e in out_edge_lst or rev_edge(e) in out_edge_lst:
            if e in out_edge_lst:
                continue
            else:
                out_edge_lst = out_edge_lst + [e]
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
            # if e in out_edge_lst or rev_edge(e) in out_edge_lst:
            if e in out_edge_lst:
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
            #if e in out_edge_lst or rev_edge(e) in out_edge_lst:
            if e in out_edge_lst:
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


def find_edge_pair_idxs(inp_edge_lst):
    edge_pairs = []
    for i in range(len(inp_edge_lst)):
        for j in range(i, len(inp_edge_lst)):
            if inp_edge_lst[i] == [inp_edge_lst[j][1], inp_edge_lst[j][0]]:
                edge_pairs = edge_pairs + [[i,j]]
    return np.array(edge_pairs)


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def find_binary_node_features_from_days(inp_day_arr):
    day_of_the_week_arr = np.mod(inp_day_arr,7)
    day_bin_arr = np.array([bin_array(int(x), 10) for x in inp_day_arr])
    day_of_the_week_bin_arr = np.array([bin_array(int(x), 3) for x in day_of_the_week_arr])
    return np.hstack((day_bin_arr,day_of_the_week_bin_arr))


def make_pyg_graph_with_edge_attr(inp_day_arr, inp_amt_arr, inp_group_arr):
    #run the above and put into PyG Data object
    intervals = [[5,9], [11,17], [25,35]]
    out_edge_lst = []
    out_edge_features_lst = []
    out_edge_labels_lst = []

    inp_norm_amt_arr = normalise_amounts(inp_amt_arr)

    for i in intervals:
        # out_edge_lst, out_edge_features_lst, out_edge_labels_lst = find_nearest_in_interval(inp_day_arr, inp_norm_amt_arr, inp_group_arr, cur_edge_lst =out_edge_lst, cur_edge_features_lst = out_edge_features_lst, cur_edge_labels_lst = out_edge_labels_lst, interval = i)

        out_edge_lst, out_edge_features_lst, out_edge_labels_lst = find_nearest_amount(inp_day_arr, inp_norm_amt_arr, inp_group_arr, cur_edge_lst =out_edge_lst, cur_edge_features_lst = out_edge_features_lst, cur_edge_labels_lst = out_edge_labels_lst)
        out_edge_lst, out_edge_features_lst, out_edge_labels_lst = find_nearest_day(inp_day_arr, inp_norm_amt_arr, inp_group_arr, cur_edge_lst =out_edge_lst, cur_edge_features_lst = out_edge_features_lst, cur_edge_labels_lst = out_edge_labels_lst)

    edge_pair_idxs_arr = find_edge_pair_idxs(out_edge_lst)

    edge_pair_idxs_ten = torch.tensor(edge_pair_idxs_arr, dtype = torch.long)

    out_edge_ten = torch.tensor(out_edge_lst)

    x = torch.tensor([0.0], dtype = torch.float).repeat(len(inp_day_arr)).unsqueeze_(-1)
    edge_index= out_edge_ten.t().contiguous()
    edge_attr = torch.tensor(out_edge_features_lst, dtype = torch.float)
    #need to reduce y so we have only 1 output per edge pair and the orders are consistent with the edge pairs tensor that will be used in the training
    y_bi = torch.tensor(out_edge_labels_lst, dtype = torch.long)
    y= y_bi[edge_pair_idxs_ten[:,0]]

    pos = torch.tensor([x for x in zip(inp_day_arr, inp_amt_arr)])

    out_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, norm=None, face=None)

    out_data.edge_pairs = edge_pair_idxs_ten

    out_data.y_bi = y_bi

    # out_data.group_id = dict(zip([i for i in range(len(inp_group_arr))], inp_group_arr)) 

    out_data.group_id = torch.tensor(inp_group_arr, dtype = torch.long)

    if len([*edge_pair_idxs_ten.shape]) != 2:
        print('edge_pair_idxs_ten: ', edge_pair_idxs_ten)

    return out_data


def make_pyg_graph_no_edge_attr(inp_day_arr, inp_amt_arr, inp_group_arr, inp_type_arr):
    #run the above and put into PyG Data object
    intervals = [[5,9], [11,17], [25,35]]
    out_edge_lst = []
    out_edge_features_lst = []
    out_edge_labels_lst = []

    ##################normalise the amounts
    inp_norm_amt_arr = normalise_amounts(inp_amt_arr)

    ##################loop through the intervals and create the edges and edge features based on the various criteria
    for _ in intervals:
        # out_edge_lst, out_edge_features_lst, out_edge_labels_lst = find_nearest_in_interval(inp_day_arr, inp_norm_amt_arr, inp_group_arr, cur_edge_lst =out_edge_lst, cur_edge_features_lst = out_edge_features_lst, cur_edge_labels_lst = out_edge_labels_lst, interval = i)
        out_edge_lst, out_edge_features_lst, out_edge_labels_lst = find_nearest_amount(inp_day_arr, inp_norm_amt_arr, inp_group_arr, cur_edge_lst =out_edge_lst, cur_edge_features_lst = out_edge_features_lst, cur_edge_labels_lst = out_edge_labels_lst)
        out_edge_lst, out_edge_features_lst, out_edge_labels_lst = find_nearest_day(inp_day_arr, inp_norm_amt_arr, inp_group_arr, cur_edge_lst =out_edge_lst, cur_edge_features_lst = out_edge_features_lst, cur_edge_labels_lst = out_edge_labels_lst)

    #################record the edge pairs so they the edges linking the same two node (but in different directions) can be easily paired if necessary
    # edge_pair_idxs_arr = find_edge_pair_idxs(out_edge_lst)
    # edge_pair_idxs_ten = torch.tensor(edge_pair_idxs_arr, dtype = torch.long)

    ###############get the edge index tensor in the right format
    out_edge_ten = torch.tensor(out_edge_lst)
    edge_index= out_edge_ten.t().contiguous()

    ################set the edge attributes
    # edge_attr = torch.tensor(out_edge_features_lst, dtype = torch.float)

    #################set the node attributes
    bin_node_features = find_binary_node_features_from_days(inp_day_arr)
    x_arr = np.concatenate((bin_node_features, np.expand_dims(inp_norm_amt_arr, axis = 1)), axis = 1)
    x = torch.tensor(x_arr, dtype = torch.float)

    #################set the labels for the edges
    #################need to reduce y so we have only 1 output per edge pair and the orders are consistent with the edge pairs tensor that will be used in the training
    # y_bi = torch.tensor(out_edge_labels_lst, dtype = torch.long)
    # y= y_bi[edge_pair_idxs_ten[:,0]]

    #################set the labels (types) for the nodes
    y = torch.tensor(inp_type_arr, dtype = torch.long)

    ##################set the data for the plotting
    pos = torch.tensor([x for x in zip(inp_day_arr, inp_amt_arr)])

    ##################create the graph object
    # out_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, norm=None, face=None)
    out_data = Data(x=x, edge_index=edge_index, edge_attr=None, y=y, pos=pos, norm=None, face=None)


    ##################add the additional attributes to the object
    # out_data.edge_pairs = edge_pair_idxs_ten
    # out_data.y_bi = y_bi
    # out_data.group_id = torch.tensor(inp_group_arr, dtype = torch.long)


    # if len([*edge_pair_idxs_ten.shape]) != 2:
    #     print('edge_pair_idxs_ten: ', edge_pair_idxs_ten)

    return out_data