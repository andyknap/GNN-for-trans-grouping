import numpy as np
import timeit
import statistics
import time
import itertools

def make_an_image_from_type(inp_day_arr, inp_amt_arr, inp_group_arr, inp_type_arr, img_size = 224, amt_range = [-1500, 0]):
    
    min_amt, max_amt = amt_range
    amt_spread = max_amt - min_amt


    a_idx = np.array([(np.abs(np.arange(min_amt,max_amt, amt_spread/img_size) - x)).argmin() for x in inp_amt_arr])

    out_img = np.zeros((img_size, img_size,3))

    out_img[a_idx.astype(int), inp_day_arr.astype(int), :] = 255

    out_target = np.zeros((2, img_size, img_size))

    for t_type in [0,1]:
        t_type_mask = np.array([t == t_type for t in inp_type_arr])
        
        t_type_a_idx = a_idx[t_type_mask]
        t_type_d_arr = inp_day_arr[t_type_mask]
        
        out_target[t_type, t_type_a_idx.astype(int) ,t_type_d_arr.astype(int)] = 1

    return out_img, out_target



def make_an_image_from_group(inp_day_arr, inp_amt_arr, inp_group_arr, inp_type_arr, img_size = 224, amt_range = [-1500, 0]):
    
    min_amt, max_amt = amt_range
    amt_spread = max_amt - min_amt

    #we should scramble the group_ids to prevent any inadvertant correlations between group type and number
    sub_dict = dict(zip(np.arange(4),np.random.choice(np.arange(4),4, replace = False)))
    inp_group_arr = np.array([sub_dict[i] for i in inp_group_arr])

    a_idx = np.array([(np.abs(np.arange(min_amt,max_amt, amt_spread/img_size) - x)).argmin() for x in inp_amt_arr])

    out_img = np.zeros((img_size, img_size,3))

    out_img[a_idx.astype(int), inp_day_arr.astype(int), :] = 255

    out_target = np.zeros((4, img_size, img_size))

    for g in [0,1,2,3]:
        g_mask = np.array([x ==g for x in inp_group_arr])
        
        g_a_idx = a_idx[g_mask]
        g_d_arr = inp_day_arr[g_mask]
        
        out_target[g, g_a_idx.astype(int) ,g_d_arr.astype(int)] = 1

    #now create the tensor of permuted targets to use in the loss fuction

    out_target_perm = np.array([out_target[np.array(x), :, :] for x in itertools.permutations([0,1,2,3])])


    return out_img, out_target_perm


