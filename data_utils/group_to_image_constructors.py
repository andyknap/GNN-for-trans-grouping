import numpy as np
import timeit
import statistics
import time

def make_an_image(inp_day_arr, inp_amt_arr, inp_group_arr, inp_type_arr, img_size = 212, amt_range = [-1500, 0]):
    
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


