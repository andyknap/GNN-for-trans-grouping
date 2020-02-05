import numpy as np
import timeit
import statistics
import time


def find_raw_amount_arr(num_periods, dist_version):
    
    if dist_version == 0:
        std =  np.random.uniform(low=0.5, high=3.0, size=None)
        raw_arr =  np.random.normal(0, std, num_periods)
        if np.max(raw_arr) == np.min(raw_arr):
            return raw_arr
        else:
            return (raw_arr - np.min(raw_arr)) / (np.max(raw_arr) - np.min(raw_arr))
        
    elif dist_version == 1:
        raw_arr =   np.random.uniform(low=-1.0, high=1.0, size=num_periods)
        if np.max(raw_arr) == np.min(raw_arr):
            return raw_arr
        else:
            return (raw_arr - np.min(raw_arr)) / (np.max(raw_arr) - np.min(raw_arr))
            
    else:
        return np.zeros(num_periods)
    

def find_amount_arr(inp_raw_amount_arr, amount_mean, amount_spread):
    return amount_spread*inp_raw_amount_arr + (amount_mean)


def find_steps_array(num_periods):

    #choose the number of steps
    num_steps=np.random.choice(3, 1, p=[0.0, 0.9, 0.1])[0]
    if num_steps ==0:
        return np.zeros(num_periods)
    step_locs_arr = np.sort(np.random.choice(np.arange(1,num_periods), size=num_steps, replace=False, p=None))
    step_arr = np.zeros((num_steps, num_periods))
    for i in range(num_steps):
        step_arr[i,np.arange(step_locs_arr[i], num_periods)] = 2*(np.random.random(1)-0.5)
    step_arr = np.sum(step_arr, axis=0)
    return  step_arr / max(np.max(step_arr), np.abs(np.min(step_arr)))


def find_outlier_array(num_periods):
    num_outliers=np.random.choice(3, 1, p=[0.0, 0.9, 0.1])[0]
    if num_outliers ==0:
        return np.zeros(num_periods)
    outliers_locs_arr = np.sort(np.random.choice(np.arange(1,num_periods), size=num_outliers, replace=False, p=None))
    outliers_arr = np.zeros(num_periods)
    outliers_arr[outliers_locs_arr] = np.random.random(num_outliers)
    return  outliers_arr / max(np.max(outliers_arr), np.abs(np.min(outliers_arr))), outliers_locs_arr


def find_skip_period_idxs(num_periods):
    num_skip_periods=np.random.choice(3, 1, p=[0.0, 0.9, 0.1])[0]
    if num_skip_periods ==0:
        return np.zeros(num_periods)
    skip_periods_arr = np.zeros(num_periods)
    skip_periods_idxs = np.random.choice(np.arange(1,num_periods), size=num_skip_periods, replace=False, p=None)
    #skip_periods_arr[skip_periods_idxs] = 1
    return skip_periods_idxs


def make_a_regular_group(period
                         , period_var
                         , num_periods
                         , amount_mean
                         , amount_spread
                         , amount_step_prop
                         , amount_outlier_prop
                         , split_payment_prob = 0
                         , extra_payment_prob = 0
                         , extra_payment_val = 0
                         , initial_payment_prob = 0
                         , initial_payment_val = 0
                         , final_payment_prob = 0
                         , final_payment_val = 0):
    
    #create the template sequence
    raw_amount_arr = find_raw_amount_arr(num_periods, np.random.randint(3, size=1)) 
    amount_arr = find_amount_arr(raw_amount_arr, amount_mean, amount_spread)
    days_arr = np.arange(0, period*num_periods, period)
    
    #step the amounts
    steps_arr = find_steps_array(num_periods)*amount_step_prop*amount_mean
    amount_arr = amount_arr+steps_arr
    
    #create some outliers
    outlier_arr, outlier_loc_arr = find_outlier_array(num_periods)*np.random.uniform(1,amount_outlier_prop, size = 1)
    amount_arr = amount_arr + outlier_arr
    
    #skip some periods
    skip_periods_idxs = find_skip_period_idxs(num_periods)
    amount_arr = np.delete(amount_arr,skip_periods_idxs)
    days_arr = np.delete(days_arr,skip_periods_idxs)
    num_periods = num_periods - len(skip_periods_idxs)
    
    #make sure everything is the same sign
    #flip randomly to be negative
    
    return days_arr, amount_arr


def make_an_irregular_group(num_trans,
                           mean_amt,
                           mean_days,
                           amt_spread,
                           days_spread,
                           amt_dist_type = '',
                           days_dist_type = ''):
    #sample just from normal to start
    raw_amt_arr = np.random.normal(loc=mean_amt,scale=amt_spread, size=num_trans)
    raw_days_arr = np.random.normal(loc=mean_days,scale=days_spread, size=num_trans)
    
    #remove ones we don't want
    select_mask = np.logical_and(np.sign(raw_amt_arr) == np.sign(mean_amt), raw_days_arr >=0)
    
    amount_arr = raw_amt_arr[select_mask]
    days_arr = raw_days_arr[select_mask].astype(int)
    
    return days_arr, amount_arr
    

def make_a_group_inner():

    # structure the labels so that (ir)regular groups are always in a given range

    #select number of regualr groups
    #num_reg_groups = np.random.randint(4,size=1)[0]

    #with the current group labelling hack this value must be <= 3
    # num_reg_groups = 1
    num_reg_groups = np.random.randint(0,3)

    #print(num_reg_groups)
    #select range of reg group sizes
    reg_group_sizes = np.random.randint(3,30, size=num_reg_groups)
    #select amount mean of reg group sizes
    reg_group_amt_means = np.random.uniform(-1000,-100, size=num_reg_groups)
    #select reg group offsets
    reg_group_offsets = np.random.randint(0,10, size=num_reg_groups)
    
    
    day_list = []
    amt_list = []
    group_id_list = []
    group_type_list = []
    
    for g in range(num_reg_groups):
        #print(g)
        
        d_arr, a_arr = make_a_regular_group(period = 7
                             , period_var = 1
                             , num_periods = reg_group_sizes[g]
                             , amount_mean = reg_group_amt_means[g]
                             , amount_spread = 10
                             , amount_step_prop = 0.2
                             , amount_outlier_prop = 0.2)
        
        if g == 0:
            reg_group_offset = 0
        else:
            reg_group_offset = reg_group_offsets[g]
        
        day_list = day_list + list(d_arr+reg_group_offset)
        amt_list = amt_list + list(a_arr)
        group_id_list = group_id_list + [g]*len(d_arr)
        group_type_list = group_type_list + [1]*len(d_arr)
        
        
    #add some irregs
    



    if len(day_list) == 0 or np.random.uniform(0,1) <= 0.5:
    #if np.random.uniform(0,1) <= -1:
    
        irreg_group_sizes = np.random.randint(3,30)
        irreg_group_amt_mean = np.random.uniform(-1000,-100)
        if len(day_list) == 0:
            irreg_group_days_mean = np.random.randint(0,150)
            irreg_group_id = 0
        else:
            irreg_group_days_mean = statistics.mean(day_list)
            irreg_group_id = max(group_id_list)+1

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #this is a bit of a hack to make sure irreg groups (of which the code only currently creates one) are labelled in a different range to regular
        irreg_group_id = 3


        d_arr, a_arr = make_an_irregular_group(num_trans = irreg_group_sizes,
                               mean_amt = irreg_group_amt_mean,
                               mean_days = irreg_group_days_mean,
                               amt_spread = 100,
                               days_spread = 25,
                               amt_dist_type = '',
                               days_dist_type = '')
        
        day_list = day_list + list(d_arr)
        amt_list = amt_list + list(a_arr)
        group_id_list = group_id_list + [irreg_group_id]*len(d_arr)
        group_type_list = group_type_list + [0]*len(d_arr)

        #looks like some of the amount truncation can leave groups of size 1 so we need to make sure these are arrays and not scalrs

        # if np.isscalar(day_list):
        #     day_list = np.expand_dims(day_list, axis = 0)
        # if np.isscalar(amt_list):
        #     amt_list = np.expand_dims(amt_list, axis = 0)
        # if np.isscalar(group_id_list):
        #     group_id_list = np.expand_dims(group_id_list, axis = 0)                      
    
    return day_list, amt_list, group_id_list, group_type_list
    

def make_a_group():
    for i in range(100):
        day_list, amt_list, group_id_list, group_type_list = make_a_group_inner()
        if len(day_list) >= 3:
            break
    return day_list, amt_list, group_id_list, group_type_list
