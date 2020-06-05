from collections import defaultdict

import numpy as np
import scipy.stats
import pandas as pd
import random


def probability_normalize(p):
    norm_p = p / np.sum(p)
    return norm_p

def hellinger_distance(p, q):
    # norm_p = probability_normalize(p)
    # norm_q = probability_normalize(q)
    BC = np.sum(np.sqrt(p * q))
    return np.sqrt(1 - BC)

def get_count_dict(df, skip_columns=0):
    count_dict = defaultdict(dict)
    for c_name in df.columns[skip_columns:]:
        c = df[c_name]
        c_count = c.groupby(c).count()
        count_dict[c_name] = defaultdict(int)
        for i in range(c_count.shape[0]):
            count_dict[c_name][c_count.index[i]] = c_count.iloc[i]
    return count_dict

def compare_count_dict(full_cd, sample_cd):
    dist_list = []
    max_diff_list = []
    mean_diff_list = []
    for c_name in full_cd.keys():
        n_keys = len(full_cd[c_name])
        p, q = np.zeros(n_keys), np.zeros(n_keys)
        for i, k in enumerate(full_cd[c_name].keys()):
            p[i] = full_cd[c_name][k]
            q[i] = sample_cd[c_name][k]
        ### Make p, q like probability
        p = probability_normalize(p)
        q = probability_normalize(q)
        ### Compare distributions
        dist_list.append(hellinger_distance(p, q))
    return dist_list

def t1_evaluation(full_data_path, index_list):
    full_df = pd.read_csv(full_data_path)
    full_cd = get_count_dict(full_df, skip_columns=1)
    print(full_cd)
    index_list = sorted(list(set(index_list)))
    sample_df = full_df.iloc[index_list, :]
    sample_cd = get_count_dict(sample_df, skip_columns=1)
    dist_list = compare_count_dict(full_cd, sample_cd)
    return np.mean(dist_list)

index_list = random.sample(range(1999999), 20000)
print(t1_evaluation('t1-data.csv', index_list))
# t1_evaluation('t1-data.csv', None)