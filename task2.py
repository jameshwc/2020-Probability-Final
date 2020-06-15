from evaluation_t2 import t2_evaluation

import pickle
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import random
import heapq

DATA_PATH = './t2_data_v2.csv'
LABEL_ATTR = 'label'
POSITIVE = 1
MAX_RULE = 391
SAMPLE_SIZE = 2000

t2 = None

def preprocess():
    df = pd.read_csv(DATA_PATH, index_col=0)
    df = df.fillna(-1)
    attrs = df.drop([LABEL_ATTR], axis=1).columns
    label_ind = list(df.columns).index(LABEL_ATTR)
    arr = df.to_numpy().astype('int8')
    obj = {
        'attrs': list(attrs),
        'label_index': label_ind,
        'arr': arr
    }
    with open('t2_data.pickle', 'wb') as f:
        pickle.dump(obj, f)

def load_data():
    with open('t2_data.pickle', 'rb') as f:
        obj = pickle.load(f)
    attrs = obj['attrs']
    label_index = obj['label_index']
    data = obj['arr']
    labels = (data[:, label_index] == POSITIVE)
    data = np.delete(data, label_index, axis=1)
    return attrs, data, labels


attrs = None

R = []
cnt = np.zeros(MAX_RULE)
cnt_pos = np.zeros(MAX_RULE)
sample_cnt = np.zeros(MAX_RULE)
sample_cnt_pos = np.zeros(MAX_RULE)
possible_rules = set()
rule_to_ind = {}
ind_to_rule = {}

def init_sample_cnt():
    for _, x, pos in R:
        update_cnt(sample_cnt, sample_cnt_pos, x, pos, 1)

def update_cnt(cnt, cnt_pos, x, pos, add):
    for i, v in enumerate(x):
        if v == -1: continue
        rule_ind = rule_to_ind[(i, v)]
        cnt[rule_ind] += add
        if pos: cnt_pos[rule_ind] += add

def get_rank(cnt, cnt_pos):
    num_rules = len(possible_rules)
    pos_ratio = np.nan_to_num(cnt_pos[:num_rules] / cnt[:num_rules], 0.0)
    order = np.argsort(pos_ratio)[::-1]
    rank = np.argsort(order)
    return rank

def update(x_ind, x, pos):
    # Check if there are any new possible rules
    for i, v in enumerate(x):
        if v == -1: continue
        if (i, v) not in possible_rules:
            rule_to_ind[(i, v)] = len(possible_rules)
            ind_to_rule[len(possible_rules)] = (i, v)
            possible_rules.add((i, v))

    # Update the count of each rule
    update_cnt(cnt, cnt_pos, x, pos, 1)

    if x_ind < SAMPLE_SIZE:
        R.append((x_ind, x, pos))
        return
    elif x_ind == SAMPLE_SIZE:
        init_sample_cnt()

    rank_data = get_rank(cnt, cnt_pos)
    rank_sample = get_rank(sample_cnt, sample_cnt_pos)

def main():
    global attrs
    attrs, data, labels = load_data()
    
    for i in range(0, data.shape[0], 1):
        update(i, data[i], labels[i])

if __name__ == '__main__':
    random.seed(777777)
    np.random.seed(777777)
    with open('/tmp2/b07902075/t2_evaluation.pickle', 'rb') as f:
        t2 = pickle.load(f)
    #print(t2.rule_whole_data)
    main()
