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
FULL_DATA_SIZE = 2458285
SAMPLE_SIZE = 2000
GT_RULE_NUM = 48

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


R = []
cnt = np.zeros(MAX_RULE)
cnt_pos = np.zeros(MAX_RULE)
possible_rules = set()
rule_to_ind = {}
ind_to_rule = {}

gt_pos_ratio = None
gt_rank = None

def update_cnt(cnt, cnt_pos, x, pos, add):
    for i, v in enumerate(x):
        if v == -1:
            continue
        rule_ind = rule_to_ind[(i, v)]
        cnt[rule_ind] += add
        if pos:
            cnt_pos[rule_ind] += add

def get_rank(cnt, cnt_pos):
    num_rules = len(possible_rules)
    pos_ratio = np.nan_to_num(cnt_pos[:num_rules] / cnt[:num_rules], 0.0)
    rank = np.argsort(pos_ratio)[::-1]
    return pos_ratio, rank

def update_stat(x_ind, x, pos):
    # Check if there are any new possible rules
    for i, v in enumerate(x):
        if v == -1: continue
        if (i, v) not in possible_rules:
            rule_to_ind[(i, v)] = len(possible_rules)
            ind_to_rule[len(possible_rules)] = (i, v)
            possible_rules.add((i, v))
    # Update the count of each rule
    update_cnt(cnt, cnt_pos, x, pos, 1)

POS = 15
current_rank = 0
pos_needed = POS
neg_needed = 0
rank_rules = None
def update_sample(x_ind, x, pos):
    global current_rank, pos_needed, neg_needed
    rule = rank_rules[current_rank]

    #for r in rank_rules[:current_rank]:
    #    if x[r[0]] == r[1]:
    #        return

    if x[rule[0]] == rule[1]:
        if pos_needed > 0 and pos:
            R.append(x_ind)
            pos_needed -= 1
        elif neg_needed > 0 and not pos:
            R.append(x_ind)
            neg_needed -= 1

    if pos_needed == 0 and neg_needed == 0:
        current_rank += 1
        pos_needed = POS
        neg_needed = current_rank

def main():
    attrs, data, labels = load_data()
    stat_ratio = 0.1
    delimiter = int(data.shape[0] * stat_ratio)
    for i in tqdm(range(0, delimiter)):
        update_stat(i, data[i], labels[i])

    global gt_pos_ratio, gt_rank
    gt_pos_ratio, gt_rank = get_rank(cnt, cnt_pos)

    global rank_rules
    rank_rules = [ind_to_rule[rule_id] for rule_id in gt_rank]

    for rule_id in gt_rank[:48]:
        rule = ind_to_rule[rule_id]
        print(f'{attrs[rule[0]]} = {rule[1]}')
    
    for i in tqdm(range(delimiter, data.shape[0])):
        update_sample(i, data[i], labels[i])
        global R
        if len(R) >= SAMPLE_SIZE:
            R = R[:SAMPLE_SIZE]
            break
    
    print(len(R))
    print(t2.evaluation(R))

if __name__ == '__main__':
    random.seed(777777)
    np.random.seed(777777)
    with open('/tmp2/b07902075/t2_evaluation.pickle', 'rb') as f:
        t2 = pickle.load(f)
    #print(t2.rule_whole_data)
    main()