import numpy as np
import random
import pandas as pd
import warnings
import time
import sys

warnings.simplefilter(action='ignore', category=FutureWarning)
data_path = sys.argv[1]

sample_ind = set()

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_data(file_name):
    df = pd.read_csv(file_name, index_col = 0).to_numpy()
    return df

def get_transpose(data):
    trans = np.transpose(data, (1, 0))
    return trans

def reassign_feature(feature):
    for row in feature:
        ind = np.unique(row)
        for i in range(len(ind)): row[row == ind[i]] = i

def calibrate(data):
    trans_df = get_transpose(data)
    reassign_feature(trans_df)
    return get_transpose(trans_df)

def get_attr_cnt(data, sample_num= 20000):
    feature = get_transpose(data)
    max_feature_cnt = 0
    for row in feature: max_feature_cnt = max(max_feature_cnt, len(np.unique(row)))
    attr_cnt = np.zeros((feature.shape[0], max_feature_cnt)).astype(np.int)
    for ind, row in enumerate(feature):
        attr, cnt = np.unique(row, return_counts = True)
        cnt = np.round(sample_num * cnt / np.sum(cnt)).astype(np.int)
        cnt[0] += sample_num - np.sum(cnt)
        for i, c in zip(attr, cnt): attr_cnt[ind][i] = c
    return attr_cnt

def dump_file(lst, file_name = 'index.txt'):
    f = open(file_name, 'w')
    for item in lst: print(item, file = f)

def fit(data, attr_cnt, epoch = 5, sample_num = 20000):
    global sample_ind

    def add_sample(ind):
        nonlocal data, attr_cnt
        global sample_ind
        sample_ind.add(ind)
        for i, attr in enumerate(data[ind]): attr_cnt[i][attr] -= 1

    def remove_sample(ind):
        nonlocal data, attr_cnt
        global sample_ind
        sample_ind.remove(ind)
        for i, attr in enumerate(data[ind]): attr_cnt[i][attr] += 1

    cp = np.hstack((data, np.arange(data.shape[0]).reshape(-1, 1)))
    feature = get_transpose(data)
    feature_ind = np.array([len(np.unique(feature[x])) for x in range(len(feature))])
    feature_ind = np.argsort(feature_ind)
    for e in range(epoch):
        print(f'running the {e + 1}-th epoch')
        t = time.time()
        for c, i in enumerate(feature_ind):
            print(c, i)
            print(attr_cnt[i])
            np.random.shuffle(cp)
            for row in cp:
                attr = row[i]
                if attr_cnt[i][attr] > 0 and row[-1] not in sample_ind: add_sample(row[-1])
                if np.max(attr_cnt[i]) == 0: break
            print(attr_cnt[i])
            if len(sample_ind) > sample_num:
                redundent = []
                for ind in sample_ind:
                    w = 0
                    for c, attr in enumerate(data[ind]):
                        if attr_cnt[c][attr] < 0:
                            w += 1
                            if c == i: w += len(data[ind])
                    redundent.append((ind, w))
                redundent = sorted(redundent, key = lambda x: x[1])
                while len(sample_ind) > sample_num:
                    cur, _ = redundent.pop()
                    if attr_cnt[i][data[cur][i]] == 0: continue
                    remove_sample(cur)
            print(attr_cnt[i])

        print(time.time() - t)
    dump_file(list(sample_ind))


t = time.time()
fix_seed(64)
df = get_data(data_path)
df = calibrate(df)
attr_cnt = get_attr_cnt(df)
print(time.time() - t)
fit(df, attr_cnt, epoch = 6)
