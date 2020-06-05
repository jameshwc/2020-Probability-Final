import pandas as pd
import numpy as np
import random, pickle
import threading
import queue
from tqdm import tqdm
from time import time

from evaluation_t1 import t1_evaluation

FULL_DATA_PATH = './t1-data-preprocessed.npy'
FULL_SIZE = 2000000
SAMPLE_SIZE = 20000
SEED = 777777

class ColInfo:
    def __init__(self, p):
        self.p = p
        self.q = None

class Evaluator:
    def __init__(self, full_arr, init_indices=[]):
        self.full_arr = full_arr
        print('[Evaluator] Building column information list...')
        self.col_info_list = self.get_col_info_list()
        self.index_list = init_indices
        print('[Evaluator] Initializing sample...')
        self.init_sample()
        print('[Evaluator] Initialization done.')

    @staticmethod
    def get_distance(p, q):
        BC = np.sum(np.sqrt(p * q))
        return np.sqrt(1 - BC)

    @staticmethod
    def prob_norm(p):
        S = np.sum(p)
        return p / S if S != 0.0 else p

    def get_col_info_list(self):
        col_info_list = []
        for col_id in range(self.full_arr.shape[1]):
            _, cnt = np.unique(self.full_arr[:, col_id], return_counts=True)
            col_info_list.append(ColInfo(self.prob_norm(cnt)))
        return col_info_list

    def init_sample(self):
        arr = self.full_arr[self.index_list]
        for col_id, col_info in enumerate(self.col_info_list):
            cnt = np.zeros(col_info.p.shape[0])
            for i in arr[:, col_id]:
                cnt[i] += 1
            col_info.q = cnt

    def add(self, indices):
        for i in indices:
            for col_id, col_info in enumerate(self.col_info_list):
                col_info.q[self.full_arr[i][col_id]] += 1

    def try_add(self, indices):
        dist_list = []
        for col_id, col_info in enumerate(self.col_info_list):
            q = col_info.q.copy()
            for i in indices:
                q[self.full_arr[i][col_id]] += 1
            dist = self.get_distance(col_info.p, self.prob_norm(q))
            dist_list.append(dist)
        return np.mean(dist_list)


def random_indices(sample_size, data_size):
    lst = [i for i in range(data_size)]
    random.shuffle(lst)
    return lst[:sample_size]

def greedy_method(init_set_size=10, batch_size=100, num_batches=100, full_data_size=FULL_SIZE, sample_size=SAMPLE_SIZE):
    R = set(random_indices(init_set_size, full_data_size))
    S = set(list(range(full_data_size))).difference(R)
    full_arr = np.load(FULL_DATA_PATH)
    evaluator = Evaluator(full_arr, init_indices=list(R))

    while len(R) < sample_size:
        print(f'|R| = {len(R)}')
        best_score = 999999
        best_batch = None
        k = min(sample_size - len(R), batch_size)
        for _ in tqdm(range(num_batches)):
            B = list(random.sample(S, k))
            score = evaluator.try_add(B)
            if score < best_score:
                best_batch = B
                best_score = score

        print(best_score)
        evaluator.add(best_batch)
        R.update(best_batch)
        S.difference_update(best_batch)

    return list(R)

if __name__ == '__main__':
    random.seed(SEED)
    np.random.seed(SEED)
    R = greedy_method(init_set_size=100, batch_size=100, num_batches=500)
    with open('R.pickle', 'wb') as f:
        pickle.dump(R, f)
