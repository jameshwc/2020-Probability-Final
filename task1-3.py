import pandas as pd
import numpy as np
import random, pickle
import threading
import queue
from tqdm import tqdm
from time import time

from evaluation_t1 import t1_evaluation

FULL_DATA_PATH = './t1-data-preprocessed.npy'
FULL_SIZE = 200000
SAMPLE_SIZE = 2000
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

    def try_(self, indices, value=1):
        dist_list = []
        for col_id, col_info in enumerate(self.col_info_list):
            q = col_info.q.copy()
            for i in indices:
                q[self.full_arr[i][col_id]] += value
            dist = self.get_distance(col_info.p, self.prob_norm(q))
            dist_list.append(dist)
        return np.mean(dist_list)
    def score(self):
        dist_list = []
        for col_id, col_info in enumerate(self.col_info_list):
            dist_list.append(self.get_distance(col_info.p, self.prob_norm(col_info.q)))
        return np.mean(dist_list)


def random_indices(sample_size, data_size):
    lst = [i for i in range(data_size)]
    random.shuffle(lst)
    return lst[:sample_size]

def greedy_method(sample: set, init_set_size=10, batch_size=100, num_batches=100, full_data_size=FULL_SIZE, sample_size=SAMPLE_SIZE):
    # R = set(random_indices(init_set_size, full_data_size))
    R = sample
    S = set(list(range(full_data_size))).difference(R)
    full_arr = np.load(FULL_DATA_PATH)
    evaluator = Evaluator(full_arr, init_indices=list(R))
    while True:
        try:
            score = 0.0
            # remove_num = None
            # for _, val in enumerate(tqdm(R)):
            #     cur_score = evaluator.try_([val], -1)
            #     if cur_score > score:
            #         score = cur_score
            #         remove_num = val
            origin_score = evaluator.score()
            print(origin_score)
            # R.remove(remove_num)
            # print(score)
            # pick_counter = 0
            while len(R) < sample_size:
                if len(R) < 500:
                    batch_size = 5
                    counter_max = 25
                elif len(R) < 1500:
                    batch_size = 1
                    counter_max = 1
                else:
                    batch_size = 1
                    counter_max = 1
                best_score = 999
                best_batch = None
                k = min(sample_size - len(R), batch_size)
                S_copy = S.copy()
                counter = 0
                while len(S_copy) != 0:
                    rnd = random.sample(S_copy, k)
                    B = list(rnd)
                    score = evaluator.try_(B)
                    if score < best_score:
                        best_score = score
                        best_batch = B
                    if score < origin_score or len(S_copy) < 180000:
                        break
                    S_copy.difference_update(B)
                print(f"|R| = {len(R)}, |S_copy| = {len(S_copy)}, update: ", best_score)
                evaluator.add(best_batch)
                R.update(best_batch)
                S.difference_update(best_batch)
                origin_score = best_score
            worst_score = 0.0
            worst_item = -1
            for i in tqdm(R):
                s = evaluator.try_([i], -1)
                if s > worst_score:
                    worst_score = s
                    worst_item = i
            R.remove(i)
            best_item = -1
            for i in tqdm(S):
                s = evaluator.try_([i])
                if s < origin_score:
                    best_item = i
                    print("updated score: ", s)
                    break
            if best_item != -1:
                R.add(best_item)
                S.add(worst_item)
                S.remove(best_item)
                evaluator.add([best_item])
            else:
                R.add(worst_item)
        except KeyboardInterrupt:
            print("capture ctrl-c interrupt... now save sample to pickle...")
            break
    return list(R)

if __name__ == '__main__':
    random.seed(SEED)
    np.random.seed(SEED)
    s = set()
    # with open('result.txt', 'r') as file:
    #     for r in file:
    #         s.add(int(r))
    with open('pickle_0.0013861048017905426', 'rb') as file:
        s = set(pickle.load(file))

    R = greedy_method(s, init_set_size=1, batch_size=1, num_batches=3000)
    print(list(R))
    with open('R.pickle.2', 'wb') as f:
        pickle.dump(R, f)
