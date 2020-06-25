import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

from evaluation_t1 import get_count_dict

need_process = ['iRPOB', 'iRemplpar', 'dAncstry2']

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="./t1-data.csv")
    args = parser.parse_args()
    return args

def main():
    args = parse()
    df = pd.read_csv(args.data, index_col=0)
    for col_name in need_process:
        col = df[col_name]
        count_df = col.groupby(col).count()
        cnt = np.zeros(count_df.shape[0])
        feat_to_ind = dict()

        for i in range(count_df.shape[0]):
            feat_to_ind[count_df.index[i]] = i

        for i in tqdm(range(col.shape[0])):
            col.iloc[i] = feat_to_ind[col.iloc[i]]

    np.save('t1-data-preprocessed.npy', df.to_numpy().astype('uint8'))

if __name__ == '__main__':
    main()
