import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ndcg_score
import pickle

class t2_evaluation:
    def __init__(self, file, label='label', positive_label=1, selfid='caseid' ):
        self.raw_df = pd.read_csv(file)
        self.selfid = selfid
        if self.selfid is not None:
            self.raw_df = self.raw_df.set_index(self.selfid)
        self._index = self.raw_df.index
        self.attrb = self.raw_df.drop([label], axis=1).columns

        self.label=label
        self.positive_label = positive_label

        self.__prob_arr = self.__ini_prob_array()

        self.__prob_arr_gt = self.__cal_prob(self.raw_df, self.__prob_arr)
        self.prob_arr_gt_final,self.rule_whole_data = self.__find_rule(self.raw_df, self.__prob_arr_gt)

        
    def evaluation(self, index_list):
        self.__prob_arr_sample_final, self.rule_sampled_data = self.get_rule(index_list)
        
        rank_gt = self.prob_arr_gt_final[:,1,:].flatten()
        rank_sample = self.__prob_arr_sample_final[:,1,:].flatten()
        
        rank_gt[rank_gt==-1] = 0
        rank_sample[rank_sample==-1] = 0
        
        return ndcg_score(np.asarray([rank_gt]), np.asarray([rank_sample]), k=61) 
        
    def __ini_prob_array(self):
        count_max_class = []
        for col in self.attrb:
            _classes = np.unique(self.raw_df[col])
            _classes = _classes[~np.isnan(_classes)]
            count_max_class.append(len(_classes))
            
        prob_arr = np.full(shape=(len(self.attrb), 2, max(count_max_class)+1), fill_value=float(-1))
        total_count = len(self.raw_df)

        for _index, _attrb in enumerate(self.attrb):
            _classes = np.unique(self.raw_df[_attrb])
            _classes = _classes[~np.isnan(_classes)]
            
            _length = len(_classes)
            prob_arr[_index][0][:_length] = _classes
        return prob_arr                           
                                           
    def __cal_prob(self, df, prob_arr):
        total_count = len(df)
        if total_count == 0:
            return prob_arr
        for _index, _attrb in tqdm(enumerate(self.attrb), total=len(self.attrb)):
            no_class = np.argwhere(prob_arr[_index][0] == -1)[0][0]
            for _class in range(no_class):
                cat_class = prob_arr[_index][0][_class]
                _count_all = len(df[(df[_attrb]==cat_class)])
                _count_positive = len(df[(df[_attrb]==cat_class)&(df[self.label]==self.positive_label)])
                if _count_all==0:
                    prob=0
                else:
                    prob = _count_positive / _count_all
                prob_arr[_index][1][_class] = prob 

        return prob_arr
                                
    def __find_rule(self, df_ori, prob_arr_ori):
        df = df_ori.copy()
        prob_arr = prob_arr_ori.copy()
        prob_arr_ranking = np.full(shape=prob_arr_ori.shape, fill_value=-1)
        count = 0
        rank = 61
        prev_length = 0
        flag=False
        rule = []
        while(len(df)>0):
            indice, category = np.unravel_index(prob_arr[:,1,:].argmax(), prob_arr[:,1,:].shape)

            prob_arr_ranking[indice, 1, category] = (rank - count) if (rank - count) >0 else 0

            condition = (df[self.attrb[indice]]==prob_arr[indice, 0, category])
            rows_to_drop = len(df[condition].index)
            df = df.drop(df[condition].index)
                
            print('attribute:', self.attrb[indice], 'category:', prob_arr[indice, 0, category], 'probability:', prob_arr[indice, 1, category], 'drop:', rows_to_drop, 'remain', len(df))
            rule.append([self.attrb[indice], prob_arr[indice, 0, category]])
            count += 1

            if len(df[(df[self.label]==self.positive_label)]) == 0:
                print(len(df[(df[self.label]!=self.positive_label)]),'negative data is remained')
                return prob_arr_ranking, rule
                
            prob_arr = self.__cal_prob(df, prob_arr)

            prev_length = len(df)
        return prob_arr_ranking, rule
    
    
    def get_gt_rule(self):
        return self.rule_whole_data
    
    def get_rule(self, index_list):
        index_list_new = [index for index in index_list if index in self._index]
        self.sample_df = self.raw_df.loc[index_list_new]
        self.sample_df = self.sample_df.dropna(how='all')

        prob_arr_sample = self.__cal_prob(self.sample_df , self.__prob_arr)
        prob_arr_sample_final, rule_sampled_data = self.__find_rule(self.sample_df, prob_arr_sample)
        return prob_arr_sample_final, rule_sampled_data

if __name__ == '__main__':
    t2 = t2_evaluation('t2-test-data.csv')
    with open('/tmp2/b07902075/t2_evaluation_test.pickle', 'wb') as f:
        pickle.dump(t2, f)