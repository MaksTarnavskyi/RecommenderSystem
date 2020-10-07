import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from tqdm import tqdm

# from implicit.bpr import BayesianPersonalizedRanking
# from implicit.approximate_als import NMSLibAlternatingLeastSquares
# from implicit.approximate_als import AnnoyAlternatingLeastSquares

import copy
import pickle
import sys
from multiprocessing import Pool

import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

class Implicit(object):

    def __init__(self):

        self.model = None
        self.mapped_trainset = None
        self.mapping_dict = None
        self.inv_mapping_dict = None
        self.max_index_of_item = None
        self.max_index_of_user = None

        self.item_users = None
        self.user_items = None

        self.k = 10

        self.param = None
        self.default_param = {'factors': 100, 'regularization':0.01, 'iterations':15,'use_native':True, 'use_cg':True, 'use_gpu':False,'calculate_training_loss':False, 'num_threads':0}

        self.mean_user_factors = None
        self.mean_item_factors = None

        self.baseline_recommend_items = None
        self.baseline_recommend_scores = None


    def fit_trainset(self, raw_train_dataset):
        trainset = copy.deepcopy(raw_train_dataset)
        #trainset = trainset.drop_duplicates(subset=['user','item'])

        self.mapping_dict, self.inv_mapping_dict = fit_coder(trainset, 'user', 'item', 'rating')
        self.mapped_trainset = code(copy.deepcopy(trainset), 'user','item','rating',self.mapping_dict)

        self.max_index_of_item = len(self.mapped_trainset.item.unique())
        self.max_index_of_user = len(self.mapped_trainset.user.unique())

        row = self.mapped_trainset.item.values
        col = self.mapped_trainset.user.values
        data = self.mapped_trainset.rating.values

        self.item_users = csr_matrix( (data,(row,col)), shape=(self.max_index_of_item,self.max_index_of_user))
        self.user_items = self.item_users.T.tocsr()
        self.user_items = bm25_weight(self.user_items, B=0.7).tocsr()*5
        self.item_users = self.user_items.T.tocsr()

        # #Experiment --------------
        # add_one = self.item_users.toarray() + 1
        # self.item_users = csr_matrix(add_one)
        # # -------------------------
        self.user_items = self.item_users.T.tocsr()

    def add_fit_trainset(self, new_raw_train_dataset):
        if self.mapped_trainset is None:
            self.fit_trainset(new_raw_train_dataset)
        else:
            new_trainset = copy.deepcopy(new_raw_train_dataset)
            new_train = code(copy.deepcopy(new_trainset), 'user','item','rating',self.mapping_dict)

            ind_item = new_train[new_train.item.isnull()].index
            ind_user = new_train[new_train.user.isnull()].index

            unknown_items = new_trainset.loc[ind_item,'item'].unique()
            unknown_users = new_trainset.loc[ind_user,'user'].unique()

            len_new_items = len(unknown_items)
            len_new_users = len(unknown_users)

            new_item_dic = {key: value for key, value in zip(unknown_items,range(self.max_index_of_item, self.max_index_of_item+len_new_items))}
            new_user_dic = {key: value for key, value in zip(unknown_users,range(self.max_index_of_user, self.max_index_of_user+len_new_users))}

            inv_new_item_dic = {value: key for key, value in zip(unknown_items,range(self.max_index_of_item, self.max_index_of_item+len_new_items))}
            inv_new_user_dic = {value: key for key, value in zip(unknown_users,range(self.max_index_of_user, self.max_index_of_user+len_new_users))}

            self.max_index_of_item += len_new_items
            self.max_index_of_user += len_new_users

            self.mapping_dict['item'].update(new_item_dic)
            self.mapping_dict['user'].update(new_user_dic)

            self.inv_mapping_dict['item'].update(inv_new_item_dic)
            self.inv_mapping_dict['user'].update(inv_new_user_dic)

            new_mapped_trainset = code(copy.deepcopy(new_raw_train_dataset), 'user','item','rating',self.mapping_dict)

            #self.mapped_trainset = self.mapped_trainset.append(new_trainset, ignore_index=True)

            self.mapped_trainset = pd.concat([self.mapped_trainset, new_mapped_trainset], ignore_index=True)
            self.mapped_trainset = self.mapped_trainset.drop_duplicates(subset=['user','item'])

            row = self.mapped_trainset.item.values
            col = self.mapped_trainset.user.values
            data = self.mapped_trainset.rating.values

            self.item_users = csr_matrix( (data,(row,col)), shape=(self.max_index_of_item,self.max_index_of_user))
            self.user_items = self.item_users.T.tocsr()

            if self.model:

                factors_for_new_unknown_users = [list(self.mean_user_factors)]*len_new_users
                if len(factors_for_new_unknown_users) > 0:
                    self.model.user_factors = np.concatenate([self.model.user_factors,factors_for_new_unknown_users])

                factors_for_new_unknown_items = [list(self.mean_item_factors)]*len_new_items
                if len(factors_for_new_unknown_items) > 0:
                    self.model.item_factors = np.concatenate([self.model.item_factors,factors_for_new_unknown_items])

                print('factors extended')


    def set_k(self, k):
        self.k = int(k)

    def fit_model(self, dic_param = {}, fit_new_model = True):
        if self.item_users is None:
            print('Firstly fit trainset')
        else:

            if fit_new_model == False:  #check if available previous model
                if not self.model:
                    fit_new_model = True

            if fit_new_model:
                d = copy.deepcopy(self.default_param)
                d.update(dic_param)
                self.param = d


            else:
                d = copy.deepcopy(self.param)
                d.update(dic_param)
                if d['factors'] != self.param['factors']:
                    print('different amount of facors! Previous: '+str(self.param['factors'])+'; Now: '+str(d['factors']) +'; Fit new model')
                    fit_new_model = True


            if fit_new_model:
                self.model = AlternatingLeastSquares(factors=d['factors'],regularization=d['regularization'],iterations=d['iterations'], use_native=d['use_native'],use_cg=d['use_cg'],use_gpu=d['use_gpu'],calculate_training_loss =d['calculate_training_loss'], num_threads=d['num_threads']) #dic_param
            else:
                previous_user_factors = self.model.user_factors
                previous_item_factors = self.model.item_factors
                self.model = AlternatingLeastSquares(factors=d['factors'],regularization=d['regularization'],iterations=d['iterations'], use_native=d['use_native'],use_cg=d['use_cg'],use_gpu=d['use_gpu'],calculate_training_loss =d['calculate_training_loss'], num_threads=d['num_threads']) #dic_param
                self.model.user_factors = previous_user_factors
                self.model.item_factors = previous_item_factors

            self.model.fit(self.item_users)

            self.mean_user_factors = self.model.user_factors.mean(axis=0)
            self.mean_item_factors = self.model.item_factors.mean(axis=0)

            scores = np.dot(self.model.item_factors,self.mean_user_factors)
            items = list(range(self.max_index_of_item))
            result = list(zip(scores, items))
            result.sort(reverse=True)
            recommend = np.array(result)

            self.baseline_recommend_items = [self.inv_mapping_dict['item'][int(item)] for _ , item in recommend]
            self.baseline_recommend_scores = [score for score , _ in recommend]


    def get_user_factors(self):
        real_user_factors = {}
        for i in range(self.max_index_of_user):
            real_user_factors[self.inv_mapping_dict['user'][i]] = list(self.model.user_factors[i])
        return real_user_factors

    def get_item_factors(self):
        real_item_factors = {}
        for i in range(self.max_index_of_item):
            real_item_factors[self.inv_mapping_dict['item'][i]] = list(self.model.item_factors[i])
        return real_item_factors


    def recommend_for_user(self, user_true_name, filter_already_liked_items = True, return_scores = False, recalculate_user=False):
        if self.mapping_dict is None:
            print('Firstly fit_trainset')
            return None
        if self.model is None:
            print('Firstly fit_model')
            return None

        if user_true_name in self.mapping_dict['user'].keys():
            user = self.mapping_dict['user'][user_true_name]
            rec = self.model.recommend(user, self.user_items, self.k, filter_already_liked_items=filter_already_liked_items, recalculate_user=recalculate_user)
            items = [self.inv_mapping_dict['item'][item] for item, _ in rec]
            scores = [score for _, score in rec]
            if return_scores:
                return items, scores
            else:
                return items
        else:
            items = self.baseline_recommend_items[:self.k]
            if return_scores:
                scores = self.baseline_recommend_scores[:self.k]
                return items, scores
            else:
                return items

    def recommend(self, users_list, filter_already_liked_items = True, return_scores = False, recalculate_user=False):
        if self.mapping_dict is None:
            print('Firstly fit_trainset')
            return None
        if self.model is None:
            print('Firstly fit_model')
            return None

        result_user_items = {}
        result_user_scores = {}

        for user_true_name in tqdm(users_list):
            if return_scores:
                items, scores = self.recommend_for_user(user_true_name, filter_already_liked_items, return_scores, recalculate_user)
                result_user_items[user_true_name] = items
                result_user_scores[user_true_name] = scores
            else:
                items = self.recommend_for_user(user_true_name, filter_already_liked_items, return_scores, recalculate_user)
                result_user_items[user_true_name] = items

        if return_scores:
            return result_user_items, result_user_scores
        else:
            return result_user_items

    def recommend_df(self, users_list, filter_already_liked_items = True, return_scores = False, column_names=['user', 'item','rating'], recalculate_user=False):
        if self.mapping_dict is None:
            print('Firstly fit_trainset')
            return None
        if self.model is None:
            print('Firstly fit_model')
            return None

        result = []

        for user_true_name in tqdm(users_list):
            user_column = [user_true_name]*int(self.k)
            if return_scores:
                items, scores = self.recommend_for_user(user_true_name, filter_already_liked_items, return_scores, recalculate_user)
                res = list(zip(user_column, items, scores))
            else:
                items = self.recommend_for_user(user_true_name, filter_already_liked_items, return_scores, recalculate_user)
                res = list(zip(user_column, items))
            result.extend(res)

        if return_scores:
            return pd.DataFrame(result, columns = column_names[:3])
        else:
            return pd.DataFrame(result, columns = column_names[:2])





    #
    # def rank_for_user(self, user):
    #     if self.max_index_of_user is None:
    #         print('Firstly fit_testset')
    #         return None
    #
    #     list_items = self.testset[self.testset.user == user].item
    #     items_to_rank = list_items[list_items < self.max_index_of_item].values
    #     items_to_end = list_items[list_items >= self.max_index_of_item].values
    #
    #     res = []
    #
    #     if user >= self.max_index_of_user:
    #         list_to_sort = []
    #         for item in items_to_rank:
    #             list_to_sort.append((round(self.item_value_counts[item]*0.001,3),item))
    #
    #         for item in items_to_end:
    #             list_to_sort.append((0,item))
    #
    #         list_to_sort.sort(reverse=True)
    #         res = [(t[1], t[0]) for t in list_to_sort]
    #     else:
    #         res = self.model.rank_items(user, self.user_items,selected_items=items_to_rank)
    #         for item in items_to_end:
    #             res.append((item, 0))
    #     return res
    #
    #
    #
    # def rank(self):
    #     if self.max_index_of_user is None:
    #         print('Firstly fit_testset')
    #         return None
    #
    #     result = pd.DataFrame(columns=['item','rating','user'])
    #
    #     users = list(self.testset.user.unique())
    #     for i in tqdm(range(len(users))):
    #         user = users[i]
    #         res = self.rank_for_user(user)
    #         df = pd.DataFrame(res, columns=['item','rating'])
    #         df['user'] = [user]*len(df)
    #
    #         result = pd.concat([result, df])
    #
    #     result = result[['user','item','rating']]
    #     output = code(copy.deepcopy(result), 'user','item','rating',self.inv_mapping_dict)
    #     output.index = range(len(output))
    #
    #     return output

    def dump_model(self, filename = 'dumped_file'):
        """
        Saving the model for further using.
        :param filename: str - path and name of file to save.
        :return:
        """
        if (self.model is None) | (self.mapped_trainset is None):
            print('Unable to dump model')
            print('Please firstly fit train dataset and train model')
        else:
            dump_obj = {'model': self.model,
                        'mapped_trainset': self.mapped_trainset,
                        'mapping_dict': self.mapping_dict,
                        'inv_mapping_dict': self.inv_mapping_dict,
                        'max_index_of_item': self.max_index_of_item,
                        'max_index_of_user': self.max_index_of_user,
                        'item_users': self.item_users,
                        'user_items': self.user_items,
                        'k': self.k,
                        'mean_user_factors': self.mean_user_factors,
                        'mean_item_factors': self.mean_item_factors,
                        'baseline_recommend_items': self.baseline_recommend_items,
                        'baseline_recommend_scores': self.baseline_recommend_scores,
                        'param': self.param
                        }
            pickle.dump(dump_obj, open(filename, 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
            print('Model has succesfuly been dumped!')



    def load_model(self,filename = 'dumped_file'):
        """
        Function to load ready to use, pre trained model from file.
        :param filename: str - path to the file with model
        :return: nothing
        """
        dump_obj = pickle.load(open(filename, 'rb'))
        self.model = dump_obj['model']
        self.mapped_trainset = dump_obj['mapped_trainset']
        self.mapping_dict = dump_obj['mapping_dict']
        self.inv_mapping_dict = dump_obj['inv_mapping_dict']
        self.max_index_of_item = dump_obj['max_index_of_item']
        self.max_index_of_user = dump_obj['max_index_of_user']
        self.item_users = dump_obj['item_users']
        self.user_items = dump_obj['user_items']
        self.k = dump_obj['k']
        self.mean_user_factors = dump_obj['mean_user_factors']
        self.mean_item_factors = dump_obj['mean_item_factors']
        self.baseline_recommend_items = dump_obj['baseline_recommend_items']
        self.baseline_recommend_scores = dump_obj['baseline_recommend_scores']
        self.param = dump_obj['param']

#---------Help functions-------------------------

def make_implicit(x, min_rating):
    if x >= min_rating:
        return 1
    else:
        return 0


def fit_coder(dataset, user_var_name, item_var_name, rating_var_name):
    """
        Function for fitting encoder based on dataset`s users and items, to transform them in numeric form
        :param dataset: pandas dataframe
        :param user_var_name: str - name of column of users,
        :param item_var_name: str - name of column of items,
        :param rating_var_name: str - name of column of ratings,
        :return:
        mapping_dict - used by code function to encode dataset in appropriate form,
        #         inv_mapping_dict - used by code function to decode dataset to row format
        """
    user_var_name = str(user_var_name)
    item_var_name = str(item_var_name)
    rating_var_name = str(rating_var_name)
    users = dataset[user_var_name].unique()
    items = dataset[item_var_name].unique()
    items_dict = {key: value for key, value in zip(items,range(0,len(items)))}
    users_dict = {key: value for key, value in zip(users,range(0,len(users)))}
    mapping_dict = {user_var_name:users_dict, item_var_name:items_dict}
    inv_map_user = {v: k for k, v in users_dict.items()}
    inv_map_item = {v: k for k, v in items_dict.items()}
    inv_mapping_dict = {user_var_name:inv_map_user, item_var_name:inv_map_item}
    return mapping_dict, inv_mapping_dict

def code(test, user_var_name, item_var_name, rating_var_name, mapping_dict):
    """
    Function for encoding/decoding dataset based mapping_dict
    It transforms dataset to friendly, appropriate for Recommender_surprise.py form or to the raw form
    :param test: pandas dataframe,
    :param user_var_name: str - name of column of users,
    :param item_var_name: str - name of column of items,
    :param rating_var_name: str - name of column of ratings,
    :param mapping_dict: - pretrained by fit_coder function dictionary that either encode(mapping dict) or decode(inv_mapping_dict) given dataset.
    :return: dataset - pandas dataframe in either raw or encoded format.
    """
    user_var_name = str(user_var_name)
    item_var_name = str(item_var_name)
    rating_var_name = str(rating_var_name)

    test[user_var_name] = test[user_var_name].map(mapping_dict[user_var_name])
    test[item_var_name] = test[item_var_name].map(mapping_dict[item_var_name])
    return test

def load_dataset(path_to_dataset, sep=',', names = ['user', 'item', 'rating', 'timestamp']):
    """
    Upload dataframe from .csv file in such a format:
    1) No header
    2) ',' separator
    3) columns in order user, item, rating
    :param path_to_dataset:
    :return: nothing
    """
    try:
        return pd.read_csv(path_to_dataset, sep=sep, names=names)
    except:
        print("Unable to upload dataset")
        print("Unexpected error:", sys.exc_info()[0])
        raise
