import pandas as pd
import numpy as np
import copy

class Metrics():
    def __init__(self, test, predict, calculate_rating_metrics = False, verbose = False):

        users_in_test = set()
        users_in_predict = set()
        self.test_item = {}
        self.predict_item = {}
        self.type_predict = ''

        if type(test) == type(pd.DataFrame()):
            test[['user', 'item']] = test[['user', 'item']].astype(str)
            users_in_test = set(test.user.unique())
            self.test_item = test.groupby(['user'])['item'].apply(lambda grp: list(grp)).to_dict()
            if verbose:
                print('type of test: pandas')

        if type(predict) == type(pd.DataFrame()):
            predict[['user', 'item']] = predict[['user', 'item']].astype(str)
            users_in_predict = set(predict.user.unique())
            self.predict_item = predict.groupby(['user'])['item'].apply(lambda grp: list(grp)).to_dict()
            self.type_predict = 'pandas'
            if verbose:
                print('type of predict: pandas')


        if type(test) == type({}):
            self.test_item = test
            users_in_test = set(list(test.keys()))
            if verbose:
                print('type of test: dict')

        if type(predict) == type({}):
            self.predict_item = predict
            users_in_predict = set(list(predict.keys()))
            self.type_predict = 'dict'
            if verbose:
                print('type of predict: dict')


        if type(list(self.test_item.keys())[0]) != type(list(self.predict_item.keys())[0]):
            print('ERROR: different type for users in dict')

        if type(list(self.test_item.values())[0][0]) != type(list(self.predict_item.values())[0][0]):
            print('ERROR: different type for items in dict')

        self.users = list(users_in_test.intersection(users_in_predict))
        if verbose:
            print('amount of users in test:', len(users_in_test))
            print('amount of users in predict:', len(users_in_predict))
            print('amount of common users:', len(self.users))


        self.metrics_for_recommend = ['Precision', 'Recall', 'mMAP', 'MAP', 'NDCG', 'mNDCG', 'MRR']
        self.metrics_for_rating = ['RMSE', 'MAE', 'MSE']

        self.available_metrics =  copy.copy(self.metrics_for_recommend)

        #for Normed Entropy
        if self.type_predict == 'pandas':
            self.available_metrics.append('Normed_entropy')
            self.proportions = (predict.item.value_counts() / predict.shape[0]).sort_values().values
            self.num_predicted_items = len(predict.item.unique())

        self.saved_result = {}

        # use rating metrics
        self.calculate_rating_metrics = calculate_rating_metrics
        if calculate_rating_metrics and type(test) == type(pd.DataFrame()) and type(predict) == type(pd.DataFrame()):
            test.rating = test.rating.astype(float)
            predict.rating = predict.rating.astype(float)
            self.available_metrics.extend(self.metrics_for_rating)
            self.test_rating = test.groupby(['user'])['rating'].apply(lambda grp: list(grp)).to_dict()
            self.predict_rating = predict.groupby(['user'])['rating'].apply(lambda grp: list(grp)).to_dict()
            self.common_rating_test = {}
            self.common_rating_predict = {}
            for user in self.users:
                test_item = self.test_item[user]
                predict_item = self.predict_item[user]
                common_items= list(set(test_item).intersection(set(predict_item)))

                mask = np.isin(test_item, common_items)
                self.common_rating_test[user] = np.array(self.test_rating[user])[mask]

                mask = np.isin(predict_item, common_items)
                self.common_rating_predict[user] = np.array(self.predict_rating[user])[mask]



    def Precision(self, k, metric_per_user=False):

        return self.calculate_metric_for_recommend(self.precision_for_user, k, metric_per_user)


    def Recall(self, k, metric_per_user=False):

        return self.calculate_metric_for_recommend(self.recall_for_user, k, metric_per_user)

    def MAP(self, k, metric_per_user=False):

        return self.calculate_metric_for_recommend(self.map_for_user, k, metric_per_user)

    def mMAP(self, k, metric_per_user=False):

        return self.calculate_metric_for_recommend(self.map_for_user, k, metric_per_user, modified_metric = True)

    def NDCG(self, k, metric_per_user=False):

        return self.calculate_metric_for_recommend(self.ndcg_for_user, k, metric_per_user, modified_metric = False)

    def mNDCG(self, k, metric_per_user=False):

        return self.calculate_metric_for_recommend(self.ndcg_for_user, k, metric_per_user, modified_metric = True)

    def MRR(self, k, metric_per_user=False):

        return self.calculate_metric_for_recommend(self.mrr_for_user, k, metric_per_user)

    #### Shannon Entropy ####
    def Normed_entropy(self):
        '''
        The Normed Entropy is 0 when a single item is always chosen or recommended,
        and 1 when n items are chosen or recommended equally often.
        '''
        if self.type_predict == 'dict':
            print('Type of predict was dict')
            print('Sorry, metric in development')
            return 0
        proportions = self.proportions
        x_logx = np.vectorize(lambda x: x * np.log2(x))
        n = self.num_predicted_items

        return - x_logx(proportions).sum() / np.log2(n)


    def calculate_metric_for_recommend(self, metric_function, k, metric_per_user=False, modified_metric = False):
        user_metric = {}
        sum_metric_for_all_users = 0.

        for user in self.users:

            metric = 0.
            if modified_metric:
                metric = metric_function(user, k, modified_metric)
            else:
                metric = metric_function(user, k)

            user_metric[user] = metric
            sum_metric_for_all_users += metric

        result_metric = sum_metric_for_all_users/len(self.users)

        if metric_per_user:
            return result_metric, user_metric
        else:
            return result_metric


    def precision_for_user(self, user, k):

        return len(set(self.test_item[user]).intersection(set(self.predict_item[user])))/float(k)

    def recall_for_user(self, user, k):
        true_items = self.test_item[user]

        return len(set(true_items).intersection(set(self.predict_item[user])))/float(len(true_items))

    def map_for_user(self, user, k, modified_metric = False):
        true_items = self.test_item[user]
        predicted_items = self.predict_item[user]
        tru_predicted = 0.
        precisions = np.zeros(k)
        for i in range(k):
            if predicted_items[i] in true_items:
                tru_predicted += 1
                precisions[i] = tru_predicted/(i+1)

        average_precision = 0.
        if modified_metric:
            l = len(true_items)
            average_precision = sum(precisions)/min(k, l)

        else:
            average_precision = sum(precisions)/k

        return average_precision


    def ndcg_for_user(self, user, k, modified_metric = False):
        true_items = self.test_item[user]
        predicted_items = self.predict_item[user]

        idcg = 0.

        l = k
        if modified_metric:
            l = min(len(true_items), k)

        for i in range(l):
            idcg += 1./np.log2(i+2)

        dcg = 0.
        for i in range(k):
            if predicted_items[i] in true_items:
                dcg += 1./np.log2(i+2)
        ndcg = dcg/idcg
        return ndcg

    def mrr_for_user(self, user, k , modified_metric = False):
        true = self.test_item[user]
        pred = self.predict_item[user]
        relevant_items = [1 if elem in true else 0 for elem in pred]
        if sum(relevant_items) == 0:
            return 0.
        else:
            index_of_first_relevant = relevant_items.index(1)
            return 1. / (index_of_first_relevant +1)




    def calculate_metric_for_rating(self, metric_function, metric_per_user=False):

        if self.calculate_rating_metrics == False:
            print('calculate_rating_metrics == False')
            if metric_per_user:
                return -1 , {}
            else:
                return -1

        user_metric = {}
        sum_metric_for_all_users = 0.

        for user in self.users:
            real_ratings = self.common_rating_test[user]
            predicted_ratings = self.common_rating_predict[user]

            metric = 0.
            if len(real_ratings) > 1:
                metric = metric_function(real_ratings, predicted_ratings)
            user_metric[user] = metric
            sum_metric_for_all_users += metric

        result_metric = sum_metric_for_all_users/len(self.users)

        if metric_per_user:
            return result_metric, user_metric
        else:
            return result_metric


    def RMSE(self, metric_per_user=False):

        return self.calculate_metric_for_rating(self.rmse_for_user, metric_per_user)

    def MAE(self, metric_per_user=False):

        return self.calculate_metric_for_rating(self.mae_for_user, metric_per_user)

    def MSE(self, metric_per_user=False):

        return self.calculate_metric_for_rating(self.mse_for_user, metric_per_user)


    def rmse_for_user(self, real_ratings, predicted_ratings):

        return np.sqrt(np.mean((real_ratings - predicted_ratings)**2))

    def mse_for_user(self, real_ratings, predicted_ratings):

        return np.mean((real_ratings - predicted_ratings)**2)

    def mae_for_user(self, real_ratings, predicted_ratings):

        return np.mean(np.absolute(real_ratings - predicted_ratings))


    def Evaluate(self, k, metrics_list = ['Precision', 'Recall', 'MAP', 'NDCG'], metric_per_user=False, verbose=False, to_pandas = True):
        result_dic = {}
        result_user_dic = {}
        for metric_name in metrics_list:
            if metric_name in self.available_metrics:
                if metric_name in self.metrics_for_recommend:
                    metric_value = -1
                    metric_user = {}

                    if metric_per_user:
                        metric_value, metric_user = getattr(self, metric_name)(k, metric_per_user)
                    else:
                        metric_value = getattr(self, metric_name)(k)

                    result_dic[metric_name +'@'+ str(k)] = metric_value
                    result_user_dic[metric_name +'@'+ str(k)] = metric_user
                    if verbose:
                        print(metric_name +'@'+ str(k) + '= ', metric_value)

                if metric_name in self.metrics_for_rating:
                    metric_value = -1
                    metric_user = {}

                    if metric_per_user:
                        metric_value, metric_user = getattr(self, metric_name)(metric_per_user)
                    else:
                        metric_value = getattr(self, metric_name)()

                    result_dic[metric_name] = metric_value
                    result_user_dic[metric_name] = metric_user

                    if verbose:
                        print(metric_name + '= ', metric_value)

                if metric_name == 'Normed_entropy':
                    metric_value = getattr(self, metric_name)()
                    result_dic[metric_name] = metric_value
                    if verbose:
                        print(metric_name + '= ', metric_value)
            else:
                print('Not available metric : ', metric_name)

        self.saved_result.update(result_dic)
        result = result_dic
        if to_pandas:
            df = pd.DataFrame(columns=list(result_dic.keys()))
            df.loc[0,:] = list(result_dic.values())
            result = df

        if metric_per_user:
            return result, result_user_dic
        else:
            return result
