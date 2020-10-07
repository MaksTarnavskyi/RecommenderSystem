import os
import time
import pandas as pd
import numpy as np
import json

#----------------------------------
#-------Functions------------------

def read_json(json_path):
    data = {}
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    return data

def write_json(json_path, data):
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent = 4)

def check_path(path):
    os.system("if [ ! -d " + path + " ]; then mkdir -p " + path + "; fi")

def get_folders(path, hide_folders=[]):
    check_path(path)
    all_folders = os.listdir(path)
    show_folders = []
    for folder in all_folders:
        if folder not in hide_folders:
            show_folders.append(folder)

    return show_folders

def check_len(my_array):
    if len(my_array) >0:
        return True
    else:
        return False

def show_true_model_type(model_type):
    if model_type == "Model 1":
        return "SVD"
    if model_type == "Model 2":
        return "BMF"
    if model_type == "Model 3":
        return "DMF"
    if model_type == "Model 4":
        return "ALS"
    if model_type == "Model 5":
        return "LightFM"

def get_visible_folders(path, hide_folders=['.ipynb_checkpoints']):
    show_folders = get_folders(path,hide_folders)
    if 'visible_files.json' in show_folders:
        path_to_visible_files = path + 'visible_files.json'
        name_files = read_json(path_to_visible_files)
        not_show = name_files['not_to_show'] + ['visible_files.json']
        show_folders = get_folders(path,hide_folders + not_show)

    return show_folders

def show_time(time):
    time=float(time)
    minutes = int(time / 60) % 60
    seconds = int(time % 60)
    hours = int(time /(60 * 60))


def users_items(user_id, rating_tab, meta_tab):
    items_to_show = rating_tab[rating_tab.user == user_id].item.values
    discover = meta_tab.loc[items_to_show,:]
    return discover


def get_df_info(path):
    """
    Function that returns main information of dataset.
    :l_r - amount of rows in dataset (ratings)
    :l_u - amount of users in dataset
    :l_i - amount of items in dataset
    :spar - Sparcity of dataset (the ratio of non-empty values to all the possible values in the table  uses-items)
    :df_head - first 5 rows of dataset
    :dist_rating - array for creating bar chart using css, shows distribution of ratings in dataset
    :am_r_per_user - array of main values about distribution of ratings in dataset (min, max,mean and others)
    """
    data = pd.read_csv(path,names=['user','item','rating','timestamp'])
    if data['timestamp'].isnull().all():
        data = data.drop('timestamp', axis=1)

    if data.loc[0,'user'] == 'user':
        print('del_header')
        data = data.loc[1:, :]
    print('start_save_without_headers')
    data.to_csv(path, index=None, header=None)
    print('saved')
    #print('')
    l_r = len(data)
    l_u = len(data.user.unique())
    l_i = len(data.item.unique())
    spar = round_decimal(float(l_r)/(l_i*l_u), 7)
    df_head = data.head()
    dist_rat = distribution_rating(data)
    am_r_per_user = r_per_user(data)
    df_info= [['Size',l_r],['Unique users',l_u],['Unique items',l_i],['Density', spar]]
    return df_head, df_info, dist_rat, am_r_per_user

def r_per_user(data):
    """
    Counting main values about distribution of ratings in dataset (min, max,mean and others)
    :param: data - pandas dataframe of dataset with ratings
    :returns: array of main values
    """
    ratings_per_user = data['user'].value_counts().describe()
    values = ratings_per_user.values[1:]
    indexes = ratings_per_user.index[1:]
    names = [str(ind) for ind in indexes ]
    for i in range(len(values)):
        values[i] = round_decimal(values[i], 2)
    for i in range(3,6):
        #names[i] = names[i] + ' - quantile'
        names[i] = 'quantile - ' + names[i]

    return list(zip(names,values))

def make_rec_list(df, user):
    """
    Creating recommendation list for current user from result dataset
    :param df - pandas dataframe of result dataset with recommendations
    :param user - string, name of user in dataset
    :returns: array of recommendations in format [place in list, name of item]
    """
    df['user'] = df['user'].apply(str)
    rec_list = df[df.user == user].loc[:,['item']]
    val = rec_list.values
    ind = range(1, len(val) +1)
    return list(zip(ind,val[:,0]))

def make_rec_list_with_base(df, user, base_df = [],check = False):
    """
    Creating 2 recommendation lists for current user from results of selected model and baseline model
    :param df - pandas dataframe with results from selected model
    :param user - string, name of user in dataset
    :param base_df - pandas dataframe with results from baseline model
    :returns: array of recommendations in format [name of item]
    """
    df['user'] = df['user'].apply(str)
    rec_list = df[df.user == user].loc[:,['item']]

    val = rec_list.values
    ind = range(1, len(val) +1)
    result = []
    if check:
        base_df['user'] = base_df['user'].apply(str)
        base_rec_list = base_df[base_df.user == user].loc[:,['item']]
        base_val = base_rec_list.values
        for i in range(min(len(val),len(base_val))):
            result.append([i+1,val[i,0], base_val[i,0]])
        return result
    else:
        return list(zip(ind,val[:,0]))


def round_decimal(x, k):
    """
    :param x - number wich we round
    :param k - how many digits after comma
    :return - number with k digits after comma
    """
    x = float(x)
    r = 10.**k
    x = round(x*r) / r
    return x

def read_value(path):
    """
    Read value from txt file
    :path - string, path to the file
    :return - string, value
    """
    name_file = open(path, "r")
    value = name_file.read()
    name_file.close()
    return value

def write_value(path, value):
    """
    Write value into txt file
    :path - string, path to the file
    :value - string
    :return - nothing
    """
    name_file = open(path, "w")
    name_file.write(value)
    name_file.close()

def distribution_rating(df):
    """
    Creating array wich help us create bar chart of distribution of ratings
    :df - pandas dataframe, our train or test dataset
    :return - array in format ['rating','amount','percentage in dataset']
    """
    val = df['rating'].value_counts()
    l = len(df)
    amount = val.values
    rating = val.index
    result_array = []
    for i in range(len(val)):
        result_array.append([rating[i],amount[i], round(float(amount[i])/l*100)])
    result_array.sort()
    return result_array

def show_ranking_df(test,result,user):
    """
    Make dataframe to show how selected model rank items for current user and compare it with true order
    :test - pandas dataframe, our test dataset
    :result - pandas dataframe, result of selected model
    :user - string, user name in dataset
    :return - array in format ['rating','amount of this rating','percentage in dataset']
    """

    test['user'] = test['user'].apply(str)
    result['user'] = result['user'].apply(str)

    my_result = result[result.user == user] # dataframe from result only with current user
    items = my_result.item.values

    my_test = test[test.user == user].loc[test['item'].isin(items)] # dataframe from tets only with current user

    sort_test = my_test.sort_values(by=['rating'],ascending=False)
    sort_test['position_in_test'] = list(range(1,len(sort_test)+1))
    sort_result = my_result.sort_values(by=['rating'],ascending=False)
    sort_result['position_in_result'] = list(range(1,len(sort_result)+1))
    df = sort_test.merge(sort_result, on='item')

    return df.loc[:,['item','position_in_test', 'position_in_result']].values

def show_ranking_df_with_base(test,result,user,base = [],check = False):
    """
    Make table to show how selected model rank items for current user and compare it with true order and baseline model
    :test - pandas dataframe, our test dataset
    :result - pandas dataframe, result of selected model
    :user - string, user name in dataset
    :return - array in format ['rating','amount of this rating','percentage in dataset']
    """

    test['user'] = test['user'].apply(str)
    result['user'] = result['user'].apply(str)

    my_result = result[result.user == user]
    items = my_result.item.values

    my_test = test[test.user == user].loc[test['item'].isin(items)]
    sort_test = my_test.sort_values(by=['rating'],ascending=False)
    sort_test['position_in_test'] = list(range(1,len(sort_test)+1))
    sort_result = my_result.sort_values(by=['rating'],ascending=False)
    sort_result['position_in_result'] = list(range(1,len(sort_result)+1))

    df = sort_test.merge(sort_result, on='item')
    if check:
        base['user'] = base['user'].apply(str)
        my_base = base[base.user == user]
        sort_base = my_base.sort_values(by=['rating'],ascending=False)
        sort_base['position_in_base'] = list(range(1,len(sort_base)+1))
        df = df.merge(sort_base, on='item')
        return df.loc[:,['item','position_in_test', 'position_in_result', 'position_in_base']].values
    return df.loc[:,['item','position_in_test', 'position_in_result']].values


def show_metric_res(ev):
    """
    Creating table to show metric values for result of current model
    :ev - dictionary, which gives evaluator function in metrics (results of metrics)
    :return - array in format ['name of metric','value of metric']
    """
    try_metric = list(ev.items())
    metric = []
    for row in try_metric:
        row = list(row)
        row[1] = round_decimal(row[1], 3)
        s = str(row[1])
        row[1] = s[:5]
        metric.append(row)
    return metric

def show_metric_res_with_base(ev, base_ev):
    """
    Creating table to show metric values for result of current model and baseline model
    :ev - dictionary, which is result of evaluator function in metrics (results of metrics)
    :return - array in format ['name of metric','value of metric']
    """
    try_metric = list(ev.items())
    base_try_metric = list(base_ev.items())
    metric = []
    for i in range(len(try_metric)):
        row = try_metric[i]
        base_row = base_try_metric[i]

        row = list(row)
        base_row = list(base_row)
        row[1] = round_decimal(row[1], 3)
        base_row[1] = round_decimal(base_row[1], 3)

        metric.append([row[0],row[1],base_row[1]])
    return metric
