# -*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np
import json
import shutil

from app import app
from flask import render_template, send_from_directory
from flask import Flask, flash, request, redirect, url_for

from werkzeug.utils import secure_filename

from app.api_tools import *

from app.models.Recommender_implicit_new import Implicit
from app.models.metrics_new import Metrics

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode
import plotly.offline as py
from plotly import tools
import plotly.figure_factory as ff
init_notebook_mode(connected=True)

import colorlover as cl

import warnings
warnings.filterwarnings('ignore')

ALLOWED_EXTENSIONS = set(['csv'])

#----check if all required directories exists
folders = ['WORK']
for name in folders:
    path = app.config[name + '_FOLDER']
    os.system("if [ ! -d " + path + " ]; then mkdir -p " + path + "; fi")


save_path = app.config['PATH_SAVE']
working_path = app.config['WORK_FOLDER']

#---------Uploads saved data-------------

dic_folder = working_path

with open(dic_folder+'predict.json', 'r') as f:
    predict = json.load(f)

with open(dic_folder+'meta_dic.json', 'r') as f:
    meta_dic = json.load(f)

with open(dic_folder+'kinopoisk_dic.json', 'r') as f:
    kino_dic = json.load(f)

with open(dic_folder+'common_dic.json', 'r') as f:
    common_dic = json.load(f)

# with open(dic_folder+'dic_metric_for_each_user.json', 'r') as f:
#     dic_metric_for_each_user = json.load(f)

with open(dic_folder+'genre_dic.json', 'r') as f:
    genre_dic = json.load(f)

with open(dic_folder+'train_dic.json', 'r') as f:
    train_dic = json.load(f)

with open(dic_folder+'test_dic.json', 'r') as f:
    test_dic = json.load(f)

with open(dic_folder+'val_count_dic.json', 'r') as f:
    val_count_dic = json.load(f)

show_for_select_items = list(val_count_dic.keys())[:500]

df_metric = pd.read_csv(dic_folder+'df_res.csv')
df_metric['user_id'] = df_metric['user_id'].apply(int)
df_metric['user_id'] = df_metric['user_id'].apply(str)
for column in df_metric.columns[1:]:
    df_metric[column] = df_metric[column].apply(round_decimal, k=4)

start_src = "https://st.kp.yandex.net/images/film_iphone/iphone360_"
end_src = ".jpg"

k = 10

#-----------------------------------------------------
#--------Routes----------------------

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():

    return render_template('index.html', title='Home')

@app.route('/plot/<name>', methods=['GET', 'POST'])
def plot(name):
    path = 'Plots/' + str(name) +'.html'
    print(path)
    return render_template(path, title='Graphic')

@app.route('/metrics_page', methods=['GET', 'POST'])
def metrics_page():

    return render_template('metrics_page.html', title='Info')



@app.route('/metrics_for_user', methods=['GET', 'POST'])
def metrics_for_user():

    current_column = df_metric.columns[0]
    ascending = False

    return render_template('metrics_for_user.html', title='Metrics for user', df_metrics_per_user = df_metric, current_column=current_column, ascending=ascending)




@app.route('/user_demonstration', methods=['GET', 'POST'])
def user_demonstration():

    user_name = request.args['user_name']
    print(user_name)
    pred_items = predict[user_name]
    test_items = test_dic[user_name]
    train_items = []
    try:
        train_items = train_dic[user_name]
    except:
        pass
    common_items = common_dic[user_name]

    return render_template('user_demonstration.html', title='Demonstration', user_name=user_name,
        pred_items=pred_items, test_items=test_items, train_items=train_items, common_items=common_items, kino_dic =kino_dic, start_src=start_src, end_src=end_src)

@app.route('/item_page', methods=['GET', 'POST'])
def item_page():

    item_name = request.args['item_name']

    meta_info = meta_dic[item_name]
    genre_info = genre_dic[item_name]

    return render_template('item_page.html', title='Info',item_name=item_name, meta_info=meta_info,genre_info=genre_info, kino_dic=kino_dic, start_src=start_src, end_src=end_src)

@app.route('/select_items', methods=['GET', 'POST'])
def select_items():

    if request.method == 'POST':
        liked_items = []
        for item in show_for_select_items:
            if request.form.get(item):
                liked_items.append(item)
        print(liked_items)
        liked_items_dic = {'liked_items': liked_items}
        with open(save_path+'liked_items_dic.json', 'w') as f:
            json.dump(liked_items_dic, f)

        return redirect(url_for('waiting_training_model'))

    return render_template('select_items.html', title='Select films', items_list=show_for_select_items, kino_dic=kino_dic, start_src=start_src, end_src=end_src)


@app.route('/waiting_training_model', methods=['GET', 'POST'])
def waiting_training_model():

    return render_template('waiting_recommend.html', title='Waiting')

@app.route('/training_model', methods=['GET', 'POST'])
def training_model():

    with open(save_path+'liked_items_dic.json', 'r') as f:
        liked_items_dic = json.load(f)


    rec = Implicit()
    rec.load_model(save_path+"model_als")

    liked_items = liked_items_dic['liked_items']
    row_user = ['new_user']*len(liked_items)
    row_score = [5.]*len(liked_items)
    new_interactions = pd.DataFrame({'user':row_user, 'item':liked_items, 'rating': row_score})
    rec.add_fit_trainset(new_interactions)
    rec.set_k(k)
    # rec.param['iterations'] = 1
    # rec.fit_model(rec.param, fit_new_model=False)
    predict = rec.recommend(['new_user'], filter_already_liked_items=True, recalculate_user=True)
    with open(save_path+'predict_dic.json', 'w') as f:
        json.dump(predict, f)

    return redirect(url_for('user_recommendations'))

    # return render_template('index.html', title='Training model')

@app.route('/user_recommendations', methods=['GET', 'POST'])
def user_recommendations():

    with open(save_path+'liked_items_dic.json', 'r') as f:
        liked_items_dic = json.load(f)

    with open(save_path+'predict_dic.json', 'r') as f:
        predict_dic = json.load(f)

    liked_items = liked_items_dic['liked_items']
    pred_items = predict_dic['new_user']

    if request.method == 'POST':
        good_predict_items = []
        for item in pred_items:
            if request.form.get(item):
                good_predict_items.append(item)
        print(good_predict_items)
        good_predict_dic = {'new_user': good_predict_items}

        with open(save_path+'good_predict_dic.json', 'w') as f:
            json.dump(good_predict_dic, f)

        mt = Metrics(good_predict_dic, predict_dic)
        metrics_dic = mt.Evaluate(k=k, metrics_list=['Precision', 'MRR', 'mMAP', 'mNDCG'], to_pandas=False)
        with open(save_path+'rec_metrics_dic.json', 'w') as f:
            json.dump(metrics_dic, f)


        # row1 = [' ']
        # row1.extend(list(metrics_dic.keys()))
        # row2 = ['Metric']
        # values = list(metrics_dic.values())
        # values = [round(val, 4) for val in values]
        # row2.extend(values)

        # data_matrix = [row1, row2]

        metric_values = list(metrics_dic.items())
        for i in range(len(metric_values)):
            metric_values[i] = list(metric_values[i])
            metric_values[i][1] = round(metric_values[i][1], 4)

        data_matrix = [[' ', 'Metric']]
        data_matrix.extend(metric_values)
        colorscale = [[0, 'rgb(0,0,0,0)'],[.5, '#f2f2f2'],[1, '#ffffff']]
        table = ff.create_table(data_matrix, index=True,  colorscale=colorscale)

        plot_path = 'app/templates/Plots/'
        config={'showLink': False, 'displayModeBar': False}
        py.plot(table, filename='app/templates/Plots/rec_metric_table.html', config=config, auto_open=False)

        return redirect(url_for('user_rec_metric'))

    return render_template('user_recommendations.html', title='Recommended_items', liked_items=liked_items, pred_items=pred_items, kino_dic=kino_dic, start_src=start_src, end_src=end_src)

@app.route('/user_rec_metric', methods=['GET', 'POST'])
def user_rec_metric():

    with open(save_path+'liked_items_dic.json', 'r') as f:
        liked_items_dic = json.load(f)

    with open(save_path+'predict_dic.json', 'r') as f:
        predict_dic = json.load(f)

    with open(save_path+'good_predict_dic.json', 'r') as f:
        good_predict_dic = json.load(f)

    with open(save_path+'rec_metrics_dic.json', 'r') as f:
        metrics_dic = json.load(f)

    liked_items = liked_items_dic['liked_items']
    pred_items = predict_dic['new_user']
    good_pred_items = good_predict_dic['new_user']

    metric_values = list(metrics_dic.items())

    return render_template('user_rec_metric.html', title='Recommended_items', liked_items=liked_items, pred_items=pred_items,good_pred_items=good_pred_items,metric_values=metric_values, kino_dic=kino_dic, start_src=start_src, end_src=end_src)




#
# @app.route('/show_metrics_for_users', methods=['GET', 'POST'])
# def show_metrics_for_users():
#
#     current_column = ''
#     ascending = False
#
#     data = read_json(json_path)
#
#     path_to_result = working_path + data['train_dataset']['folder_name']+'/Models/' + data['model']['model_name'] + '/Validation_datasets/' + data['validation_dataset']['folder_name'] + '/Result'
#     metadata_path = working_path + data['train_dataset']['folder_name']+'/Metadata/'
#
#     try:
#         current_column = data['current_column']
#         ascending = data['ascending']
#     except:
#         pass
#
#
#
#     type = data['training_model_details']['type']
#     #cache = read_value(app.config['VALUES_FOLDER']+ 'need_cache.txt')
#
#     k = ''
#     if type=='recommend':
#         k = str(data['training_model_details']['k_for_recommend'])
#     else:
#         k = str(data['validation_dataset']['k_for_rank'])
#
#
#
#     df_metrics_per_user = pd.read_csv(path_to_result + '/metrics_per_user.csv')
#
#     columns = ['user', 'presicion@'+k,'recall@'+k, 'map@'+k, 'ndcg@'+k]
#     if type == 'rank':
#         columns = ['user', 'ndcg@'+k]
#     # for col in df_metrics_per_user.columns[::-1]:
#     #     columns.append(col)
#     #
#     # if data["training_model_details"]['type'] == 'recommend':  #swap precision and recall
#     #     t = columns[1]
#     #     columns[1] = columns[2]
#     #     columns[2] = t
#     #
#     # df_metrics_per_user = df_metrics_per_user.loc[:, columns]
#
#     df_metrics_per_user = df_metrics_per_user[columns]
#     for column in columns[1:]:
#         df_metrics_per_user[column] = df_metrics_per_user[column].apply(round_decimal, k=4)
#
#     have_metadata = False
#     try:
#
#         # if check_len(metadata_path):
#         #     have_metadata = True
#
#         if data['meta_dataset']:
#             have_metadata = True
#     except:
#         pass
#
#     if request.method == 'POST' or current_column:
#         try:
#             colum_name = request.form.get('sort_column')
#             if current_column == colum_name:
#                 ascending = not(ascending)
#             current_column = colum_name
#             data['current_column'] = current_column
#             data['ascending'] = ascending
#             write_json(json_path, data)
#             df_metrics_per_user = df_metrics_per_user.sort_values(current_column, ascending=ascending)
#         except:
#             pass
#             print('error')
#
#
#
#
#     return render_template('Result/show_metrics_for_users.html', title='show_metrics_for_userst', df_metrics_per_user=df_metrics_per_user, have_metadata=have_metadata, current_column=current_column, ascending=ascending)


#
#
# @app.route('/metadata', methods=['GET', 'POST'])
# def metadata():
#
#     user_name = ''
#     try:
#         user_name = request.args['user_name']
#     except:
#         pass
#
#     show = False
#     no_user = 'no'
#
#
#     data = read_json(json_path)
#     train_dataset_path = working_path + data['train_dataset']['folder_name']+'/Train_dataset/'+ data['train_dataset']['file_name']
#     metadata_path = working_path + data['train_dataset']['folder_name']+'/Metadata/'+data['meta_dataset']['folder_name']+'/'+ data['meta_dataset']['file_name']
#
#     validation_path = working_path + data['train_dataset']['folder_name'] + '/Validation_datasets/' + data['validation_dataset']['folder_name']
#     validation_dataset_path = validation_path +'/Validation_dataset/'+ data['validation_dataset']['file_name']
#
#     model_path = working_path + data['train_dataset']['folder_name']+'/Models/' + data['model']['model_name']
#     path_to_result = model_path + '/Validation_datasets/' + data['validation_dataset']['folder_name'] + '/Result'
#     #path_to_result = validation_path + '/Result'
#
#     model_name = data['model']['model_name']
#     model_type = data['model']['model_type']
#
#     show_model_name = show_true_model_type(model_type)
#
#     print(show_model_name)
#
#     type = data['training_model_details']['type']
#     base = ''
#     if data['training_model_details']['compare_with_BaseLine'] == 'yes':
#         base = 'BaseLine'
#
#     filename = ''
#     base_filename = ''
#     if type == 'rank':
#         filename = 'rank.csv'
#     else:
#         filename = 'recommendations.csv'
#
#
#
#
#     if user_name or request.method == 'POST' :
#
#         if user_name or request.form['btn'] == "Show result":
#             user = user_name
#             if not user_name:
#                 user = request.form['input-user-id']
#             train = pd.read_csv(train_dataset_path, names=['user','item','rating','timestamp'])
#             validation = pd.read_csv(validation_dataset_path, names=['user','item','rating','timestamp'])
#             predict = pd.read_csv(path_to_result +'/'+ filename, names=['user','item','rating'])
#             base_predict = []
#             if base:
#                 base_predict = pd.read_csv(path_to_result + '/base_' +filename, names=['user','item','rating'])
#             meta = pd.read_csv(metadata_path,names=['item', 'imUrl','title', 'description', 'price','brand'])
#             meta.index = meta.item
#
#             show=True
#             if type == 'recommend':
#                 train_items = users_items(user, train, meta).values
#                 validation_items = users_items(user, validation, meta).values
#                 recommend_items = users_items(user, predict, meta).values
#
#                 validation_items_names = []
#                 for item in validation_items:
#                     validation_items_names.append(item[0])
#
#                 recommend_items_names = []
#                 for item in recommend_items:
#                     recommend_items_names.append(item[0])
#
#                 common_items = set(recommend_items_names).intersection(set(validation_items_names))
#
#                 if len(recommend_items) < 1:
#                     no_user = 'yes'
#                 base_recommend_items = []
#                 if base:
#                     base_recommend_items = users_items(user, base_predict, meta).values
#
#                 return render_template('Result/metadata_new.html', title='metadata',type=type, show=show, no_user=no_user, train_items=train_items, validation_items=validation_items,recommend_items=recommend_items, base_recommend_items=base_recommend_items, user=user, base=base, show_model_name=show_model_name, common_items=common_items)
#
#             if type =='rank':
#                 user_validation = validation[validation.user == user].sort_values(by=['rating'],ascending=False)
#
#                 validation_items = users_items(user, user_validation, meta).values
#                 recommend_items = users_items(user, predict, meta).values
#                 if len(recommend_items) < 1:
#                     no_user = 'yes'
#                 base_recommend_items = []
#                 if base:
#                     base_recommend_items = users_items(user, base_predict, meta).values
#
#                 return render_template('Result/metadata_new.html', title='metadata',type=type, show=show, no_user=no_user, validation_items=validation_items,recommend_items=recommend_items, base_recommend_items=base_recommend_items, user=user, base=base, show_model_name=show_model_name)
#
#
#     return render_template('Result/metadata_new.html', title='metadata')
#
#
#
#
# @app.route('/item_page', methods=['GET', 'POST'])
# def item_page():
#     item_name = request.args['item_name']
#     item_image = request.args['item_image']
#     item_title = request.args['item_title']
#     item_description = request.args['item_description']
#     item_price = request.args['item_price']
#
#     return render_template('Result/item_page.html', title='item_page', item_name=item_name, item_image=item_image, item_title=item_title, item_description=item_description,item_price=item_price)
#
# @app.route('/license', methods=['GET', 'POST'])
# def license():
#
#     return render_template('License.html', title='License')
#
#
# #-----------------
# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
