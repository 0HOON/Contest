# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# 잡케어 추천 알고리즘 경진대회
# ===================
# https://dacon.io/competitions/official/235863/overview/description
#
# 한국고용정보원에서 일자리를 탐색중인 구직자에게 개인별 데이터를 기반으로 맞춤형 서비스 및 컨텐츠를 추천하는 모델을 만들고자 한다. 

# ## 1. 데이터 살펴보기

# +
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='cividis')
# -

df_train = pd.read_csv('Jobcare_data/train.csv', index_col='id')
df_test = pd.read_csv('Jobcare_data/test.csv', index_col='id')
df_train.head()

# +
col_cat = ['person_attribute_a',  'person_prefer_c', 
           'contents_attribute_i', 'contents_attribute_a',
           'contents_attribute_j_1', 'contents_attribute_j', 'contents_attribute_c', 
           'contents_attribute_k', 'contents_attribute_m']

col_bin = ['d_l_match_yn', 'd_m_match_yn', 'd_s_match_yn',
            'h_l_match_yn', 'h_m_match_yn', 'h_s_match_yn']

col_cnt = ['person_attribute_a_1', 'person_attribute_b', 'person_prefer_e', 'contents_attribute_e']

col_code = ['person_prefer_d_1', 'person_prefer_d_2', 'person_prefer_d_3',
            'person_prefer_h_1', 'person_prefer_h_2', 'person_prefer_h_3',
            'contents_attribute_l', 'contents_attribute_d', 'contents_attribute_h',
            'person_rn', 'contents_rn']

col_match = ['person_attribute_a', 'person_prefer_c', 'person_prefer_e']

drop_features = ['person_prefer_f', 'person_prefer_g', 'contents_open_dt']

# -

df_train.describe()

df_train.loc[:,["d_l_match_yn", "d_m_match_yn", "d_s_match_yn", "person_prefer_d_1", "person_prefer_d_2", "person_prefer_d_3", "contents_attribute_d"]]

df_train.loc[:,["h_l_match_yn", "h_m_match_yn", "h_s_match_yn", "person_prefer_h_1", "person_prefer_h_2", "person_prefer_h_3", "contents_attribute_h"]]

# contents_attribute의 D는 세분류, H는 중분류 코드이다.

df_train.info()

df_train[df_train['_s_match_yn']].describe()

df_train['target'].describe()

df_train.columns[0:-2]


# +
def bar_plot(col, data, hue=None):
    f, ax =  plt.subplots(figsize=(10, 5))
    sns.countplot(x=col, hue=hue, data=data, alpha=0.5)
    plt.plot()
    
col_cat = ['person_attribute_a',  'person_prefer_c', 'person_prefer_f', 
           'person_prefer_g', 'contents_attribute_i', 'contents_attribute_a',
           'contents_attribute_j_1', 'contents_attribute_j', 'contents_attribute_c', 
           'contents_attribute_k', 'contents_attribute_m']

col_cnt = ['person_attribute_a_1', 'person_attribute_b', 'person_prefer_e', 'contents_attribute_e']

for col in col_cat:
    bar_plot(col, df_train)
    
# -

for col in col_cat:
    bar_plot(col, df_test)

# person_prefer_g, person_prefer_f의 경우 test, train data의 모든 값이 1로 통일되어있는 의미 없는 변수이다.

for col in col_cnt:
    bar_plot(col, df_train)

for col in col_cnt:
    bar_plot(col, df_test)

# cnt_col의 경우 순서의 차이가 의미가 있다고 명시된 feature들이므로 이 feature들간의 차이를 새로운 feature로 넣는 것도 좋을 것으로 보인다.

col_cat.remove('person_prefer_g')
col_cat.remove('person_prefer_f')

# +
col_code = ['d_l_match_yn', 'd_m_match_yn', 'd_s_match_yn',
            'h_l_match_yn', 'h_m_match_yn', 'h_s_match_yn',
            'person_prefer_d_1', 'person_prefer_d_2', 'person_prefer_d_3',
            'person_prefer_h_1', 'person_prefer_h_2', 'person_prefer_h_3',
            'contents_attribute_l', 'contents_attribute_d', 'contents_attribute_h',
            'person_rn', 'contents_rn'
           ]

for col in col_code:
    print("\n{}".format(col))
    print(len(df_train[col].unique()))
# -

# ~match_yn 의 이름을 가진 feature는 회원 선호 속성 D, H의 1번만을 기준으로한다. 2, 3번과 맞는지도 확인해 feature로 넣어도 좋을 것 같다.

# ## 2. light gbm 모델 만들어보기

# +
from scipy import sparse as ssp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score

from time import time
import datetime


# -

def round_pred(pred):
    return list(map(round, pred))


def eval_f1(preds, dtrain):
    labels = dtrain.get_label()
    preds = threshold(preds, th=0.5)
    return 'f1', f1_score(labels, preds), True


# lgb 모델에서 사용할 f1 평가함수 정의.

# ### 2-1. 데이터에 별다른 처리 하지 않기

# +
drop_features = ['person_prefer_f', 'person_prefer_g', 'contents_open_dt']

X = df_train.drop(drop_features, axis=1)
y = X.pop('target')
X = np.squeeze(X.values)
y = np.squeeze(y.values)
X_test = df_test.drop(drop_features, axis=1)
X_test = np.squeeze(X_test.values)
# -

params = {"objective": "binary",
         "boosting_type": 'gbdt',
         "learning_rate": 0.1,
         "num_leaves": 15,
         "max_bin": 256,
         "feature_fraction": 0.6,
         "verbosity": 0,
         "drop_rate": 0.1,
         "is_unbalance": False,
         "max_drop": 50,
         "min_child_samples": 10,
         "min_child_weight": 150,
         "min_split_gain": 0,
         "subsample": 0.9,
         "early_stopping_rounds": 100,
         "device_type": 'gpu'
         }

# K-Fold 방식으로 여러번 학습시키기.

# +
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=112)
num_boost_round = 10000


final_cv_train = np.zeros(len(df_train))
final_cv_pred = np.zeros(len(df_test))

begin_time = time()

for s in range(3):

    cv_train = np.zeros(len(df_train))
    cv_pred = np.zeros(len(df_test))

    params['seed'] = s
        
    best_trees = []
    fold_scores = []
    
    kf = kfold.split(X, y)
    
    for i, (t, v) in enumerate(kf):
        X_train, X_val, y_train, y_val = X[t, :], X[v, :], y[t], y[v]
        ds_train = lgb.Dataset(X_train, y_train)
        ds_val = lgb.Dataset(X_val, y_val)
        
        bst = lgb.train(
            params, 
            ds_train, 
            num_boost_round,
            valid_sets=ds_val,
            feval=eval_f1,
            verbose_eval=100,

        )
        
        best_trees.append(bst.best_iteration)
        
        cv_train[v] += bst.predict(X_val)
        cv_pred += bst.predict(X_test, num_iteration=bst.best_iteration)

        score = f1_score(y_val, round_pred(cv_train[v]))
        print(score)
        fold_scores.append(score)
    
    cv_pred /= NFOLDS
    final_cv_pred += cv_pred

    final_cv_train += cv_train
    
    print("cv score:")
    print(f1_score(y, round_pred(cv_train)))
    print("{} score:".format(s + 1), f1_score(y, round_pred(final_cv_train / (s + 1))))
    print(fold_scores)
    print(best_trees, np.mean(best_trees))
    print(str(datetime.timedelta(seconds=time() - begin_time)))
# -

final_cv_pred/3

df_cv = pd.DataFrame(round_pred(final_cv_pred/3), columns=['target'])
df_cv.index.name = 'id'

y

df_cv.to_csv('sub_0.csv')

# ### 2-2 One hot encoding & match / diff

# f1
# - onehot 0.644~
# - match_code 0.6425
# - count_cat 0.6276
# - match_code + count_cat 0.6315
# - match_code + count_cat/cnt 0.6449
# - match_code + count_cnt 0.6445
# - match_code + count_cat/cnt/code 0.6527
# - match_code + count_cat/cnt/code + match_sum 0.6535
# - match_code + count_cat/cnt/code + match_sum + cnt onehot 0.6568
# - match_code + count_cat/cnt/code + match_sum + target cnt 0.6534
# - match_code + count_cat/cnt/code + match_sum + target cnt + new 0.6575
# - match_code + count_cat/cnt/code + match_sum + target cnt + new - below 10 0.6584
# - match_code + count_cat/cnt/code + match_sum + target cnt + new - below 20 0.6566

code_d = pd.read_csv('Jobcare_data/속성_D_코드.csv', index_col='속성 D 코드')
code_h = pd.read_csv('Jobcare_data/속성_H_코드.csv', index_col='속성 H 코드')
code_l = pd.read_csv('Jobcare_data/속성_L_코드.csv', index_col='속성 L 코드')
code_l.head()


# return d_l/m/s_match_yn dataframes
def is_match(col_p, col_c, df, code):
    df_m = code.loc[df[col_p], :].reset_index() == code.loc[df[col_c], :].reset_index()
    df_m = df_m.add_prefix(col_p[-1])
    return df_m


# +
import copy

def threshold(pred, th=0.5):
    l = copy.deepcopy(pred)
    for i, p in enumerate(pred):
        if (p > th):
            l[i] = 1
        else:
            l[i] = 0
    return l


# +
# onehot_cat & cnt
count = 0
for col in col_cat + col_cnt:
    if count == 0 :
        onehot_cols = pd.get_dummies(df_train[col], prefix=col)
        count += 1
    else:
        onehot_cols = pd.concat([onehot_cols, pd.get_dummies(df_train[col], prefix=col)], axis=1)
        
# match_cols
count = 0
for col in col_match:
    df = pd.DataFrame(df_train[col] == df_train['contents_attribute_{}'.format(col[-1])], columns=['match_{}'.format(col[-1])])
    if count == 0 :
        match_cols = df
        count += 1
    else:
        match_cols = pd.concat([match_cols, df], axis=1)
        
# diff_e
diff_e = pd.DataFrame(abs(df_train['contents_attribute_e'] - df_train['person_prefer_e']), columns=['diff_e'])

# match_code
col_d_p = [x for x in col_code if 'd' in x and 'person' in x]
col_h_p = [x for x in col_code if 'h' in x and 'person' in x]

match_code = []
for col in col_d_p:
    match_code.append(is_match(col, 'contents_attribute_d', df_train, code_d))
    
for col in col_h_p:
    match_code.append(is_match(col, 'contents_attribute_h', df_train, code_h))


df_match_code = pd.concat(match_code, axis=1)

df_match_code['match_sum'] = df_match_code.sum(axis=1) # match_ sum

# df_code
col_d = col_d_p + ['contents_attribute_d']
col_h = col_h_p + ['contents_attribute_h']

df_code_train = []

for col in col_d:
    df_code_train.append(code_d.loc[df_train[col], :].reset_index().add_prefix(col + '_'))
    
for col in col_h:
    df_code_train.append(code_h.loc[df_train[col], :].reset_index().add_prefix(col + '_'))
    
df_code_train.append(code_l.loc[df_train['contents_attribute_l'], :].reset_index().add_prefix('contents_attribute_l_'))

df_code_train = pd.concat(df_code_train, axis=1)

# count_cols
count_cols = pd.DataFrame()
for col in col_cnt + col_cat + col_code[-2:]: 
    u = df_train[col].value_counts().to_dict()
    count_cols[col + '_count'] = df_train[col].map(lambda x: u.get(x, 0))
    print(col)

for col in df_code_train.columns:
    u = df_code_train[col].value_counts().to_dict()
    count_cols[col + '_count'] = df_code_train[col].map(lambda x:u.get(x, 0))
    print(col)

# df_new
person_features = [c for c in df_train.columns if ('person_a' in c or 'person_p' in c)]
contents_features = [c for c in df_train.columns if 'contents_a' in c]
df_new = pd.DataFrame()
count = 0
for col in person_features:
    if count == 0:
        df_new['person_new'] = df_train[col].astype(str) + '_'
        count = 1
    else:
        df_new['person_new'] += df_train[col].astype(str) + '_'

count = 0
for col in contents_features:
    if count == 0:
        df_new['contents_new'] = df_train[col].astype(str) + '_'
        count = 1
    else:
        df_new['contents_new'] += df_train[col].astype(str) + '_'        

u_person = df_new['person_new'].value_counts().to_dict()
u_contents = df_new['contents_new'].value_counts().to_dict()

df_new['person_new'] = df_new['person_new'].map(lambda x: u_person.get(x, 0))
df_new['contents_new'] = df_new['contents_new'].map(lambda x: u_contents.get(x, 0))


df_train_onehot = df_train.drop(col_cat + col_cnt + drop_features + col_bin + col_code[:-2], axis=1)
df_train_onehot = pd.concat([df_train_onehot, onehot_cols, match_cols, diff_e, df_match_code, df_code_train, count_cols,  df_new], axis=1)
df_train_onehot = df_train_onehot.astype('float32')
df_train_onehot.info()

# +
# onehot_cat & cnt
count = 0
for col in col_cat + col_cnt:
    if count == 0 :
        onehot_cols = pd.get_dummies(df_test[col], prefix=col)
        count += 1
    else:
        onehot_cols = pd.concat([onehot_cols, pd.get_dummies(df_test[col], prefix=col)], axis=1)
        
# match_cols
count = 0
for col in col_match:
    df = pd.DataFrame(df_test[col] == df_test['contents_attribute_{}'.format(col[-1])], columns=['match_{}'.format(col[-1])])
    if count == 0 :
        match_cols = df
        count += 1
    else:
        match_cols = pd.concat([match_cols, df], axis=1)

        
# diff_e
diff_e = pd.DataFrame(abs(df_test['contents_attribute_e'] - df_test['person_prefer_e']), columns=['diff_e'])

col_d = [x for x in col_code if 'd' in x and 'person' in x]
col_h = [x for x in col_code if 'h' in x and 'person' in x]

# match_code
match_code = []
for col in col_d:
    match_code.append(is_match(col, 'contents_attribute_d', df_test, code_d))
    
for col in col_h:
    match_code.append(is_match(col, 'contents_attribute_h', df_test, code_h))

df_match_code = pd.concat(match_code, axis=1)

df_match_code['match_sum'] = df_match_code.sum(axis=1) # match_ sum

# df_code
col_d = col_d_p + ['contents_attribute_d']
col_h = col_h_p + ['contents_attribute_h']

df_code_test = []

for col in col_d:
    df_code_test.append(code_d.loc[df_test[col], :].reset_index().add_prefix(col + '_'))
for col in col_h:
    df_code_test.append(code_h.loc[df_test[col], :].reset_index().add_prefix(col + '_'))
df_code_test.append(code_l.loc[df_test['contents_attribute_l'], :].reset_index().add_prefix('contents_attribute_l_'))
df_code_test = pd.concat(df_code_test, axis=1)


# count_cols
count_cols = pd.DataFrame()
for col in col_cnt + col_cat + col_code[-2:]: 
    u = df_train[col].value_counts().to_dict()
    count_cols[col + '_count'] = df_test[col].map(lambda x: u.get(x, 0))
    print(col)

for col in df_code_test.columns:
    u = df_code_train[col].value_counts().to_dict()
    count_cols[col + '_count'] = df_code_test[col].map(lambda x:u.get(x, 0))
    print(col)

# df_new

person_features = [c for c in df_train.columns if ('person_a' in c or 'person_p' in c)]
df_new = pd.DataFrame()
count = 0
for col in person_features:
    if count == 0:
        df_new['person_new'] = df_test[col].astype(str) + '_'
        count = 1
    else:
        df_new['person_new'] += df_test[col].astype(str) + '_'
        
count = 0
for col in contents_features:
    if count == 0:
        df_new['contents_new'] = df_test[col].astype(str) + '_'
        count = 1
    else:
        df_new['contents_new'] += df_test[col].astype(str) + '_'        

df_new['person_new'] = df_new['person_new'].map(lambda x: u_person.get(x, 0))
df_new['contents_new'] = df_new['contents_new'].map(lambda x: u_contents.get(x, 0))

df_test_onehot = df_test.drop(col_cat + col_cnt + drop_features + col_bin + col_code[:-2],  axis=1)
df_test_onehot = pd.concat([df_test_onehot, onehot_cols, match_cols, diff_e, df_match_code, df_code_test, count_cols, df_new], axis=1)
df_test_onehot = df_test_onehot.astype('float32')
df_test_onehot.info()
# -

df_fi = pd.read_csv('feature_importance_5.csv', index_col=0)
not_imp_fea = df_fi.loc[df_fi['importance'] < 15, 'name'].values
len(not_imp_fea)

df_train_onehot.drop(not_imp_fea, axis=1, inplace=True)
df_test_onehot.drop(not_imp_fea, axis=1, inplace=True)

y = df_train_onehot.pop('target')
X = ssp.csr_matrix(df_train_onehot.values)
X_test = ssp.csr_matrix(df_test_onehot.values)

params = {"objective": "binary",
         "boosting_type": 'gbdt',
         "learning_rate": 0.1,
         "num_leaves": 31,
         "max_depth": -1,
         "max_bin": 256,
         "feature_fraction": 0.6,
         "verbosity": -1,
         "drop_rate": 0.1,
         "is_unbalance": False,
         "max_drop": 50,
         "min_child_samples": 2000,
         "min_child_weight": 150,
         "min_split_gain": 0,
         "subsample": 0.9,
         "early_stopping_rounds": 150,
         "device_type": 'gpu'
         }

# +
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=112)
num_boost_round = 10000

kf = kfold.split(X, y)

(t, v) = next(kf)

X_train, X_val, y_train, y_val = X[t, :], X[v, :], y[t], y[v]
ds_train = lgb.Dataset(X_train, y_train)
ds_val = lgb.Dataset(X_val, y_val)

record = {}
record_eval = lgb.record_evaluation(record)
early = lgb.early_stopping(100)

bst = lgb.train(
            params, 
            ds_train, 
            num_boost_round,
            valid_sets=ds_val,
            feval=eval_f1,
            verbose_eval=100,

        )

print('best score: {}'.format(bst.best_score))
# 0.701287
# -

pred = bst.predict(X)
for i in range(30, 50):
    print(i/100, f1_score(y, threshold(pred, th=i/100)))
#0.4 0.7271 num_leaves 31 max_depth -1

df_fi = pd.DataFrame(bst.feature_importance(), columns=['importance'])
df_fi['name'] = df_train_onehot.columns
df_fi = df_fi.sort_values('importance', ascending=False)
df_fi.to_csv('feature_importance.csv')

# +
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=112)
num_boost_round = 10000


final_cv_train = np.zeros(len(df_train))
final_cv_pred = np.zeros(len(df_test))

begin_time = time()

for s in range(6):

    cv_train = np.zeros(len(df_train))
    cv_pred = np.zeros(len(df_test))

    params['seed'] = s
        
    best_trees = []
    fold_scores = []
    
    kf = kfold.split(X, y)
    
    for i, (t, v) in enumerate(kf):
        X_train, X_val, y_train, y_val = X[t, :], X[v, :], y[t], y[v]
        ds_train = lgb.Dataset(X_train, y_train)
        ds_val = lgb.Dataset(X_val, y_val)
        
        bst = lgb.train(
            params, 
            ds_train, 
            num_boost_round,
            valid_sets=ds_val,
            feval=eval_f1,
            verbose_eval=100,

        )
        
        best_trees.append(bst.best_iteration)
        
        cv_train[v] += bst.predict(X_val)
        cv_pred += bst.predict(X_test, num_iteration=bst.best_iteration)

        score = f1_score(y_val, threshold(cv_train[v], th=0.36))
        print(score)
        fold_scores.append(score)
        print(str(datetime.timedelta(seconds=time() - begin_time)))
    
    cv_pred /= NFOLDS
    final_cv_pred += cv_pred

    final_cv_train += cv_train
    
    print("cv score:")
    print(f1_score(y, threshold(cv_train, th=0.36)))
    print("{} score:".format(s + 1), f1_score(y, threshold(final_cv_train / (s + 1), th=0.36)))
    print(fold_scores)
    print(best_trees, np.mean(best_trees))
    print(str(datetime.timedelta(seconds=time() - begin_time)))
# -
df_final_pred = pd.DataFrame(final_cv_pred, columns=['target'])
df_final_pred.index.name = 'id'
df_final_pred.to_csv('th_5_l_test.csv')
df_final_train = pd.DataFrame(final_cv_train, columns=['target'])
df_final_train.index.name = 'id'
df_final_train.to_csv('th_5_l_train.csv')

for i in range(30, 50):
    th = i/100
    
    print(i, f1_score(y, np.squeeze(threshold(final_cv_train/6, th=th))))

final_cv_pred

df_th5 = pd.DataFrame(np.squeeze(threshold(final_cv_pred/6, th=0.36)), columns=['target'])
df_th5.index.name = 'id'
df_th5

df_th5.to_csv('sub_th5_l_code.csv')

nn_cv_pred = pd.read_csv('nn_cv_pred.csv', index_col='id')
nn_cv_train = pd.read_csv('nn_cv_train.csv', index_col='id')

final_cv_pred / 6

final_en_train = (final_cv_train / 6) * 0.5  + (np.squeeze(nn_cv_train.values)) * 0.5
final_en_train

for i in range(30, 50):
    th = i/100
    print(th, f1_score(y, threshold(final_en_train, th=th)))

final_en_pred = (final_cv_pred / 6) * 0.5  + (np.squeeze(nn_cv_pred.values) / 5 ) * 0.5
final_en_pred

sub_en_pred = pd.DataFrame(threshold(final_en_pred, th=0.37), columns=['target'])
sub_en_pred.index.name = 'id'
sub_en_pred.to_csv('sub_en.csv')

# ### 2-3 embeding

from keras.layers import Dense, PReLU, BatchNormalization, Dropout, Embedding, Flatten, Input, Concatenate
from keras.models import Model

code_d

# +
col_d = [x for x in col_code if '_d' in x]
col_h = [x for x in col_code if '_h' in x]
col_l = ['contents_attribute_l']

d_max = 1258
h_max = 314
l_max = 2025

output_dim = 32

inputs = []
flatten_layers = []

# code d
input_d = Input(shape=(4, ), dtype='int32')
embed_d = Embedding(d_max, output_dim, input_length=4)(input_d)
drop_d = Dropout(0.25)(embed_d)
flatten_d = Flatten()(drop_d)
inputs.append(input_d)
flatten_layers.append(flatten_d)

# code h
input_h = Input(shape=(4, ), dtype='int32')
embed_h = Embedding(h_max, output_dim, input_length=4)(input_h)
drop_h = Dropout(0.25)(embed_h)
flatten_h = Flatten()(drop_h)
inputs.append(input_h)
flatten_layers.append(flatten_h)

# code l
input_l = Input(shape=(1, ), dtype='int32')
embed_l = Embedding(l_max, output_dim, input_length=1)(input_l)
drop_l = Dropout(0.25)(embed_l)
flatten_l = Flatten()(drop_l)
inputs.append(input_l)
flatten_layers.append(flatten_l)

flatten = Concatenate()(flatten_layers)

output = Dense(1, activation='sigmoid')(flatten)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

# -

df_emb_train = df_train.loc[:, col_d + col_h + col_l].copy().astype('int32')
df_emb_test = df_test.loc[:, col_d + col_h + col_l].copy().astype('int32')
df_emb.info()

# +
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=112)

kf = kfold.split(X, y)

num_seeds = 1
cv_train = np.zeros(len(df_train_onehot))
cv_pred = np.zeros(len(df_test_onehot))

x_test_list = []
x_test_list.append(df_emb_test.loc[:, col_d].values.reshape(-1, 4))
x_test_list.append(df_emb_test.loc[:, col_h].values.reshape(-1, 4))
x_test_list.append(df_emb_test.loc[:, col_l].values.reshape(-1, 1))


for s in range(num_seeds):
    
    np.random.seed(s)

    for (t, v) in kfold.split(df_emb_train.values, y):
        
        ytr = y[t]       
        yval = y[v]

        xtr_list, xval_list = [], []

        xtr_list.append(df_emb_train.loc[t, col_d].values.reshape(-1, 4))
        xtr_list.append(df_emb_train.loc[t, col_h].values.reshape(-1, 4))
        xtr_list.append(df_emb_train.loc[t, col_l].values.reshape(-1, 1))
        
        xval_list.append(df_emb_train.loc[v, col_d].values.reshape(-1, 4))
        xval_list.append(df_emb_train.loc[v, col_h].values.reshape(-1, 4))
        xval_list.append(df_emb_train.loc[v, col_l].values.reshape(-1, 1))

        model.fit(
            x = xtr_list,
            y = ytr,
            validation_data=[xval_list, yval],
            batch_size=512,
#            verbose=0,
            epochs=20
        )
        
        val_pred = np.squeeze(model.predict(xval_list))
        cv_train[v] += val_pred
        print("f1: {}".format(f1_score(y[v], threshold(val_pred, th=0.5))))

        cv_pred += np.squeeze(model.predict(x_test_list))

    print("-----------------")
    print("seed{}_f1: {}".format(s, f1_score(y, threshold(cv_train / (s + 1), th=0.5))))
    print("-----------------")


# -

# ## 3. NN model

def mk_Dataset(X, y):
    ds_ = tf.data.Dataset.from_tensor_slices((X.values.astype(float), y.values.astype(float)))

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    ds = (
        ds_
        .batch(32)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )
    
    return ds


def Dense_model():
    
    model = keras.Sequential([
        layers.Dense(512, input_shape=[43], kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='relu')

    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    
    return model


def f1_metrics(y_true, y_pred):


df_train_onehot.info()

# +
y = df_train_onehot.pop('target')
X_cnt = ssp.csr_matrix(df_train_cnt.values)
X_cat = ssp.csr_matrix(df_train_cat.values)

X_cnt_test = ssp.csr_matrix(df_test_cnt.values)
X_cat_test = ssp.csr_matrix(df_test_cat.values)

# +
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, Add

inputs = []
flatten_layers = []

for e, c in enumerate(emb_col):
  input_c = Input(shape=(1, ))
  num_c = df_train_cat[c].max()
  embed_c = Embedding(output_dim=num_c, input_dim=6, input_length=1)(input_c)
  embed_c = Dropout(0.25)(embed_c)
  flatten_c = Flatten()(embed_c)
  inputs.append(input_c)
  flatten_layers.append(flatten_c)

input_cnt = layers.Input(shape=(X_cnt.shape[1],), dtype='float64')
flatten_layers.append(input_cnt)
inputs.append(input_cnt)

flatten = layers.add(flatten_layers)

model = keras.Sequential([
        layers.Dense(512, kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.75),

        layers.Dense(64, kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='sigmoid')

    ])(flatten)
    
model.compile(optimizer='adam', loss="binary_crossentropy")

NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=112)

kf = kfold.split(X, y)

(t, v) = next(kf)

ds_train = mk_Dataset(df_train_onehot.loc[t, :], y[t])
ds_val = mk_Dataset(df_train_onehot.loc[v, :], y[v])

early = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

history = model.fit(
        ds_train,
        validation_data=ds_val,
#        verbose=0,
        callbacks=[early],
        epochs=20
    )
# -



# +
from sklearn.model_selection import StratifiedKFold

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

num_seeds = 5
cv_train = np.zeros(len(df_train_target))
cv_pred = np.zeros(len(df_test_target))

early = keras.callbacks.EarlyStopping(patience=300, restore_best_weights=True)

for s in range(num_seeds):
    
    np.random.seed(s)

    for (tr_idx, te_idx) in kfold.split(df_train_target, df_train_y):
        ds_train = mk_Dataset(df_train_target.loc[tr_idx, :], df_train_y.loc[tr_idx])
        ds_test = mk_Dataset(df_train_target.loc[te_idx, :], df_train_y.loc[te_idx])
        
        model = Dense_model()
        
        model.fit(
            ds_train,
            validation_data=ds_test,
            verbose=0,
            callbacks=[early],
            epochs=2000
        )
        
        cv_train[te_idx] += np.squeeze(model.predict(df_train_target.loc[te_idx, :].values.astype(float)))
        print("mse: {}".format(get_mse( cv_train[te_idx] / (s + 1) , df_train_y[te_idx]) ))

        cv_pred += np.squeeze(model.predict(df_test_target.values.astype(float)))

    print("-----------------")
    print("seed{}_mse: {}".format(s, get_mse(cv_train / (s + 1), df_train_y)))
    print("-----------------")
