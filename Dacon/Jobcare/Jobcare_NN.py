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

# +
df_train = pd.read_csv('Jobcare_data/train.csv', index_col='id')
df_test = pd.read_csv('Jobcare_data/test.csv', index_col='id')

code_d = pd.read_csv('Jobcare_data/속성_D_코드.csv', index_col='속성 D 코드')
code_h = pd.read_csv('Jobcare_data/속성_H_코드.csv', index_col='속성 H 코드')
code_l = pd.read_csv('Jobcare_data/속성_L_코드.csv', index_col='속성 L 코드')

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


# +
from scipy import sparse as ssp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score

from time import time
import datetime


# +
def round_pred(pred):
    return list(map(round, pred))

def eval_f1(preds, dtrain):
    labels = dtrain.get_label()
    preds = threshold(preds, th=0.5)
    return 'f1', f1_score(labels, preds), True

# return d_l/m/s_match_yn dataframes
def is_match(col_p, col_c, df, code):
    df_m = code.loc[df[col_p], :].reset_index() == code.loc[df[col_c], :].reset_index()
    df_m = df_m.add_prefix('match' + col_p[-1] + '_')
    return df_m

import copy

def threshold(pred, th=0.5):
    l = copy.deepcopy(pred)
    for i, p in enumerate(pred):
        if (p > th):
            l[i] = 1
        else:
            l[i] = 0
    return l


# -

# ## NN model

col_match

df_train.drop(col_cat + col_cnt + drop_features + col_bin + col_code, axis=1)

# +
'''# onehot_cat & cnt
count = 0
for col in col_cat + col_cnt:
    if count == 0 :
        onehot_cols = pd.get_dummies(df_train[col], prefix=col)
        count += 1
    else:
        onehot_cols = pd.concat([onehot_cols, pd.get_dummies(df_train[col], prefix=col)], axis=1)'''
        
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

# df_code   ####don't use####
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

df_train_cnt = pd.concat([match_cols, diff_e, df_match_code, count_cols,  df_new], axis=1)
df_train_cnt = df_train_cnt.astype('float32')

for col in df_train_cnt.columns: # min max scale
    col_max = df_train_cnt[col].max()
    col_min = df_train_cnt[col].min()
    df_train_cnt[col] = (df_train_cnt[col] - col_min) / (col_max - col_min)

df_train_cnt.info()

# +
'''# onehot_cat & cnt
count = 0
for col in col_cat + col_cnt:
    if count == 0 :
        onehot_cols = pd.get_dummies(df_test[col], prefix=col)
        count += 1
    else:
        onehot_cols = pd.concat([onehot_cols, pd.get_dummies(df_test[col], prefix=col)], axis=1)'''
        
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

# df_code #### don't use ####
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

df_test_cnt = pd.concat([match_cols, diff_e, df_match_code, count_cols,  df_new], axis=1)
df_test_cnt = df_test_cnt.astype('float32')

for col in df_test_cnt.columns: # min max scale
    col_max = df_train_cnt[col].max()
    col_min = df_train_cnt[col].min()
    df_test_cnt[col] = (df_test_cnt[col] - col_min) / (col_max - col_min)

df_test_cnt.info()
# -

df_train_emb = df_train.drop(drop_features + col_bin + col_code + ['target'], axis=1).astype('int32')
df_test_emb = df_test.drop(drop_features + col_bin + col_code, axis=1).astype('int32')
df_train_emb.info()

# +
from keras.layers import Dense, PReLU, BatchNormalization, Dropout, Embedding, Flatten, Input, Concatenate
from keras.models import Model

col_d = [x for x in col_code if '_d' in x]
col_h = [x for x in col_code if '_h' in x]
col_l = ['contents_attribute_l']

d_max = 1258
h_max = 314
l_max = 2025


def nn_model():
    inputs = []
    flatten_layers = []
    output_dim = 64
        
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
    
    # df_emb
    for c in df_train_emb.columns:
        input_c = Input(shape=(1, ), dtype='int32')
        num_c = df_train_emb[c].max()
        embed_c = Embedding(num_c, 8, input_length=1)(input_c)
        embed_c = Dropout(0.25)(embed_c)
        flatten_c = Flatten()(embed_c)
        inputs.append(input_c)
        flatten_layers.append(flatten_c)

    # df_cnt
    input_cnt = layers.Input(shape=(X_cnt_train.shape[1],), dtype='float32')
    inputs.append(input_cnt)
    flatten_layers.append(input_cnt)

    
    flatten = Concatenate()(flatten_layers)

    dense = Dense(512, kernel_initializer=keras.initializers.he_normal())(flatten)
    dense = PReLU()(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.75)(dense)

    dense = Dense(64, kernel_initializer=keras.initializers.he_normal())(dense)
    dense = PReLU()(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)

    outputs = Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='sigmoid')(dense)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

    return model


# -

y = df_train['target'].copy()
X_cnt_train = df_train_cnt.values
X_cnt_test = df_test_cnt.values
X_emb_train = df_train_emb.values
X_emb_test = df_test_emb.values
df_code_train = df_train.loc[:, col_d + col_h + col_l].copy().astype('int32')
df_code_test = df_test.loc[:, col_d + col_h + col_l].copy().astype('int32')

# +
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=112)

kf = kfold.split(X_cnt_train, y)

(t, v) = next(kf)

y_tr = y[t]
y_val = y[v]

x_tr = []
x_val = []

# code d h l
x_tr.append(df_train.loc[t, col_d].values.reshape(-1, 4))
x_tr.append(df_train.loc[t, col_h].values.reshape(-1, 4))
x_tr.append(df_train.loc[t, col_l].values.reshape(-1, 1))

x_val.append(df_train.loc[v, col_d].values.reshape(-1, 4))
x_val.append(df_train.loc[v, col_h].values.reshape(-1, 4))
x_val.append(df_train.loc[v, col_l].values.reshape(-1, 1))

# enb
for i in range(X_emb_train.shape[1]):
    x_tr.append(X_emb_train[t, i].reshape(-1, 1))
    x_val.append(X_emb_train[v, i].reshape(-1, 1))

# cnt
x_tr.append(X_cnt_train[t])
x_val.append(X_cnt_train[v])

model = nn_model()

history = model.fit(
    x = x_tr,
    y = y_tr,
    validation_data=[x_val, y_val],
    batch_size=512,
    epochs=20
)

# +
x_list = []

# code d h l
x_list.append(df_train[col_d].values.reshape(-1, 4))
x_list.append(df_train[col_h].values.reshape(-1, 4))
x_list.append(df_train[col_l].values.reshape(-1, 1))

# enb
for i in range(X_emb_train.shape[1]):
    x_list.append(X_emb_train[:, i].reshape(-1, 1))

# cnt
x_list.append(X_cnt_train)

pred = np.squeeze(model.predict(x_list))

for i in range(30, 50):
    th = i/100
    print(th, f1_score(y, threshold(pred, th=th)))
    
#7216 64 64
#7211 128 16(l)
#7175 64 16(l)
#7224 64 16


#7096 16 10 6415
#7140 32 10 6404
#7222 64 10 6416
#7328 128 10 6411 overfit
#7461 256 10 6402 overfit
#7604 512 10 overfit

# +
x_test = []

# code d h l
x_test.append(df_test.loc[:, col_d].values.reshape(-1, 4))
x_test.append(df_test.loc[:, col_h].values.reshape(-1, 4))
x_test.append(df_test.loc[:, col_l].values.reshape(-1, 1))

# enb
for i in range(X_emb_train.shape[1]):
    x_test.append(X_emb_test[:, i].reshape(-1, 1))

x_test.append(X_cnt_test)

pred_test = np.squeeze(model.predict(x_test))
df_pred = pd.DataFrame(pred_test, columns=['target'])
df_pred.index.name = 'id'
df_pred
# -

df_pred['target'] = threshold(df_pred['target'], th=0.39)
df_pred

df_pred.to_csv('nn_64_20.csv')

df_pred[df_pred['target'] == 1].count()

df_pred[df_pred['target'] < 0.4].count()

# +

df_pred.to_csv('pred_256.csv')
# -

a = pd.read_csv('sub_th.csv', index_col='id')
a.describe()







# +
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=112)

kf = kfold.split(X_cnt_train, y)

(t, v) = next(kf)




NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=112)

kf = kfold.split(X, y)

num_seeds = 1
cv_train = np.zeros(len(df_train))
cv_pred = np.zeros(len(df_test))

x_test_list = []
for i in range(X_cat.shape[1]):
  x_test_list.append(X_cat_test[:, i].reshape(-1, 1))
x_test_list.append(X_cnt_test)

for s in range(num_seeds):
    
    np.random.seed(s)

    for (t, v) in kfold.split(X_cat, y):
        y_tr = y[t]
        y_val = y[v]

        x_tr = []
        x_val = []

        # code d h l
        x_tr.append(df_train.loc[t, col_d].values.reshape(-1, 4))
        x_tr.append(df_train.loc[t, col_h].values.reshape(-1, 4))
        x_tr.append(df_train.loc[t, col_l].values.reshape(-1, 1))

        x_val.append(df_train.loc[v, col_d].values.reshape(-1, 4))
        x_val.append(df_train.loc[v, col_h].values.reshape(-1, 4))
        x_val.append(df_train.loc[v, col_l].values.reshape(-1, 1))

        # enb
        for i in range(X_emb_train.shape[1]):
        x_tr.append(X_emb_train[t, i].reshape(-1, 1))
        x_val.append(X_emb_train[v, i].reshape(-1, 1))

        # cnt
        x_tr.append(X_cnt_train[t])
        x_val.append(X_cnt_train[v])

        model = nn_model()

        model.fit(
        x = x_tr,
        y = y_tr,
        validation_data=[x_val, y_val],
        batch_size=512,
        epochs=20
        )
        
        val_pred = np.squeeze(model.predict(xval_cat_list))
        cv_train[v] += val_pred
        print("accuracy: {}".format(f1_score(y[v], threshold(val_pred, th=0.5))))

        cv_pred += np.squeeze(model.predict(x_test_list))

    print("-----------------")
    print("seed{}_mse: {}".format(s, f1_score(y, threshold(cv_train / (s + 1), th=0.5))))
    print("-----------------")
