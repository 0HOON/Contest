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

# 펭귄 몸무게 예측 경진대회(2)
# =======
# https://dacon.io/competitions/official/235862/overview/description
#
# 9개의 feature로 펭귄의 몸무게를 예측하는 모델을 만들어 성능을 겨루는 대회이다. 성능은 rmse로 평가한다. 이전에 참가했던 심장 질환 예측 경진대회와 유사한 유형의 자료 및 목표이지만 이번엔 Classification이 아니라 Regression 모델을 만든다.

# ## 모든 feature 이용해보기

# Body Mass와 관련 없어 보였던 Island, Clutch Completion, Delta 15 N, 13 C 등의 feature를 제거했던 것이 성능의 저하를 가져왔던 것은 아닐까? 모든 feature를 이용해보자.
# train, test 데이터를 불러와 결측치를 제거하고, target encoding 및 min max scaling도 적용해준다.

# +
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='cividis')

# +
df_train = pd.read_csv("./train.csv", index_col="id")
df_test = pd.read_csv("./test.csv", index_col="id")

df_train.isna().sum()
# -

df_test.isna().sum()

df_train.dropna(subset = ["Sex", "Delta 15 N (o/oo)", "Delta 13 C (o/oo)"], inplace=True)
df_train.isna().sum()


def outlier_idx(df, col, weight=1.5):
    q_25 = np.percentile(df[col], 25)
    q_75 = np.percentile(df[col], 75)
    
    iqr = q_75 - q_25
    
    lower_bound = q_25 - iqr
    upper_bound = q_75 + iqr
    
    o_idx = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
    
    return o_idx


cols = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Body Mass (g)", "Delta 15 N (o/oo)", "Delta 13 C (o/oo)"]
for col in cols:
    outlier_idx_ = outlier_idx(df_train, col)
    df_train.drop(outlier_idx_, inplace=True)
df_train.describe()

df_train

df_train.reset_index(drop=True, inplace=True)
df_train.index.name = "id"

# +
cat_cols = ["Species", "Island", "Sex", "Clutch Completion"]

df_train_target = df_train.copy()

for col in cat_cols:
    print("\n\n{}".format(col))
    for u in df_train[col].unique():
        cat_idx = df_train_target[df_train_target[col] == u].index
        cat_mean = df_train_target[df_train_target[col] == u].loc[:, "Body Mass (g)"].mean()
        df_train_target.loc[cat_idx, col] = cat_mean
        print("\n{}: {}".format(u, cat_mean))    

df_train_target.head()


# +
def min_max_scale(df, col):
    col_min = df[col].min()
    col_max = df[col].max()
    df[col] = ( df[col] - col_min) / (col_max - col_min)

cnt_cols = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Delta 15 N (o/oo)", "Delta 13 C (o/oo)"]

for col in cnt_cols:
    min_max_scale(df_train_target, col)

body_min = df_train["Body Mass (g)"].min()
body_max = df_train["Body Mass (g)"].max()

for col in cat_cols:
    df_train_target[col] = (df_train_target[col] - body_min) / (body_max - body_min)
    df_train_target[col] = df_train_target[col].astype(float)
df_train_target.head()
# -

df_train_y = df_train_target.pop("Body Mass (g)")
df_train_y_rescaled = (df_train_y - body_min) / (body_max - body_min)

train_n = int(df_train_target.count()[0]*0.8)

# ### test set

# +
df_test_target = df_test.copy()

for col in cat_cols:
    for u in df_train[col].unique():
        cat_idx = df_test_target[df_test_target[col] == u].index
        cat_mean = df_train[df_train[col] == u].loc[:, "Body Mass (g)"].mean()
        df_test_target.loc[cat_idx, col] = cat_mean

df_test_target.head()

# +
for col in cnt_cols:
    col_max = df_train[col].max()
    col_min = df_train[col].min()
    df_test_target[col] = (df_test_target[col] - col_min) / (col_max - col_min)

for col in cat_cols:
    df_test_target[col] = (df_test_target[col] - body_min) / (body_max - body_min)
    df_test_target[col] = df_test_target[col].astype(float)
    
df_test_target.head()
# -

# ## test set 결측치 채우기

# Sex 뿐만 아니라 다른 결측치도 모두 채워주자.

df_test.isna().sum()

# ### Sex 

# +
df_fill = df_train_target.copy()
df_fill = df_fill.drop(["Delta 15 N (o/oo)", "Delta 13 C (o/oo)", "Sex"], axis=1)

df_fill_sex = df_fill.drop(["Island", "Clutch Completion"], axis=1)
df_fill_sex_y = df_train_target["Sex"].copy()

for i, u in enumerate(df_fill_sex_y.unique()):
    df_fill_sex_y.loc[(df_fill_sex_y == u)] = i
    
df_fill_sex_y

# +
ds_ = tf.data.Dataset.from_tensor_slices((df_fill.values, df_fill_sex_y.values)).shuffle(buffer_size=500)
ds_train = ds_.take(train_n)
ds_val = ds_.skip(train_n)

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train = ds_train = (
    ds_train
    .batch(32)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds_val = (
    ds_val
    .batch(32)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
# -

ds_train.element_spec

# +
model_fill_sex = keras.Sequential([
    layers.Dense(512, input_shape=[6], kernel_initializer=keras.initializers.he_normal()),
    layers.PReLU(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(64, kernel_initializer=keras.initializers.he_normal()),
    layers.PReLU(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    
    layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='sigmoid')
    
])
# -

model_fill_sex.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['binary_accuracy'])

# +
early = keras.callbacks.EarlyStopping(patience=100, min_delta=0.001, restore_best_weights=True)

history_fill_sex = model_fill_sex.fit(
    ds_train,
    validation_data=ds_val,
    callbacks=[early],
    epochs=2000
)
# -

history_df = pd.DataFrame(history_fill_sex.history)
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
print("Highest validation accuracy: {}".format(history_df.val_binary_accuracy.max()))

# +
df_fill_test = df_test_target.drop(["Sex", "Delta 15 N (o/oo)", "Delta 13 C (o/oo)"], axis=1)

df_fill_test_sex = df_fill_test.drop(["Island", "Clutch Completion"], axis=1)
idx_to_fill = df_test_target[df_test_target["Sex"].isna()].index

pred_fill = np.squeeze(model_fill_sex.predict(df_fill_test.loc[idx_to_fill].values))
pred_fill = list(map(round, pred_fill))
for i, c in enumerate(pred_fill):
    if (c == 0):
        pred_fill[i] = "MALE"
    elif (c == 1):
        pred_fill[i] = "FEMALE"

df_test.loc[idx_to_fill, "Sex"] = pred_fill
df_test.isna().sum()
# -

# ## Delta 15 N

df_test[df_test.isna().sum(axis=1) > 0]

# +
#df_fill_N = df_fill.drop(["Island", "Clutch Completion"], axis=1)

df_fill_N_y = df_train["Delta 15 N (o/oo)"]

# +
ds_ = tf.data.Dataset.from_tensor_slices((df_fill.values, df_fill_N_y.values)).shuffle(buffer_size=500)
ds_train = ds_.take(train_n)
ds_val = ds_.skip(train_n)

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train = ds_train = (
    ds_train
    .batch(32)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds_val = (
    ds_val
    .batch(32)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
# -

ds_train.element_spec

# +
model_fill_N = keras.Sequential([
    layers.Dense(512, input_shape=[6], kernel_initializer=keras.initializers.he_normal()),
    layers.PReLU(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(64, kernel_initializer=keras.initializers.he_normal()),
    layers.PReLU(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    
    layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='relu')
    
])
# -

model_fill_N.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# +
early = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

history_fill_N = model_fill_N.fit(
    ds_train,
    validation_data=ds_val,
    callbacks=[early],
    epochs=2000
)
# -

history_df = pd.DataFrame(history_fill_N.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Lowest validation loss: {}".format(history_df.val_loss.min()))

# +
idx_to_fill = df_test_target[df_test["Delta 15 N (o/oo)"].isna()].index

pred_fill = np.squeeze(model_fill_N.predict(df_fill_test.loc[idx_to_fill].values))

df_test.loc[idx_to_fill, "Delta 15 N (o/oo)"] = pred_fill
df_test.isna().sum()
# -

# ## Delta 13 C

# +
df_fill_C = df_fill.drop(["Island", "Clutch Completion"], axis=1)

df_fill_C_y = df_train["Delta 13 C (o/oo)"]

# +
ds_ = tf.data.Dataset.from_tensor_slices((df_fill_C.values, df_fill_C_y.values)).shuffle(buffer_size=500)
ds_train = ds_.take(train_n)
ds_val = ds_.skip(train_n)

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train = ds_train = (
    ds_train
    .batch(32)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds_val = (
    ds_val
    .batch(32)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
# -

ds_train.element_spec

# +
model_fill_C = keras.Sequential([
    layers.Dense(512, input_shape=[4], kernel_initializer=keras.initializers.he_normal()),
    layers.PReLU(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(64, kernel_initializer=keras.initializers.he_normal()),
    layers.PReLU(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    
    layers.Dense(1, kernel_initializer=keras.initializers.he_normal())
    
])
# -

model_fill_C.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# +
early = keras.callbacks.EarlyStopping(patience=100,  restore_best_weights=True)

history_fill_C = model_fill_C.fit(
    ds_train,
    validation_data=ds_val,
    callbacks=[early],
    epochs=2000
)
# -

history_df = pd.DataFrame(history_fill_C.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Lowest validation loss: {}".format(history_df.val_loss.min()))

# +
idx_to_fill = df_test[df_test["Delta 13 C (o/oo)"].isna()].index

pred_fill = np.squeeze(model_fill_C.predict(df_fill_test_sex.loc[idx_to_fill].values)) 
pred_fill

# -

df_test.loc[idx_to_fill, "Delta 13 C (o/oo)"] = pred_fill
df_test.isna().sum()

# +
df_test_target = df_test.copy()

for col in cat_cols:
    for u in df_train[col].unique():
        cat_idx = df_test_target[df_test_target[col] == u].index
        cat_mean = df_train[df_train[col] == u].loc[:, "Body Mass (g)"].mean()
        df_test_target.loc[cat_idx, col] = cat_mean

df_test_target.head()

# +
for col in cnt_cols:
    col_max = df_train[col].max()
    col_min = df_train[col].min()
    df_test_target[col] = (df_test_target[col] - col_min) / (col_max - col_min)

for col in cat_cols:
    df_test_target[col] = (df_test_target[col] - body_min) / (body_max - body_min)
    df_test_target[col] = df_test_target[col].astype(float)
    
df_test_target.head()
# -

df_test_target.isna().sum()

df_test.to_csv("test_filled.csv")
df_train.to_csv("train_droped.csv")

df_train_target.to_csv("train_target.csv")
df_train_y.to_csv("train_y.csv")
df_test_target.to_csv("test_target.csv")


# ## CV로 모델 훈련

# 이제 데이터 전처리는 완료했다. 이 데이터를 바탕으로 다시 학습시켜보자.

def Dense_model():
    
    model = keras.Sequential([
        layers.Dense(512, input_shape=[9], kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.75),

        layers.Dense(64, kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='relu')

    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss="mse")
    
    return model


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


def get_mse(pred, y):
    rmse = np.sqrt(np.multiply((y-pred), (y-pred))) / len(pred)
    return sum(rmse)


# +
from sklearn.model_selection import StratifiedKFold

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

num_seeds = 5
cv_train = np.zeros(len(df_train_target))
cv_pred = np.zeros(len(df_test_target))

early = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

for s in range(num_seeds):
    
    np.random.seed(s)

    for (tr_idx, te_idx) in kfold.split(df_train_target, df_train_y):
        ds_train = mk_Dataset(df_train_target.loc[tr_idx, :], df_train_y_rescaled.loc[tr_idx])
        ds_test = mk_Dataset(df_train_target.loc[te_idx, :], df_train_y_rescaled.loc[te_idx])
        
        model = Dense_model()
        
        model.fit(
            ds_train,
            validation_data=ds_test,
            verbose=0,
#            callbacks=[early],
            epochs=2000
        )
        
        model.fit(
            ds_train,
            validation_data=ds_test,
            verbose=0,
            callbacks=[early],
            epochs=2000
        )
        
        cv_train[te_idx] += np.squeeze(model.predict(df_train_target.loc[te_idx, :].values.astype(float)))
        print("mse: {}".format(get_mse(( cv_train[te_idx] * (body_max - body_min) / (s + 1)) + body_min, df_train_y[te_idx]) ))

        cv_pred += np.squeeze(model.predict(df_test_target.values.astype(float)))

    print("-----------------")
    print("seed{}_mse: {}".format(s, get_mse( ( (cv_train / (s + 1)) * (body_max - body_min) ) + body_min, df_train_y)))
    print("-----------------")
# -

cv_pred = cv_pred / (NFOLDS * num_seeds)
cv_pred = ( cv_pred * (body_max - body_min) ) + body_min
df_cv_pred = pd.DataFrame(cv_pred, columns=["Body Mass (g)"])
df_cv_pred.index.name = "id"
df_cv_pred

df_cv_pred.to_csv("cv_pred_dense.csv")

import lightgbm as lgb

# +

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

param = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'random_state': 30,
         'objective': 'mse', 'device_type': 'gpu'}

num_seeds = 16
n_round = 10000

cv_train_lgb = np.zeros(len(df_train_target))
cv_pred_lgb = np.zeros(len(df_test_target))

early = lgb.early_stopping(100)

for s in range(num_seeds):
    
    param['random_state'] = s

    for (tr_idx, te_idx) in kfold.split(df_train_target, df_train_y):
        
        train_data_lgb = lgb.Dataset(df_train_target.loc[tr_idx, :].values, label=df_train_y[tr_idx].values)
        val_data_lgb = train_data_lgb.create_valid(df_train_target.loc[te_idx, :].values, label=df_train_y[te_idx].values)
        
        bst = lgb.train(param, train_data_lgb, n_round, valid_sets=[val_data_lgb], callbacks=[early])
        print('best score: {}'.format(bst.best_score))

        cv_train_lgb[te_idx] += np.squeeze(bst.predict(df_train_target.loc[te_idx, :].values.astype(float)))
        print("mse: {}".format(get_mse( cv_train_lgb[te_idx] / (s + 1) , df_train_y[te_idx]) ))

        cv_pred_lgb += np.squeeze(bst.predict(df_test_target.values.astype(float)))

    print("-----------------")
    print("seed{}_mse: {}".format(s, get_mse( cv_train_lgb / (s + 1), df_train_y)))
    print("-----------------")
# -

cv_pred_lgb = cv_pred_lgb / (NFOLDS * num_seeds)
df_cv_pred_lgb = pd.DataFrame(cv_pred_lgb, columns=["Body Mass (g)"])
df_cv_pred_lgb.index.name = "id"
df_cv_pred_lgb

cv_pred_mix = cv_pred * 0.5 + cv_pred_lgb * 0.5
df_cv_mix = pd.DataFrame(cv_pred_mix, columns=["Body Mass (g)"])
df_cv_mix.index.name = "id"
df_cv_mix

df_cv_mix.to_csv("sub_mix_af.csv")


# ## Feature 추가

# 아직 성능이 부족하다. 서로 곱하고 나누고 제곱하여 새로운 feature들을 만들어 넣어 feature를 추가해본다.

def min_max_scale_2(df, col, col_min, col_max):
    df[col] = ( df[col] - col_min) / (col_max - col_min)


# +
from itertools import combinations

for i, col in enumerate(combinations(cnt_cols, 2)):
    df_train_target["m_{}".format(i)] = df_train[col[0]] * df_train[col[1]]
    df_train_target["d_{}".format(i)] = df_train[col[0]] / df_train[col[1]]
    df_test_target["m_{}".format(i)] = df_test[col[0]] * df_test[col[1]]
    df_test_target["d_{}".format(i)] = df_test[col[0]] / df_test[col[1]]
    
    col_max_m = df_train_target["m_{}".format(i)].max()
    col_max_d = df_train_target["d_{}".format(i)].max()
    col_min_m = df_train_target["m_{}".format(i)].min()
    col_min_d = df_train_target["d_{}".format(i)].min()
    
    min_max_scale_2(df_train_target, "m_{}".format(i), col_min_m, col_max_m)
    min_max_scale_2(df_train_target, "d_{}".format(i), col_min_d, col_max_d)
    min_max_scale_2(df_test_target, "m_{}".format(i), col_min_m, col_max_m)
    min_max_scale_2(df_test_target, "d_{}".format(i), col_min_d, col_max_d)

    
for i, col in enumerate(cnt_cols):
    df_train_target["p_{}".format(i)] = df_train[col] * df_train[col]
    df_test_target["p_{}".format(i)] = df_test[col] * df_test[col]

    col_max_p = df_train_target["p_{}".format(i)].max()
    col_min_p = df_train_target["p_{}".format(i)].min()

    min_max_scale_2(df_train_target, "p_{}".format(i), col_min_p, col_max_p)
    min_max_scale_2(df_test_target, "p_{}".format(i), col_min_p, col_max_p)

# -

df_test_target.count(axis=1)


def Dense_model():
    
    model = keras.Sequential([
        layers.Dense(512, input_shape=[34], kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='relu')

    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    
    return model


# +
from sklearn.model_selection import StratifiedKFold

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

num_seeds = 5
cv_train = np.zeros(len(df_train_target))
cv_pred = np.zeros(len(df_test_target))

early = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

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
# -

cv_pred = cv_pred / (NFOLDS * num_seeds)
df_cv_pred = pd.DataFrame(cv_pred, columns=["Body Mass (g)"])
df_cv_pred.index.name = "id"
df_cv_pred

df_cv_pred.to_csv("cv_pred_dense_relu.csv")

# +
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

param = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 110, 'random_state': 30,
         'objective': 'mse', 'device_type': 'gpu', 'verbosity' : -1}

num_seeds = 16
n_round = 10000

cv_train_lgb = np.zeros(len(df_train_target))
cv_pred_lgb = np.zeros(len(df_test_target))

early = lgb.early_stopping(100)

for s in range(num_seeds):
    
    param['random_state'] = s

    for (tr_idx, te_idx) in kfold.split(df_train_target, df_train_y):
        
        train_data_lgb = lgb.Dataset(df_train_target.loc[tr_idx, :].values, label=df_train_y.loc[tr_idx].values)
        val_data_lgb = train_data_lgb.create_valid(df_train_target.loc[te_idx, :].values, label=df_train_y.loc[te_idx].values)
        
        bst = lgb.train(param, train_data_lgb, n_round, valid_sets=[val_data_lgb], callbacks=[early])
        print('best score: {}'.format(bst.best_score))

        cv_train_lgb[te_idx] += np.squeeze(bst.predict(df_train_target.loc[te_idx, :].values.astype(float)))
        print("mse: {}".format(get_mse( cv_train_lgb[te_idx] / (s + 1) , df_train_y.loc[te_idx]) ))

        cv_pred_lgb += np.squeeze(bst.predict(df_test_target.values.astype(float)))

    print("-----------------")
    print("seed{}_mse: {}".format(s, get_mse( cv_train_lgb / (s + 1), df_train_y)))
    print("-----------------")
# -

cv_pred_lgb = cv_pred_lgb / (NFOLDS * num_seeds)
df_cv_pred_lgb = pd.DataFrame(cv_pred_lgb, columns=["Body Mass (g)"])
df_cv_pred_lgb.index.name = "id"
df_cv_pred_lgb

df_cv_pred_lgb.to_csv('cv_pred_lgb.csv')

cv_pred_mix = cv_pred * 0.5 + cv_pred_lgb * 0.5
df_cv_mix = pd.DataFrame(cv_pred_mix, columns=["Body Mass (g)"])
df_cv_mix.index.name = "id"
df_cv_mix

df_cv_mix.to_csv("sub_cv_mix_interfeature_2.csv")

# ## output rescale and sigmoid

# test set 결측치를 채우는 도중 output을 rescaling하여 classifier처럼 마지막 층의 activation을 sigmoid 함수로 설정하면 성능이 향상되는 것을 보았다. 최종 모델에도 적용해보자.

df_train_y_rescaled


def Dense_model():
    
    model = keras.Sequential([
        layers.Dense(512, input_shape=[34], kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='sigmoid')

    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    
    return model


# +
from sklearn.model_selection import StratifiedKFold

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

num_seeds = 5
cv_train = np.zeros(len(df_train_target))
cv_pred = np.zeros(len(df_test_target))

early = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

for s in range(num_seeds):
    
    np.random.seed(s)

    for (tr_idx, te_idx) in kfold.split(df_train_target, df_train_y):
        ds_train = mk_Dataset(df_train_target.loc[tr_idx, :], df_train_y_rescaled.loc[tr_idx])
        ds_test = mk_Dataset(df_train_target.loc[te_idx, :], df_train_y_rescaled.loc[te_idx])
        
        model = Dense_model()
        
        model.fit(
            ds_train,
            validation_data=ds_test,
            verbose=0,
            callbacks=[early],
            epochs=2000
        )
        
        cv_train[te_idx] += np.squeeze(model.predict(df_train_target.loc[te_idx, :].values.astype(float)))
        print("mse: {}".format(get_mse( ((cv_train[te_idx] / (s + 1)) * (body_max - body_min)) + body_min, df_train_y[te_idx]) ))

        cv_pred += np.squeeze(model.predict(df_test_target.values.astype(float)))

    print("-----------------")
    print("seed{}_mse: {}".format(s, get_mse( ((cv_train / (s + 1)) * (body_max - body_min)) + body_min, df_train_y)))
    print("-----------------")
# -

cv_pred = cv_pred / (NFOLDS * num_seeds)
cv_pred = ( cv_pred * (body_max - body_min) )+ body_min
df_cv_pred = pd.DataFrame(cv_pred, columns=["Body Mass (g)"])
df_cv_pred.index.name = "id"
df_cv_pred

df_cv_pred.to_csv("cv_pred_dense_sig.csv")

# +
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

param = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 110, 'random_state': 30,
         'objective': 'mse', 'device_type': 'gpu', 'verbosity' : -1}

num_seeds = 16
n_round = 10000

cv_train_lgb = np.zeros(len(df_train_target))
cv_pred_lgb = np.zeros(len(df_test_target))

early = lgb.early_stopping(100)

for s in range(num_seeds):
    
    param['random_state'] = s

    for (tr_idx, te_idx) in kfold.split(df_train_target, df_train_y):
        
        train_data_lgb = lgb.Dataset(df_train_target.loc[tr_idx, :].values, label=df_train_y_rescaled.loc[tr_idx].values)
        val_data_lgb = train_data_lgb.create_valid(df_train_target.loc[te_idx, :].values, label=df_train_y_rescaled.loc[te_idx].values)
        
        bst = lgb.train(param, train_data_lgb, n_round, valid_sets=[val_data_lgb], callbacks=[early])
        print('best score: {}'.format(bst.best_score))

        cv_train_lgb[te_idx] += np.squeeze(bst.predict(df_train_target.loc[te_idx, :].values.astype(float)))
        print("mse: {}".format(get_mse( ((cv_train_lgb[te_idx] / (s + 1)) * (body_max - body_min)) + body_min, df_train_y.loc[te_idx]) ))

        cv_pred_lgb += np.squeeze(bst.predict(df_test_target.values.astype(float)))

    print("-----------------")
    print("seed{}_mse: {}".format(s, get_mse( ((cv_train_lgb / (s + 1)) * (body_max - body_min)) + body_min, df_train_y)))
    print("-----------------")
# -

cv_pred_lgb = cv_pred_lgb / (NFOLDS * num_seeds)
cv_pred_lgb = ( cv_pred_lgb * (body_max - body_min) )+ body_min
df_cv_pred_lgb = pd.DataFrame(cv_pred_lgb, columns=["Body Mass (g)"])
df_cv_pred_lgb.index.name = "id"
df_cv_pred_lgb

df_cv_pred_lgb.to_csv("cv_pred_lgb_sig.csv")

cv_pred_mix = cv_pred * 0.4 + cv_pred_lgb * 0.3 + np.squeeze(df_cv_pred_relu.values) * 0.3
df_cv_mix = pd.DataFrame(cv_pred_mix, columns=["Body Mass (g)"])
df_cv_mix.index.name = "id"
df_cv_mix

df_cv_mix.to_csv("sub_cv_mix_interfeature_3.csv")
