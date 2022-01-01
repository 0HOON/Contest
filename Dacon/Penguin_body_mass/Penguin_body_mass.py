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

# 펭귄 몸무게 예측 경진대회
# =======
# https://dacon.io/competitions/official/235862/overview/description
#
#

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
# -

df_train = pd.read_csv('./train.csv', index_col='id')
df_train.head()

df_train.describe()

print("Species info")
print(df_train.Species.describe())
print("unique value: {}".format(df_train.Species.unique()))

# 종의 종류는 총 3가지. Gentoo, Chinstrap, Adelie.

print("Island info")
print(df_train.Island.describe())
print("unique values: {}".format(df_train.Island.unique()))

# 근처 섬 이름도 3가지. Biscoe, Dream, Torgersen. 

pd.plotting.scatter_matrix(df_train, alpha=1, figsize=(12,12), diagonal="kde")

plt.figure(figsize=(12, 12))
cols = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Delta 15 N (o/oo)", "Delta 13 C (o/oo)"]
for i, col in enumerate(cols):
    plt.subplot(3, 3, i+1)
    sns.scatterplot(x=col, y="Body Mass (g)", data=df_train, hue="Species", style="Sex", legend=False)

plt.figure(figsize=(12, 12))
cols = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Delta 15 N (o/oo)", "Delta 13 C (o/oo)"]
for i, col in enumerate(cols):
    plt.subplot(3, 3, i+1)
    sns.scatterplot(x=col, y="Body Mass (g)", data=df_train, hue="Clutch Completion", legend=False)

sns.scatterplot(x="Clutch Completion", y="Body Mass (g)", hue="Species", data=df_train)

plt.figure(figsize=(12, 4))
for i, s in enumerate(df_train["Species"].unique()):
    plt.subplot(1, 3, i+1)
    sns.scatterplot(x="Clutch Completion", y="Body Mass (g)",  data=df_train[df_train["Species"]==s])
    plt.title(s.split()[0])

sns.scatterplot(x="Island", y="Body Mass (g)", hue="Species", data=df_train)

sns.scatterplot(x="Island", y="Body Mass (g)", hue="Species", 
                data=df_train[df_train["Species"] != "Gentoo penguin (Pygoscelis papua)"])

for i in df_train["Island"].unique():
    df_ = df_train[df_train["Species"] == "Adelie Penguin (Pygoscelis adeliae)"]
    print("\n\n" + i)
    print(df_[df_["Island"] == i].mean())

# 위 그래프들에서 Delta 15 N, Delta 13 C, Clutch Completion, island의 경우 body mass와 큰 연관은 없는 것으로 보인다. 우선 이 특성들은 제외하고 학습시켜보자.

df_train_fixed = df_train.drop(["Delta 15 N (o/oo)", "Delta 13 C (o/oo)", "Clutch Completion", "Island"], axis=1)
df_train_fixed.head()

# apply **one hot encoding** categorical features and **rescale** continuous values

species_df = pd.get_dummies(df_train["Species"])
Island_df = pd.get_dummies(df_train["Sex"])
Island_df = Island_df.drop("MALE", axis=1)

df_train_onehot = df_train_fixed.drop(["Species", "Sex"], axis=1)
df_train_onehot = pd.concat([df_train_onehot, species_df, Island_df], axis=1)
df_train_onehot.head()

df_train_y = df_train_onehot.pop("Body Mass (g)")
for col in df_train_onehot.columns:
    df_train_onehot[col] = df_train_onehot[col] / df_train_onehot[col].max()

df_train_onehot.head()

df_train_y_rescaled = df_train_y / df_train_y.max()
df_train_y_rescaled.head()

# 데이터 준비 완료. 데이터셋이 적으므로 KFold Cross validation method로 학습시켜보자.

train_n = int(df_train_onehot.count()[0]*0.8)
train_n

# +
ds_ = tf.data.Dataset.from_tensor_slices((df_train_onehot.values, df_train_y.values)).shuffle(buffer_size=500)
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

# ## Dense model

# +
model_Dense = keras.Sequential([
    layers.Dense(64, input_shape=[7], activation='relu'),
    
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
     layers.Dropout(0.3),
    
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
     layers.Dropout(0.3),
    
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    
    layers.Dense(1, activation='relu')
    
])
# -

# 우선 6층짜리 Dense model로 시험해보자

model_Dense.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss="mse")

# +
early = keras.callbacks.EarlyStopping(patience=100, min_delta=0.001, restore_best_weights=True)

history_Dense = model_Dense.fit(
    ds_train,
    validation_data=ds_val,
    callbacks=[early],
    epochs=1000
)
# -

history_df = pd.DataFrame(history_Dense.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Lowest validation mse: {}".format(history_df.val_loss.min()))

# +
df_test = pd.read_csv("./test.csv", index_col="id")

species_df = pd.get_dummies(df_test["Species"])
Island_df = pd.get_dummies(df_test["Sex"])
Island_df = Island_df.drop("MALE", axis=1)

df_test_onehot = df_test.drop(["Species", "Sex", "Delta 15 N (o/oo)", "Delta 13 C (o/oo)", "Clutch Completion", "Island"], axis=1)
df_test_onehot = pd.concat([df_test_onehot, species_df, Island_df], axis=1)

for col in df_test_onehot.columns:
    if (col in df_train.columns):
        df_test_onehot[col] = df_test_onehot[col] / df_train[col].max()

df_test_onehot.head()
# -

pred = model_Dense.predict(df_test_onehot)
pred

df_pred = pd.DataFrame(data=pred, columns=['Body Mass (g)'])
df_pred.index.name = "id"
df_pred.head()

df_pred.to_csv('submission.csv')

# ## Dense model (simpler)

# +
model_Dense_s = keras.Sequential([
    layers.Dense(32, input_shape=[7], activation='relu'),
    
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(1, activation='relu')
    
])
# -

# 우선 6층짜리 Dense model로 시험해보자

model_Dense_s.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

# +
early = keras.callbacks.EarlyStopping(patience=100, min_delta=0.001, restore_best_weights=True)

history_Dense_s = model_Dense_s.fit(
    ds_train,
    validation_data=ds_val,
    callbacks=[early],
    epochs=1000
)
# -

history_df = pd.DataFrame(history_Dense_s.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Lowest validation mse: {}".format(history_df.val_loss.min()))

# ## light GBM

# +
import lightgbm as lgb

train_data_lgb = lgb.Dataset(df_train_onehot[:train_n].values, label=df_train_y[:train_n].values)
val_data_lgb = train_data_lgb.create_valid(df_train_onehot[train_n:].values, label=df_train_y[train_n:].values)

# +
param = {'num_leaves': 31, 'objective': 'mse', 'device_type': 'gpu'}
n_round = 100

record = {}
record_eval = lgb.record_evaluation(record)
early = lgb.early_stopping(20)

bst = lgb.train(param, train_data_lgb, n_round, valid_sets=[val_data_lgb], callbacks=[record_eval, early])
bst.save_model('lgb_model.txt', num_iteration=bst.best_iteration)
print('best score: {}'.format(bst.best_score))
# -

pred = bst.predict(df_test_onehot.values)
pred

df_pred = pd.DataFrame(data=pred, columns=['Body Mass (g)'])
df_pred.index.name = "id"
df_pred.head()

df_pred.to_csv('submission_bst.csv')

lgb.plot_importance(bst)

lgb.plot_metric(record)

# ## Output rescaling

# +
import lightgbm as lgb

train_data_lgb_or = lgb.Dataset(df_train_onehot[:train_n].values, label=df_train_y_rescaled[:train_n].values)
val_data_lgb_or = train_data_lgb.create_valid(df_train_onehot[train_n:].values, label=df_train_y_rescaled[train_n:].values)

# +
param = {'num_leaves': 31, 'objective': 'mse', 'device_type': 'gpu'}
n_round = 100

record = {}
record_eval = lgb.record_evaluation(record)
early = lgb.early_stopping(100)

bst_or = lgb.train(param, train_data_lgb_or, n_round, valid_sets=[val_data_lgb_or], callbacks=[record_eval, early])
bst_or.save_model('lgb_model_or.txt', num_iteration=bst.best_iteration)
print('best score: {}'.format(bst.best_score))
# -

pred = bst_or.predict(df_test_onehot.values) * df_train_y.max()
pred

df_pred = pd.DataFrame(data=pred, columns=['Body Mass (g)'])
df_pred.index.name = "id"
df_pred.head()

df_pred.to_csv("submission_or.csv")

# +
ds_ = tf.data.Dataset.from_tensor_slices((df_train_onehot.values, df_train_y_rescaled.values)).shuffle(buffer_size=500)
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

# +
model_Dense = keras.Sequential([
    layers.Dense(64, input_shape=[7], activation='relu'),
    
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
     layers.Dropout(0.3),
    
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
     layers.Dropout(0.3),
    
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    
    layers.Dense(1, activation='relu')
    
])
# -

model_Dense.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss="mse")

# +
early = keras.callbacks.EarlyStopping(patience=100, min_delta=0.001, restore_best_weights=True)

history_Dense = model_Dense.fit(
    ds_train,
    validation_data=ds_val,
    callbacks=[early],
    epochs=1000
)
# -

history_df = pd.DataFrame(history_Dense.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Lowest validation mse: {}".format(history_df.val_loss.min()))

pred = model_Dense.predict(df_test_onehot) * df_train_y.max()
df_pred = pd.DataFrame(data=pred, columns=['Body Mass (g)'])
df_pred.index.name = "id"
df_pred.head()

df_pred.to_csv("submission_dense_or.csv")

# ## Grid Search for Light GBM

X_train = df_train_onehot.values
y_train = df_train_y.values

# +
from sklearn.model_selection import GridSearchCV

grid_params ={'boosting_type': ['gbdt', 'dart', 'goss', 'rf'], 'max_depth' : range(1, 20, 2) , 'n_estimators': range(50, 150, 10), 'learning_rate':[0.001, 0.01, 0.1], 'random_state':[30]}
grid_search = GridSearchCV(lgb.LGBMRegressor(), grid_params, scoring='neg_root_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)
# -

print('Best_params: {}'.format(grid_search.best_params_))
print('Best_score: {:.4f}'.format(grid_search.best_score_))

# grid search로 찾은 최적의 parameter들. 이것들을 적용해 새로 학습시켜보자.

# +
param = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 100, 'random_state': 30,
         'objective': 'mse', 'device_type': 'gpu'}
n_round = 1000

record = {}
record_eval = lgb.record_evaluation(record)
early = lgb.early_stopping(20)

bst_gs = lgb.train(param, train_data_lgb, n_round, valid_sets=[val_data_lgb], callbacks=[record_eval, early])
bst_gs.save_model('lgb_model_gs.txt', num_iteration=bst_gs.best_iteration)
print('best score: {}'.format(bst_gs.best_score))
# -

pred = bst_gs.predict(df_test_onehot.values)
pred

df_pred = pd.DataFrame(data=pred, columns=['Body Mass (g)'])
df_pred.index.name = "id"
df_pred.head()

df_pred.to_csv('submission_bst_grid.csv')

lgb.plot_importance(bst_gs)

lgb.plot_metric(record)

# 제출 전 모델 선택

test_sample = df_train_onehot.sample(40, random_state=10)
test_sample_y = df_train_y.sample(40, random_state=10)


def get_mse(pred, y):
    rmse = np.sqrt(np.multiply((y-pred), (y-pred))) / len(pred)
    return sum(rmse)


# +
pred_d = np.squeeze(model_Dense.predict(test_sample) * df_train_y.max())
pred_g = np.squeeze(bst_gs.predict(test_sample.values))

print("rmse for dense: {}\nrmse for grid: {}".format(get_mse(pred_d, test_sample_y.values), get_mse(pred_g, test_sample_y.values)))
# -

# ## 결측치 조정

df_train.isna().sum()

df_train[df_train["Sex"].isna()]

df_train.dropna(subset=["Sex"], inplace=True)
df_train["Sex"].isna().sum()


# Sex column에서 na 제거.

def outlier_idx(df, col, weight=1.5):
    q_25 = np.percentile(df[col], 25)
    q_75 = np.percentile(df[col], 75)
    
    iqr = q_75 - q_25
    
    lower_bound = q_25 - iqr
    upper_bound = q_75 + iqr
    
    o_idx = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
    
    return o_idx


cols = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Body Mass (g)"]
for col in cols:
    outlier_idx_ = outlier_idx(df_train, col)
    df_train.drop(outlier_idx_, inplace=True)
df_train.describe()

# apply **one hot encoding** categorical features and **rescale** continuous values

species_df = pd.get_dummies(df_train["Species"])
Island_df = pd.get_dummies(df_train["Sex"])
Island_df = Island_df.drop("MALE", axis=1)

df_train_onehot = df_train.drop(["Species", "Sex", "Delta 15 N (o/oo)", "Delta 13 C (o/oo)", "Clutch Completion", "Island"], axis=1)
df_train_onehot = pd.concat([df_train_onehot, species_df, Island_df], axis=1)
df_train_onehot.head()

df_train_y = df_train_onehot.pop("Body Mass (g)")
for col in df_train_onehot.columns:
    df_train_onehot[col] = df_train_onehot[col] / df_train_onehot[col].max()

# +
train_data_lgb = lgb.Dataset(df_train_onehot[:train_n].values, label=df_train_y[:train_n].values)
val_data_lgb = train_data_lgb.create_valid(df_train_onehot[train_n:].values, label=df_train_y[train_n:].values)

param = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 140, 'random_state': 30,
         'objective': 'mse', 'device_type': 'gpu'}
n_round = 1000

record = {}
record_eval = lgb.record_evaluation(record)
early = lgb.early_stopping(100)

bst_no = lgb.train(param, train_data_lgb, n_round, valid_sets=[val_data_lgb], callbacks=[record_eval, early])
bst_no.save_model('lgb_model_no.txt', num_iteration=bst_no.best_iteration)
print('best score: {}'.format(bst_no.best_score))
# -

lgb.plot_importance(bst_no)

lgb.plot_metric(record)

pred = bst_no.predict(df_test_onehot.values)
pred

df_pred = pd.DataFrame(data=pred, columns=['Body Mass (g)'])
df_pred.index.name = "id"
df_pred.head()

df_pred.to_csv('submission_bst_no.csv')

# ## one hot 대신 target encoding

df_train_onehot.head()

# +
mean_A = df_train[df_train["Species"] == "Adelie Penguin (Pygoscelis adeliae)"].loc[:, "Body Mass (g)"].mean()
mean_C = df_train[df_train["Species"] == "Chinstrap penguin (Pygoscelis antarctica)"].loc[:, "Body Mass (g)"].mean()
mean_G = df_train[df_train["Species"] == "Gentoo penguin (Pygoscelis papua)"].loc[:, "Body Mass (g)"].mean()
mean_M = df_train[df_train["Sex"] == "MALE"].loc[:, "Body Mass (g)"].mean()
mean_F = df_train[df_train["Sex"] == "FEMALE"].loc[:, "Body Mass (g)"].mean()

idx_A = df_train[df_train["Species"] == "Adelie Penguin (Pygoscelis adeliae)"].index
idx_C = df_train[df_train["Species"] == "Chinstrap penguin (Pygoscelis antarctica)"].index
idx_G = df_train[df_train["Species"] == "Gentoo penguin (Pygoscelis papua)"].index
idx_M = df_train[df_train["Sex"] == "MALE"].index
idx_F = df_train[df_train["Sex"] == "FEMALE"].index

df_train_target = df_train.copy()

df_train_target.loc[idx_A, "Species"] = mean_A 
df_train_target.loc[idx_C, "Species"] = mean_C 
df_train_target.loc[idx_G, "Species"] = mean_G 
df_train_target.loc[idx_M, "Sex"] = mean_M
df_train_target.loc[idx_F, "Sex"] = mean_F

df_train_target.head()
# -

df_train_target = df_train_target.drop(["Delta 15 N (o/oo)", "Delta 13 C (o/oo)", "Clutch Completion", "Island"], axis=1)
df_train_target.head()


# +
def min_max_scale(df, col):
    col_min = df[col].min()
    col_max = df[col].max()
    df[col] = ( df[col] - col_min) / (col_max - col_min)

cols = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)"]

for col in cols:
    min_max_scale(df_train_target, col)
df_train_target.head()

# +
cols = ["Species", "Sex"]
for col in cols:
    df_train_target[col] = (df_train_target[col] - df_train["Body Mass (g)"].min()) / (df_train["Body Mass (g)"].max() - df_train["Body Mass (g)"].min())

df_train_target.head()
# -

# ### test set

# +
idx_A = df_test[df_test["Species"] == "Adelie Penguin (Pygoscelis adeliae)"].index
idx_C = df_test[df_test["Species"] == "Chinstrap penguin (Pygoscelis antarctica)"].index
idx_G = df_test[df_test["Species"] == "Gentoo penguin (Pygoscelis papua)"].index
idx_M = df_test[df_test["Sex"] == "MALE"].index
idx_F = df_test[df_test["Sex"] == "FEMALE"].index

df_test_target = df_test.copy()

df_test_target.loc[idx_A, "Species"] = mean_A 
df_test_target.loc[idx_C, "Species"] = mean_C 
df_test_target.loc[idx_G, "Species"] = mean_G 
df_test_target.loc[idx_M, "Sex"] = mean_M
df_test_target.loc[idx_F, "Sex"] = mean_F

df_test_target = df_test_target.drop(["Delta 15 N (o/oo)", "Delta 13 C (o/oo)", "Clutch Completion", "Island"], axis=1)

cols = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)"]

for col in cols:
    col_min = df_train[col].min()
    col_max = df_train[col].max()
    df_test_target[col] = (df_test_target[col] - col_min) / (col_max - col_min)

cols = ["Species", "Sex"]
for col in cols:
    df_test_target[col] = (df_test_target[col] - df_train["Body Mass (g)"].min()) / (df_train["Body Mass (g)"].max() - df_train["Body Mass (g)"].min())

df_test_target.head()

# +
train_data_lgb = lgb.Dataset(df_train_target[:train_n].values, label=df_train_y[:train_n].values)
val_data_lgb = train_data_lgb.create_valid(df_train_target[train_n:].values, label=df_train_y[train_n:].values)

param = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 140, 'random_state': 30,
         'objective': 'mse', 'device_type': 'gpu'}
n_round = 1000

record = {}
record_eval = lgb.record_evaluation(record)
early = lgb.early_stopping(100)

bst_target = lgb.train(param, train_data_lgb, n_round, valid_sets=[val_data_lgb], callbacks=[record_eval, early])
bst_target.save_model('lgb_model_target.txt', num_iteration=bst_target.best_iteration)
print('best score: {}'.format(bst_target.best_score))
# -

lgb.plot_importance(bst_target)

lgb.plot_metric(record)

pred = bst_target.predict(df_test_target.values)
pred

df_pred = pd.DataFrame(data=pred, columns=['Body Mass (g)'])
df_pred.index.name = "id"
df_pred.head()

df_pred.to_csv('submission_bst_target.csv')

# dense 모델도 학습시켜보자

# +
ds_ = tf.data.Dataset.from_tensor_slices((df_train_target.values.astype(float), df_train_y_rescaled.values)).shuffle(buffer_size=500)
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
model_Dense = keras.Sequential([
    layers.Dense(512, input_shape=[5], kernel_initializer=keras.initializers.he_normal()),
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

model_Dense.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss="mse")

# +
early = keras.callbacks.EarlyStopping(patience=100, min_delta=0.001, restore_best_weights=True)

history_Dense = model_Dense.fit(
    ds_train,
    
    validation_data=ds_val,
#    callbacks=[early],
    epochs=2000
)
# -

history_df = pd.DataFrame(history_Dense.history)
history_df.loc[1000:, ['loss', 'val_loss']].plot()
print("Lowest validation mse: {}".format(history_df.val_loss.min()))

test_sample = df_train_target.sample(40, random_state=10)
test_sample_y = df_train_y.sample(40, random_state=10)


def get_mse(pred, y):
    rmse = np.sqrt(np.multiply((y-pred), (y-pred))) / len(pred)
    return sum(rmse)


# +
pred_d = np.squeeze(model_Dense.predict(test_sample.values.astype(float)))
pred_g = np.squeeze(bst_target.predict(test_sample.values.astype(float))) 
pred_d = (pred_d * (col_max - col_min)) + col_min

pred_mix = (pred_d + pred_g)/2

print("rmse for dense: {}\nrmse for bst: {}\n mix: {}".format(get_mse(pred_d, test_sample_y.values), get_mse(pred_g, test_sample_y.values), get_mse(pred_mix, test_sample_y.values)))
# -

# ### feature 추가

# +
from itertools import combinations

cols = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)"]

for i, col in enumerate(combinations(cols, 2)):
    df_train_target["m_{}".format(i)] = df_train_target[col[0]] * df_train_target[col[1]]
    df_train_target["d_{}".format(i)] = df_train_target[col[0]] / df_train_target[col[1]]
    df_test_target["m_{}".format(i)] = df_test_target[col[0]] * df_train_target[col[1]]
    df_test_target["d_{}".format(i)] = df_test_target[col[0]] / df_test_target[col[1]]

# +
ds_ = tf.data.Dataset.from_tensor_slices((df_train_target.values.astype(float), df_train_y_rescaled.values)).shuffle(buffer_size=500)
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
model_Dense = keras.Sequential([
    layers.Dense(512, input_shape=[11], kernel_initializer=keras.initializers.he_normal()),
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

model_Dense.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss="mse")

# +
early = keras.callbacks.EarlyStopping(patience=100, min_delta=0.001, restore_best_weights=True)

history_Dense = model_Dense.fit(
    ds_train,
    
    validation_data=ds_val,
    callbacks=[early],
    epochs=2000
)
# -

history_df = pd.DataFrame(history_Dense.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Lowest validation mse: {}".format(history_df.val_loss.min()))

# +
train_data_lgb = lgb.Dataset(df_train_target[:train_n].values, label=df_train_y[:train_n].values)
val_data_lgb = train_data_lgb.create_valid(df_train_target[train_n:].values, label=df_train_y[train_n:].values)

param = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 1, 'n_estimators': 140, 'random_state': 30,
         'objective': 'mse', 'device_type': 'gpu'}
n_round = 1000

record = {}
record_eval = lgb.record_evaluation(record)
early = lgb.early_stopping(100)

bst_target = lgb.train(param, train_data_lgb, n_round, valid_sets=[val_data_lgb], callbacks=[record_eval, early])
bst_target.save_model('lgb_model_target.txt', num_iteration=bst_target.best_iteration)
print('best score: {}'.format(bst_target.best_score))
# -

lgb.plot_importance(bst_target)

lgb.plot_metric(record)

test_sample = df_train_target.sample(40, random_state=10)
test_sample_y = df_train_y.sample(40, random_state=10)

# +
pred_d = np.squeeze(model_Dense.predict(test_sample.values.astype(float)))
pred_g = np.squeeze(bst_target.predict(test_sample.values.astype(float))) 
pred_d = (pred_d * (col_max - col_min)) + col_min

pred_mix = (pred_d + pred_g)/2

print("rmse for dense: {}\nrmse for bst: {}\n mix: {}".format(get_mse(pred_d, test_sample_y.values), get_mse(pred_g, test_sample_y.values), get_mse(pred_mix, test_sample_y.values)))
# -

df_test_target.isna().sum()

# +
pred_d = np.squeeze(model_Dense.predict(df_test_target.values.astype(float)))
pred_d = (pred_d * (col_max - col_min)) + col_min
pred_g = np.squeeze(bst_target.predict(df_test_target.values.astype(float))) 

pred_mix = (pred_d + pred_g)/2
# -

df_pred = pd.DataFrame(pred_mix, columns=["Body Mass (g)"])
df_pred.index.name = "id"
df_pred

df_pred.to_csv('submisssion_interfeature.csv')
