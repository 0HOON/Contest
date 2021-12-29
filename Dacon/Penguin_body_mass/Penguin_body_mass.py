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
