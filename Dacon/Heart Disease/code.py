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

# +
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='cividis')
# -

df_train = pd.read_csv('./train.csv', index_col='id')
df_train.head()

df_train.describe()

# **cp**: 가슴 통증(chest pain) 종류 
# 0 : asymptomatic 무증상
# 1 : atypical angina 일반적이지 않은 협심증
# 2 : non-anginal pain 협심증이 아닌 통증
# 3 : typical angina 일반적인 협심증
#
# **restecg**: (resting electrocardiographic) 휴식 중 심전도 결과 
# 0: showing probable or definite left ventricular hypertrophy by Estes' criteria
# 1: 정상
# 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
#
# **slope**: (the slope of the peak exercise ST segment) 활동 ST 분절 피크의 기울기
# 0: downsloping 하강
# 1: flat 평탄
# 2: upsloping 상승
#
# **ca**: number of major vessels colored by flouroscopy 형광 투시로 확인된 주요 혈관 수 (0~3 개) 
# Null 값은 숫자 4로 인코딩됨 -> 4는 없음
#
# **thal**: thalassemia 지중해빈혈 여부
# 0 = Null 
# 1 = normal 정상
# 2 = fixed defect 고정 결함
# 3 = reversable defect 가역 결함
#

cp_df = pd.get_dummies(df_train.cp, prefix='cp')
restecg_df = pd.get_dummies(df_train.restecg, prefix='restecg')
slope_df = pd.get_dummies(df_train.slope, prefix='slope')
ca_df = pd.get_dummies(df_train.ca, prefix='ca')
thal_df = pd.get_dummies(df_train.thal, prefix='thal')

# 위 특성들은 주의할 것들. one hot encoding해주자.

cp_df = cp_df.drop('cp_0', axis=1)
restecg_df = restecg_df.drop('restecg_1', axis=1)
slope_df = slope_df.drop('slope_1', axis=1)
thal_df = thal_df.drop(['thal_0', 'thal_1'], axis=1)

df_train_onehot = df_train.drop(['cp', 'restecg', 'slope', 'ca', 'thal'], axis=1)
df_train_onehot = pd.concat([df_train_onehot, cp_df, restecg_df, slope_df, ca_df, thal_df], axis=1)
df_train_onehot.head()

# - age: 나이
# - trestbps: (resting blood pressure) 휴식 중 혈압(mmHg)
# - chol: (serum cholestoral) 혈중 콜레스테롤 (mg/dl)
# - thalach: (maximum heart rate achieved) 최대 심박수
# - oldpeak: (ST depression induced by exercise relative to rest) 휴식 대비 운동으로 인한 ST 하강

continuous_f = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
max_val = []
for i in continuous_f:
    print("max value for {} : {}".format(i, df_train[i].max()))

for i in continuous_f:
    df_train_onehot[i] = df_train_onehot[i] / df_train_onehot[i].max()
df_train_onehot.head()

# 연속적인 값들은 0과 1 사이로 rescaling.

target_train = df_train_onehot.pop('target')

df_train_onehot.values

dataset = tf.data.Dataset.from_tensor_slices((df_train_onehot.values, target_train.values))

train_n = int(df_train_onehot.count()[0]*0.7)
train_n



ds_train = dataset.take(train_n)
ds_val = dataset.skip(train_n)

ds_train.element_spec

# +
AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train = (
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
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, input_shape=[21], activation='relu'),
    
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
    
    layers.Dense(1, activation='sigmoid')
    
])
# -

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# +
early = keras.callbacks.EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)

history = model.fit(
    ds_train,
    validation_data=ds_val,
    callbacks=[early],
    epochs=100
)
# -

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
print("Highest validation accuracy: {}".format(history_df.val_binary_accuracy.max()))

df_test = pd.read_csv('./test.csv', index_col='id')
df_test.head()

cp_df = pd.get_dummies(df_test.cp, prefix='cp')
restecg_df = pd.get_dummies(df_test.restecg, prefix='restecg')
slope_df = pd.get_dummies(df_test.slope, prefix='slope')
ca_df = pd.get_dummies(df_test.ca, prefix='ca')
thal_df = pd.get_dummies(df_test.thal, prefix='thal')

cp_df = cp_df.drop('cp_0', axis=1)
restecg_df = restecg_df.drop('restecg_1', axis=1)
slope_df = slope_df.drop('slope_1', axis=1)
ca_df = ca_df.drop('ca_4', axis=1)
thal_df = thal_df.drop(['thal_0', 'thal_1'], axis=1)

df_test_onehot = df_test.drop(['cp', 'restecg', 'slope', 'ca', 'thal'], axis=1)
df_test_onehot = pd.concat([df_test_onehot, cp_df, restecg_df, slope_df, ca_df, thal_df], axis=1)
df_test_onehot.head()

for i in continuous_f:
    df_test_onehot[i] = df_test_onehot[i] / df_train[i].max()
df_test_onehot.head()

ds_test = tf.data.Dataset.from_tensor_slices((df_test_onehot.values))
ds_test = ds_test.batch(1)

df_pred = pd.DataFrame(data=model.predict(ds_test), columns=['target'])
df_pred['id'] = df_pred.index + 1

# +
df_pred = df_pred[['id', 'target']]

df_pred['target'] = list(map(round, df_pred['target']))
df_pred
# -

df_pred.to_csv('submission.csv', index=False)
