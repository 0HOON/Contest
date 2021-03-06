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

# 심장 질환 예측 경진대회
# ==========
# https://dacon.io/competitions/official/235848/overview/description
#
# 참여해본 두 번째 경진대회. 이번엔 유명한 데이터인 심장 질환 관련 수치들을 가지고 심장 질환 여부를 예측하는 Binary Classification 모델을 구성하는 것이 목적이었다. 다루는 데이터가 수치로 주어져 있어 데이터 분석 및 처리가 수월할 것으로 예상한다.

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

# 필요한 패키지 임포트 및 데이터 불러오기. data frame으로 데이터의 대략적인 모양새를 살펴본다.

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
# Null 값은 숫자 4로 인코딩됨 -> 4는 training data에는 없고, test data에서 한 건 발견된다.
#
# **thal**: thalassemia 지중해빈혈 여부
# 0 = Null 
# 1 = normal 정상
# 2 = fixed defect 고정 결함
# 3 = reversable defect 가역 결함
#
# - *위 feature들은 수치가 아니라 class를 나타내므로 one hot encoding을 해주는 것이 좋을 것이다.*

cp_df = pd.get_dummies(df_train.cp, prefix='cp')
restecg_df = pd.get_dummies(df_train.restecg, prefix='restecg')
slope_df = pd.get_dummies(df_train.slope, prefix='slope')
ca_df = pd.get_dummies(df_train.ca, prefix='ca')
thal_df = pd.get_dummies(df_train.thal, prefix='thal')

cp_df = cp_df.drop('cp_0', axis=1)
restecg_df = restecg_df.drop('restecg_1', axis=1)
slope_df = slope_df.drop('slope_1', axis=1)
thal_df = thal_df.drop(['thal_0', 'thal_1'], axis=1)

df_train_onehot = df_train.drop(['cp', 'restecg', 'slope', 'ca', 'thal'], axis=1)
df_train_onehot = pd.concat([df_train_onehot, cp_df, restecg_df, slope_df, ca_df, thal_df], axis=1)
df_train_onehot.head()

# **age**: 나이
# **trestbps**: (resting blood pressure) 휴식 중 혈압(mmHg)
# **chol**: (serum cholestoral) 혈중 콜레스테롤 (mg/dl)
# **thalach**: (maximum heart rate achieved) 최대 심박수
# **oldpeak**: (ST depression induced by exercise relative to rest) 휴식 대비 운동으로 인한 ST 하강
#
# - 위 feature들은 연속적인 수치를 나타낸다. 이런 수치들은 0~1 사이의 값으로 rescaling 해주는 것이 바람직할 것이다.

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

# 전처리한 data frame을 dataset으로 준비한다. 

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

# 처음으로 시도해볼 모델은 6층으로 쌓은 dense 모델이다. 많지 않은 feature와 dataset에는  이런 단순한 모델이 효과적이지 않을까?

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

# validationa accuracy가 67%정도로 만족스럽지 못한 결과이다. 그래도 우선 이 모델로 submission을 생성해보자.

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

# ## Light GBM
#
# 두 번째로 시도해볼 모델은 light gbm모델이다. light gbm은 ensemble 모델 중 gradiant boosting method 계열의 각광받는 모델이다. 다른 boosting method 계열의 모델보다 계산량은 적고, 성능은 비슷하거나 앞서는, 좋은 모습을 보여주고 있는 모델이라고 한다.

# +
import lightgbm as lgb

train_data_lgb = lgb.Dataset(df_train_onehot[:train_n].values, label=target_train[:train_n].values)
val_data_lgb = train_data_lgb.create_valid(df_train_onehot[train_n:].values, label=target_train[train_n:].values)

# +
param = {'num_leaves': 31, 'objective': 'binary', 'device_type': 'gpu'}
n_round = 100

record = {}
record_eval = lgb.record_evaluation(record)
early = lgb.early_stopping(20)

bst = lgb.train(param, train_data_lgb, n_round, valid_sets=[val_data_lgb], callbacks=[record_eval, early])
bst.save_model('lgb_model.txt', num_iteration=bst.best_iteration)
print('best score: {}'.format(bst.best_score))
# -

lgb_pred = bst.predict(df_test_onehot.values)

df_pred_lgb = pd.DataFrame(data=lgb_pred, columns=['target'])
df_pred_lgb['id'] = df_pred_lgb.index + 1

# +
df_pred_lgb = df_pred_lgb[['id', 'target']]

df_pred_lgb['target'] = list(map(round, df_pred_lgb['target']))
df_pred_lgb
# -

df_pred_lgb.to_csv('submission_lgb.csv', index=False)

lgb.plot_importance(bst)

lgb.plot_metric(record)

# CV

# +
train_data_lgb_cv = lgb.Dataset(df_train_onehot.values, label=target_train.values)

record={}
bst_cv = lgb.cv(param, train_data_lgb_cv, n_round, callbacks=[early], return_cvbooster=True)
bst_cv['cvbooster']
# -

bst_cv_model = bst_cv['cvbooster']
lgb_cv_pred_ = bst_cv_model.predict(df_test_onehot.values)

lgb_cv_pred = np.zeros(152)
for pred in lgb_cv_pred_:
    lgb_cv_pred += pred
lgb_cv_pred = lgb_cv_pred/5
lgb_cv_pred

df_pred_lgb_cv = pd.DataFrame(data=lgb_cv_pred, columns=['target'])
df_pred_lgb_cv['id'] = df_pred_lgb_cv.index + 1

# +
df_pred_lgb_cv = df_pred_lgb_cv[['id', 'target']]

df_pred_lgb_cv['target'] = list(map(round, df_pred_lgb_cv['target']))
df_pred_lgb_cv
# -

df_pred_lgb_cv.to_csv('submission_lgb_cv.csv', index=False)

# ## grid search
#
# light gbm에는 정말 많은 parameter 값이 있다. 그 중 일부라도 grid search로 최적의 parameter를 찾아보자.

X_train = df_train_onehot.values
y_train = target_train.values

lgbm = lgb.LGBMClassifier()
lgbm.fit(X_train[:train_n], y_train[:train_n])
lgbm.score(X_train[:train_n], y_train[:train_n])

from sklearn.metrics import classification_report
X_test = df_test_onehot.values
pred_test = lgbm.predict(X_train[train_n:])
pred_test_ = list(map(round, np.squeeze(pred_test)))
print(classification_report(y_train[train_n:], pred_test_))

# +
from sklearn.model_selection import GridSearchCV

grid_params ={'boosting_type': ['gbdt', 'dart', 'goss', 'rf'], 'max_depth' : range(1, 20, 2) , 'n_estimators': range(50, 150, 10), 'learning_rate':[0.01, 0.1], 'random_state':[24]}
grid_search = GridSearchCV(lgb.LGBMClassifier(), grid_params, cv=5)
grid_search.fit(X_train, y_train)
# -

print('Best_params: {}'.format(grid_search.best_params_))
print('Best_score: {:.4f}'.format(grid_search.best_score_))

# grid search로 찾은 최적의 parameter들. 이것들을 적용해 새로 학습시켜보자.

# +
train_data_lgb_cv = lgb.Dataset(df_train_onehot.values, label=target_train.values)
param = {'boosting': 'dart', 'learning_rate': 0.1, 'max_depth': 5, 'num_iterations': 50, 'seed': 24, 'device_type': 'gpu'}
n_round = 100

dart_cv = lgb.cv(param, train_data_lgb_cv, n_round, return_cvbooster=True)

# +
dart_cv_model = dart_cv['cvbooster']
dart_cv_pred_ = dart_cv_model.predict(df_test_onehot.values)

dart_cv_pred = np.zeros(152)
for pred in dart_cv_pred_:
    dart_cv_pred += pred
dart_cv_pred = dart_cv_pred/5

# +
df_pred_dart_cv = pd.DataFrame(data=dart_cv_pred, columns=['target'])
df_pred_dart_cv['id'] = df_pred_dart_cv.index + 1

df_pred_dart_cv = df_pred_dart_cv[['id', 'target']]

df_pred_dart_cv['target'] = list(map(round, df_pred_dart_cv['target']))
df_pred_dart_cv
# -

df_pred_dart_cv.to_csv('submission_dart_cv.csv', index=False)

# 최종 f1 score는 0.7826. 데이터와 모델에 대한 충분한 이해가 없이 주먹구구식으로 접근했던 것 같아 아쉬움이 남는다. 다음에는 좀 더 여유를 두고 참여하여 충분한 이해를 바탕으로 만들어보자.
