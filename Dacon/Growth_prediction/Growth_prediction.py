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

# 생육 기간 예측 경진대회
# =======
#
# https://dacon.io/competitions/official/235851/overview/description
#
# KIST에서 Dacon을 통해 개최한 생육 기간 예측 경진대회에 참여해 보았다. 
# 한 쌍의 식물 이미지을 입력으로 받아 그 두 사진이 며칠 차이나는 것인지 예측하는 모델을 만드는 것이 목표이다. 주어진 식물 이미지는 총 두 종류였고, (BC: 청경채, LT: 적상추) 입력으로 받을 이미지가 어떤 식물인지는 파일명을 통해 알 수 있었다. 
#
# train 데이터는 약 40일간 각각의 식물을 키우며 하루 단위로 상태를 촬영한 것으로, 파일명을 통해 이미지가 어떤 식물의 며칠차 이미지인지 구별할 수 있었다.
#
# 두 가지 방법을 생각해보았는데, 아래와 같다.
# 1. 하나의 이미지를 입력으로 받아서 며칠차의 이미지인지 예측하는 모델을 식물별로 만들고, 이전/이후 이미지에 대해 각각 예측한 후에 그 둘의 차를 구하는 방법
# 2. 두 개의 이미지를 입력으로 받아서 두 이미지가 며칠 차이나는지 예측하는 모델을 만드는 방법.
#
# 1번의 경우 학습할 때 주어진 데이터를 직관적으로 이용할 수 있고, 모델 구성이 쉽지만 학습 데이터의 양이 적어 완성된 모델의 정확도가 비교적 낮을 것이다. 반면 2번의 경우에는 학습 데이터의 양이 NC2 = N(N-1)/2 으로 많아져 모델의 정확도가 올라갈 수 있지만 주어진 학습 데이터에서 새로운 학습 데이터를 생성하는 과정을 거쳐야 한다.
#
# 주어진 시간이 길지 않았기에 나에게 조금 더 익숙한 1번 방법을 선택했다. 

# ## import & Constructing Dataset

# +
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory

import matplotlib.pyplot as plt

import numpy as np

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='cividis')


# +
import pandas as pd
import os

path = "./data/open/train_dataset"
file_list_BC = []
file_list_LT = []
df_BC = pd.DataFrame(columns=['file_path', 'day'])
df_LT = pd.DataFrame(columns=['file_path', 'day'])
for i in range(9):
    file_path = path + "/BC" + "/BC_0{}".format(i+1)
    file_list_BC=os.listdir(file_path)
    for img in file_list_BC:
        df_BC = df_BC.append({'file_path': file_path + img, 'day': float(img[-6:-4])}, ignore_index=True)
for i in range(10):
    file_path = path + "/LT" + "/LT_0{}".format(i+1)
    if i == 9:
        file_path = path + "/LT/LT_10"
    file_list_LT=os.listdir(file_path)
    for img in file_list_LT:
        df_LT = df_LT.append({'file_path': file_path + img, 'day': float(img[-6:-4])}, ignore_index=True)

# +
ds_bc_ = keras.preprocessing.image_dataset_from_directory(
    path+'/BC',
    labels=df_BC.day.to_list(),
    label_mode='int',
    image_size=[256, 256],
    interpolation='nearest',
    batch_size=1,
    shuffle=False
)

ds_lt_ = keras.preprocessing.image_dataset_from_directory(
    path+'/LT',
    labels=df_LT.day.to_list(),
    label_mode='int',
    image_size=[256, 256],
    interpolation='nearest',
    batch_size=1,
    shuffle=False
)
# -

ds_train_bc_ = ds_bc_.take(int(353 * 0.7))
ds_val_bc_ = ds_bc_.skip(int(353*0.7))
ds_train_lt_ = ds_lt_.take(int(400 * 0.7))
ds_val_lt_ = ds_lt_.take(int(400 * 0.7))


# +
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) 
    label = tf.cast(label, tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train_bc = (
    ds_train_bc_
    .map(convert_to_float)
    .unbatch()
    .batch(8)
    .shuffle(500)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds_val_bc = (
    ds_val_bc_
    .map(convert_to_float)
    .unbatch()
    .batch(8)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds_train_lt = (
    ds_train_lt_
    .map(convert_to_float)
    .unbatch()
    .batch(8)
    .shuffle(500)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds_val_lt = (
    ds_val_lt_
    .map(convert_to_float)
    .unbatch()
    .batch(8)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
# -

# 모듈 임포트 및 데이터 세트를 구성한다. 주어진 이미지의 크기가 너무 큰 상황이므로 크기를 (256, 256)으로 줄여 불러오고, 70%는 train dataset으로, 30%는 validation dataset으로 나눠준 후 `.cache()`,`.shuffle()`,`prefetch` 등의 전처리 과정을 거친다.

# ## my_conv_model

# ### model for BC

# +
my_conv_model_bc = keras.Sequential([
    
    layers.InputLayer(input_shape=[256, 256, 3]),
    
    preprocessing.RandomContrast(factor=0.2),
    preprocessing.RandomFlip(mode='horizontal_and_vertical'),
    preprocessing.RandomRotation(factor=0.05),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),

    layers.BatchNormalization(),
    layers.Conv2D(filters=16, kernel_size=7, activation='relu', padding='same'),
    layers.Conv2D(filters=32, kernel_size=7, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(filters=128, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    layers.BatchNormalization(),
    layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalAveragePooling2D(),
    
    layers.BatchNormalization(),
    layers.Dense(16, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(16, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='relu')
    
])
# -

# 가장 처음으로 시도해본 모델이다. 부족한 데이터셋을 보완하기 위해 Data Augmentation layer부터 시작하여 convolution층을 쌓았다.

my_conv_model_bc.compile(
    optimizer='adam',
    loss='mse'
)

# +
early = keras.callbacks.EarlyStopping(patience=30, min_delta=0.001, restore_best_weights=True)

history_bc = my_conv_model_bc.fit(
    ds_train_bc,
    validation_data=ds_val_bc,
    callbacks=[early],
    epochs=200
)
# -

history_bc_df = pd.DataFrame(history_bc.history)
history_bc_df.plot()
history_bc_df.val_loss.min()

# ### model for LT

# +
my_conv_model_lt = keras.Sequential([
    
    layers.InputLayer(input_shape=[256, 256, 3]),
    
    preprocessing.RandomContrast(factor=0.2),
    preprocessing.RandomFlip(mode='horizontal_and_vertical'),
    preprocessing.RandomRotation(factor=0.05),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),

    layers.BatchNormalization(),
    layers.Conv2D(filters=16, kernel_size=7, activation='relu', padding='same'),
    layers.Conv2D(filters=32, kernel_size=7, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(filters=128, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalAveragePooling2D(),
    
    layers.BatchNormalization(),
    layers.Dense(16, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(16, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='relu')
    
])
# -

my_conv_model_lt.compile(
    optimizer='adam',
    loss='mse'
)

history_lt = my_conv_model_lt.fit(
    ds_train_lt,
    validation_data=ds_val_lt,
    callbacks=[early],
    epochs=200
)

history_lt_df = pd.DataFrame(history_lt.history)
history_lt_df.plot()
history_lt_df.val_loss.min()

test_data = pd.read_csv('./data/open/test_dataset/test_data.csv')
test_data.head()

# ### prediction

# +
bf_pred = []
af_pred = []

for i in range(3960):
    split_before = test_data.loc[i, 'before_file_path'].split('_')
    split_after = test_data.loc[i, 'after_file_path'].split('_')
    
    before_file_path = './data/open/test_dataset/{}/{}/{}.png'.format(split_before[1], split_before[2], test_data.loc[i, 'before_file_path'])
    after_file_path = './data/open/test_dataset/{}/{}/{}.png'.format(split_after[1], split_after[2], test_data.loc[i, 'after_file_path'])
    
    before_img = keras.preprocessing.image.load_img(before_file_path, target_size=[256, 256])
    after_img = keras.preprocessing.image.load_img(after_file_path, target_size=[256, 256])
    
    before_img = tf.image.convert_image_dtype(before_img, dtype=tf.float32)
    after_img = tf.image.convert_image_dtype(after_img, dtype=tf.float32)
    before_img = tf.expand_dims(before_img, 0)
    after_img = tf.expand_dims(after_img, 0)
    
    if (split_before[1] == 'LT'):
        bf_pred.append(my_conv_model_lt(before_img))
        af_pred.append(my_conv_model_lt(after_img))
    else:
        bf_pred.append(my_conv_model_bc(before_img))
        af_pred.append(my_conv_model_bc(after_img))
        
    
    
    
# -

# 학습된 모델들을 이용해 이제 제출할 파일을 준비한다. test_data 파일에서 전/후 이미지 경로를 가져온 후 각각의 이미지를 불러와 학습시킨 모델로 예측한다. 그 결과는 각각 bf_pred, af_pred에 저장한다. 

a = tf.squeeze(bf_pred)
b = tf.squeeze(af_pred)

pred = b-a

pred_np = pred.numpy()

pred_np

a = [x.numpy() for x in bf_pred]
b = [x.numpy() for x in af_pred]

test_data['time_delta'] = pred_np
test_data.head()

sub = test_data.loc[:,['idx', 'time_delta']].to_csv('./submission_3.csv', index=False)

# bf_pred와 af_pred의 차이를 계산하여 최종 결과를 산출하고 파일로 저장한다.

test_data['time_delta_round'] = round(test_data['time_delta'])

sub = test_data.loc[:, ['idx', 'time_delta_round']]
sub = sub.rename(columns={'time_delta_round': 'time_delta'})
sub.head()

sub.to_csv('./submission_round_2.csv', index=False)

# 반올림한 결과도 제출해본다.

# ## Feature vectors of images with Inception V3 trained on the iNaturalist (iNat) 2017 dataset.
#
# pretrained Inception V3 모델을 이용한 모델이다.

# ### model for BC

# +
import tensorflow_hub as hub

model_InceptionV3_bc = keras.Sequential([
    layers.InputLayer(input_shape=[256, 256, 3]),
    
    preprocessing.RandomContrast(factor=0.2),
    preprocessing.RandomFlip(mode='horizontal_and_vertical'),
    preprocessing.RandomRotation(factor=0.05),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
    
    hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", trainable=False),
    layers.Dense(1, activation='relu')
])
# -

model_InceptionV3_bc.compile(
    optimizer='adam',
    loss='mse'
)

# +
early = keras.callbacks.EarlyStopping(patience=30, min_delta=0.001, restore_best_weights=True)

history_inceptionV3_bc = model_InceptionV3_bc.fit(
    ds_train_bc,
    validation_data=ds_val_bc,
    callbacks=[early],
    epochs=200
)
# -

df_history_inceptionV3_bc = pd.DataFrame(history_inceptionV3_bc.history)
df_history_inceptionV3_bc.plot()
print(df_history_inceptionV3_bc.val_loss.min())

# ### model for LT

model_InceptionV3_lt = keras.Sequential([
    layers.InputLayer(input_shape=[256, 256, 3]),
    
    preprocessing.RandomContrast(factor=0.2),
    preprocessing.RandomFlip(mode='horizontal_and_vertical'),
    preprocessing.RandomRotation(factor=0.05),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
    
    hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", trainable=False),
    layers.Dense(1, activation='relu')
])

model_InceptionV3_lt.compile(
    optimizer='adam',
    loss='mse'
)

history_inceptionV3_lt = model_InceptionV3_lt.fit(
    ds_train_lt,
    validation_data=ds_val_lt,
    callbacks=[early],
    epochs=200
)

df_history_inceptionV3_lt = pd.DataFrame(history_inceptionV3_lt.history)
df_history_inceptionV3_lt.plot()
print(df_history_inceptionV3_lt.val_loss.min())

# ### prediction

# +
bf_pred = []
af_pred = []

for i in range(3960):
    split_before = test_data.loc[i, 'before_file_path'].split('_')
    split_after = test_data.loc[i, 'after_file_path'].split('_')
    
    before_file_path = './data/open/test_dataset/{}/{}/{}.png'.format(split_before[1], split_before[2], test_data.loc[i, 'before_file_path'])
    after_file_path = './data/open/test_dataset/{}/{}/{}.png'.format(split_after[1], split_after[2], test_data.loc[i, 'after_file_path'])
    
    before_img = keras.preprocessing.image.load_img(before_file_path, target_size=[256, 256])
    after_img = keras.preprocessing.image.load_img(after_file_path, target_size=[256, 256])
    
    before_img = tf.image.convert_image_dtype(before_img, dtype=tf.float32)
    after_img = tf.image.convert_image_dtype(after_img, dtype=tf.float32)
    before_img = tf.expand_dims(before_img, 0)
    after_img = tf.expand_dims(after_img, 0)
    
    if (split_before[1] == 'LT'):
        bf_pred.append(model_InceptionV3_lt(before_img))
        af_pred.append(model_InceptionV3_lt(after_img))
    else:
        bf_pred.append(model_InceptionV3_bc(before_img))
        af_pred.append(model_InceptionV3_bc(after_img))
        
    
    
# -

pred = tf.squeeze(af_pred) - tf.squeeze(bf_pred)
pred_np = pred.numpy()
print(pred_np)

sub_inception = test_data
sub_inception['time_delta'] = pred_np
sub_inception.loc[sub_inception.time_delta < 1, 'time_delta'] = 1
sub_inception.head()

sub_inception.loc[:, ['idx', 'time_delta']].to_csv('./submission_inceptionv3_1.csv', index=False)

# ## EfficientNet
#
# Feature vectors of images with EfficientNet V2 with input size 480x480, trained on imagenet-ilsvrc-2012-cls (ILSVRC-2012-CLS).
# Google의 pretrained EfficientNet 모델을 이용해보았다.

# ### Constructing Dataset
#  
# 모델에 최적화된 input size에 맞게 다시 데이터셋을 불러온다.

# +
ds2_bc_ = keras.preprocessing.image_dataset_from_directory(
    path+'/BC',
    labels=df_BC.day.to_list(),
    label_mode='int',
    image_size=[480, 480],
    interpolation='nearest',
    batch_size=1,
    shuffle=False
)

ds2_lt_ = keras.preprocessing.image_dataset_from_directory(
    path+'/LT',
    labels=df_LT.day.to_list(),
    label_mode='int',
    image_size=[480, 480],
    interpolation='nearest',
    batch_size=1,
    shuffle=False
)
# -

ds2_train_bc_ = ds2_bc_.take(int(353 * 0.7))
ds2_val_bc_ = ds2_bc_.skip(int(353*0.7))
ds2_train_lt_ = ds2_lt_.take(int(400 * 0.7))
ds2_val_lt_ = ds2_lt_.take(int(400 * 0.7))


# +
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) 
    label = tf.cast(label, tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds2_train_bc = (
    ds2_train_bc_
    .map(convert_to_float)
    .unbatch()
    .batch(8)
    .shuffle(500)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds2_val_bc = (
    ds2_val_bc_
    .map(convert_to_float)
    .unbatch()
    .batch(8)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds2_train_lt = (
    ds2_train_lt_
    .map(convert_to_float)
    .unbatch()
    .batch(8)
    .shuffle(500)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds2_val_lt = (
    ds2_val_lt_
    .map(convert_to_float)
    .unbatch()
    .batch(8)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
# -

# ### model for BC

model_EfficientNet_bc = keras.Sequential([
    layers.InputLayer(input_shape=[480, 480, 3]),
    
    preprocessing.RandomContrast(factor=0.2),
    preprocessing.RandomFlip(mode='horizontal_and_vertical'),
    preprocessing.RandomRotation(factor=0.05),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
    
    hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2", trainable=False),
    layers.Dense(1, activation='relu')
])

model_EfficientNet_bc.compile(
    optimizer='adam',
    loss='mse'
)

history_EfficientNet_bc = model_EfficientNet_bc.fit(
    ds2_train_bc,
    validation_data=ds2_val_bc,
    callbacks=[early],
    epochs=200
)

df_history_EfficientNet_bc = pd.DataFrame(history_EfficientNet_bc.history)
df_history_EfficientNet_bc.plot()
print(df_history_EfficientNet_bc.val_loss.min())

# ### model for LT

model_EfficientNet_lt = keras.Sequential([
    layers.InputLayer(input_shape=[480, 480, 3]),
    
    preprocessing.RandomContrast(factor=0.2),
    preprocessing.RandomFlip(mode='horizontal_and_vertical'),
    preprocessing.RandomRotation(factor=0.05),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
    
    hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2", trainable=False),
    layers.Dense(1, activation='relu')
])

model_EfficientNet_lt.compile(
    optimizer='adam',
    loss='mse'
)

history_EfficientNet_lt = model_EfficientNet_lt.fit(
    ds2_train_lt,
    validation_data=ds2_val_lt,
    callbacks=[early],
    epochs=200
)

df_history_EfficientNet_lt = pd.DataFrame(history_EfficientNet_lt.history)
df_history_EfficientNet_lt.plot()
print(df_history_EfficientNet_lt.val_loss.min())

# ### prediction

# +
bf_pred = []
af_pred = []

for i in range(3960):
    split_before = test_data.loc[i, 'before_file_path'].split('_')
    split_after = test_data.loc[i, 'after_file_path'].split('_')
    
    before_file_path = './data/open/test_dataset/{}/{}/{}.png'.format(split_before[1], split_before[2], test_data.loc[i, 'before_file_path'])
    after_file_path = './data/open/test_dataset/{}/{}/{}.png'.format(split_after[1], split_after[2], test_data.loc[i, 'after_file_path'])
    
    before_img = keras.preprocessing.image.load_img(before_file_path, target_size=[480, 480])
    after_img = keras.preprocessing.image.load_img(after_file_path, target_size=[480, 480])
    
    before_img = tf.image.convert_image_dtype(before_img, dtype=tf.float32)
    after_img = tf.image.convert_image_dtype(after_img, dtype=tf.float32)
    before_img = tf.expand_dims(before_img, 0)
    after_img = tf.expand_dims(after_img, 0)
    
    if (split_before[1] == 'LT'):
        bf_pred.append(model_EfficientNet_lt(before_img))
        af_pred.append(model_EfficientNet_lt(after_img))
    else:
        bf_pred.append(model_EfficientNet_bc(before_img))
        af_pred.append(model_EfficientNet_bc(after_img))
        
    
    
# -

pred = tf.squeeze(af_pred) - tf.squeeze(bf_pred)
pred_np = pred.numpy()
print(pred_np)

sub = test_data
sub['time_delta'] = pred_np
sub.head()

sub.loc[sub['time_delta'] <= 1, 'time_delta'] = 1
sub.head()

sub.loc[:, ['idx', 'time_delta']].to_csv('./submission_EfficientNet.csv', index=False)

# ## Center Crop
#
# 주어진 이미지를 보니, 정중앙의 식물 하나만 이용하는 것이 학습에 더 도움이 될 것이라는 생각이 들었다. 화분 및 도구 등 필요 없는 부분이 이미지의 양 옆에 많이 포함되어 있었기 때문이다. Center Crop layer를 모델에 추가하여 이 생각을 적용해보았다.

# ### Constructing Dataset
#
# Center Crop을 적용하기 적당한 사이즈로 불러온다.

# +
ds3_bc_ = keras.preprocessing.image_dataset_from_directory(
    path+'/BC',
    labels=df_BC.day.to_list(),
    label_mode='int',
    image_size=[821, 1093],
    interpolation='nearest',
    batch_size=1,
    shuffle=False
)

ds3_lt_ = keras.preprocessing.image_dataset_from_directory(
    path+'/LT',
    labels=df_LT.day.to_list(),
    label_mode='int',
    image_size=[821, 1093],
    interpolation='nearest',
    batch_size=1,
    shuffle=False
)
# -

ds3_train_bc_ = ds3_bc_.take(int(353 * 0.7))
ds3_val_bc_ = ds3_bc_.skip(int(353*0.7))
ds3_train_lt_ = ds3_lt_.take(int(400 * 0.7))
ds3_val_lt_ = ds3_lt_.take(int(400 * 0.7))

# +
ds3_train_bc = (
    ds3_train_bc_
    .map(convert_to_float)
    .unbatch()
    .batch(8)
    .shuffle(500)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds3_val_bc = (
    ds3_val_bc_
    .map(convert_to_float)
    .unbatch()
    .batch(8)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds3_train_lt = (
    ds3_train_lt_
    .map(convert_to_float)
    .unbatch()
    .batch(8)
    .shuffle(500)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds3_val_lt = (
    ds3_val_lt_
    .map(convert_to_float)
    .unbatch()
    .batch(8)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
# -

# ### model for BC

model_crop_bc = keras.Sequential([
    layers.InputLayer(input_shape=[821, 1093, 3]),
    layers.CenterCrop(299, 299),
    
    preprocessing.RandomContrast(factor=0.2),
    preprocessing.RandomFlip(mode='horizontal_and_vertical'),
    preprocessing.RandomRotation(factor=0.05),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
    
    hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", trainable=False),
    layers.Dense(1, activation='relu')
])

model_crop_bc.compile(
    optimizer='adam',
    loss='mse'
)

history_crop_bc = model_crop_bc.fit(
    ds3_train_bc,
    validation_data=ds3_val_bc,
    callbacks=[early],
    epochs=200
)

df_history_crop_bc = pd.DataFrame(history_crop_bc.history)
df_history_crop_bc.plot()
print(df_history_crop_bc.val_loss.min())

# ### model for LT

model_crop_lt = keras.Sequential([
    layers.InputLayer(input_shape=[821, 1093, 3]),
    layers.CenterCrop(299, 299),
    
    preprocessing.RandomContrast(factor=0.2),
    preprocessing.RandomFlip(mode='horizontal_and_vertical'),
    preprocessing.RandomRotation(factor=0.05),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
    
    hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", trainable=False),
    layers.Dense(1, activation='relu')
])

model_crop_lt.compile(
    optimizer='adam',
    loss='mse'
)

history_crop_lt = model_crop_lt.fit(
    ds3_train_lt,
    validation_data=ds3_val_lt,
    callbacks=[early],
    epochs=200
)

df_history_crop_lt = pd.DataFrame(history_crop_lt.history)
df_history_crop_lt.plot()
print(df_history_crop_lt.val_loss.min())

# ### prediction

# +
bf_pred = []
af_pred = []

for i in range(3960):
    split_before = test_data.loc[i, 'before_file_path'].split('_')
    split_after = test_data.loc[i, 'after_file_path'].split('_')
    
    before_file_path = './data/open/test_dataset/{}/{}/{}.png'.format(split_before[1], split_before[2], test_data.loc[i, 'before_file_path'])
    after_file_path = './data/open/test_dataset/{}/{}/{}.png'.format(split_after[1], split_after[2], test_data.loc[i, 'after_file_path'])
    
    before_img = keras.preprocessing.image.load_img(before_file_path, target_size=[821, 1093])
    after_img = keras.preprocessing.image.load_img(after_file_path, target_size=[821, 1093])
    
    before_img = tf.image.convert_image_dtype(before_img, dtype=tf.float32)
    after_img = tf.image.convert_image_dtype(after_img, dtype=tf.float32)
    before_img = tf.expand_dims(before_img, 0)
    after_img = tf.expand_dims(after_img, 0)
    
    if (split_before[1] == 'LT'):
        bf_pred.append(model_crop_lt(before_img))
        af_pred.append(model_crop_lt(after_img))
    else:
        bf_pred.append(model_crop_bc(before_img))
        af_pred.append(model_crop_bc(after_img))
        
    
    
# -

pred = tf.squeeze(af_pred) - tf.squeeze(bf_pred)
pred_np = pred.numpy()
print(pred_np)

sub_crop = test_data
sub_crop['time_delta'] = pred_np
sub_crop.loc[sub_inception.time_delta < 1, 'time_delta'] = 1
sub_crop.head()

sub_inception.loc[:, ['idx', 'time_delta']].to_csv('./submission_crop.csv', index=False)

# ## Result

# 가장 좋은 예측을 한 모델은 Center Crop을 적용한 모델이었다. 최종 점수는 8.3315319077. 첫 제출 당시 점수 11.9935508021에 비하면 많은 발전이 있었지만 아직 많이 부족하다. 최종 순위는 86위. 아쉬운 결과이다.
#
# - 모델의 구조 및 그에 따른 데이터 양의 한계가 가장 아쉽다. 시작할 당시 더 효과적일 것이라고 생각한 1번 방식 보다는 2번 방식이 옳은 길이었다는 생각이 든다. 이렇게 모델을 따로 만드는 방식은 사용할 수 있는 데이터가 400개 이하로, 데이터의 개수가 너무 적었다. Data Augmentation으로 데이터를 보강해보기도 했지만 그것으로는 부족했던 것으로 보인다.
# - 식물 개체에 따른 성장의 차이가 생각보다 컸다. 이런 점은 학습 데이터의 양이 부족한 이번 환경에서 더 큰 단점이 되었고, 학습에서 큰 걸림돌이 되었을 것이다.
