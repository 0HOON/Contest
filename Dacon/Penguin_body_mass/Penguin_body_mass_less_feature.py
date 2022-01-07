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

df_train = pd.read_csv("./train_droped.csv", index_col="id")
df_test = pd.read_csv("./test_filled.csv", index_col="id")

# +
df_train_target = pd.read_csv("./train_target.csv", index_col="id")
df_test_target = pd.read_csv("./test_target.csv", index_col="id")
df_train_y = pd.read_csv('./train_y.csv', index_col="id").squeeze()

body_max = df_train["Body Mass (g)"].max()
body_min = df_train["Body Mass (g)"].min()

df_train_y_rescaled = (df_train_y - body_min) / (body_max - body_min)

df_train_target.head()
# -

train_n = int(df_train_target.count()[0]*0.8)

df_train_target.drop(["Island", "Clutch Completion"], axis=1, inplace=True)

df_test_target.drop(["Island", "Clutch Completion"], axis=1, inplace=True)

df_train_target.loc[df_train_target["Sex"] > 0.5, "Sex"] = 1
df_train_target.loc[df_train_target["Sex"] < 0.5, "Sex"] = 0
df_test_target.loc[df_test_target["Sex"] > 0.5, "Sex"] = 1
df_test_target.loc[df_test_target["Sex"] < 0.5, "Sex"] = 0
df_train_target

# +
from itertools import combinations

def min_max_scale_2(df, col, col_min, col_max):
    df[col] = ( df[col] - col_min) / (col_max - col_min)

cnt_cols = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Delta 15 N (o/oo)", "Delta 13 C (o/oo)"]
cat_cols = ["Species", "Island", "Sex", "Clutch Completion"]


for i, col in enumerate(combinations(cnt_cols, 2)):
    df_train_target["m_{}_{}".format(col[0], col[1])] = df_train[col[0]] * df_train[col[1]]
    df_train_target["d_{}_{}".format(col[0], col[1])] = df_train[col[0]] / df_train[col[1]]
    df_test_target["m_{}_{}".format(col[0], col[1])] = df_test[col[0]] * df_test[col[1]]
    df_test_target["d_{}_{}".format(col[0], col[1])] = df_test[col[0]] / df_test[col[1]]
    
    col_max_m = df_train_target["m_{}_{}".format(col[0], col[1])].max()
    col_max_d = df_train_target["d_{}_{}".format(col[0], col[1])].max()
    col_min_m = df_train_target["m_{}_{}".format(col[0], col[1])].min()
    col_min_d = df_train_target["d_{}_{}".format(col[0], col[1])].min()
    
    min_max_scale_2(df_train_target, "m_{}_{}".format(col[0], col[1]), col_min_m, col_max_m)
    min_max_scale_2(df_train_target, "d_{}_{}".format(col[0], col[1]), col_min_d, col_max_d)
    min_max_scale_2(df_test_target, "m_{}_{}".format(col[0], col[1]), col_min_m, col_max_m)
    min_max_scale_2(df_test_target, "d_{}_{}".format(col[0], col[1]), col_min_d, col_max_d)

    
for i, col in enumerate(cnt_cols):
    df_train_target["p_{}".format(i)] = df_train[col] * df_train[col]
    df_test_target["p_{}".format(i)] = df_test[col] * df_test[col]

    col_max_p = df_train_target["p_{}".format(i)].max()
    col_min_p = df_train_target["p_{}".format(i)].min()

    min_max_scale_2(df_train_target, "p_{}".format(i), col_min_p, col_max_p)
    min_max_scale_2(df_test_target, "p_{}".format(i), col_min_p, col_max_p)

# -

for cnt_col in cnt_cols:
    for col in ["Species", "Sex"]:
        df_train_target["m_{}_{}".format(cnt_col, col)] = df_train_target[cnt_col] * df_train_target[col]
#        df_train_target["d_{}_{}".format(cnt_col, col)] = df_train_target[cnt_col] / df_train_target[col]
        df_test_target["m_{}_{}".format(cnt_col, col)] = df_test_target[cnt_col] * df_test_target[col]
#        df_test_target["d_{}_{}".format(cnt_col, col)] = df_test_target[cnt_col] / df_test_target[col]

df_train_target["p_Sp"] = df_train_target["Species"] * df_train_target["Species"]
df_test_target["p_Sp"] = df_test_target["Species"] * df_test_target["Species"]

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor

# +
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_train_selected, df_train_y, test_size=0.1)

# +
score_List=[]

model = KNeighborsRegressor(n_neighbors=5).fit(x_train, y_train)
score_List.append(f'KNeighborsRegressor: train_score: {model.score(x_train, y_train)}, val_score: {model.score(x_test, y_test)}')

model = LinearRegression().fit(x_train, y_train)
score_List.append(f'LinearRegression: train_score: {model.score(x_train, y_train)}, val_score: {model.score(x_test, y_test)}')

model = Ridge().fit(x_train, y_train)
score_List.append(f'Ridge: train_score: {model.score(x_train, y_train)}, val_score: {model.score(x_test, y_test)}')

model = Lasso().fit(x_train, y_train)
score_List.append(f'Lasso: train_score: {model.score(x_train, y_train)}, val_score: {model.score(x_test, y_test)}')

model = DecisionTreeRegressor().fit(x_train, y_train)
score_List.append(f'DecisionTreeRegressor: train_score: {model.score(x_train, y_train)}, val_score: {model.score(x_test, y_test)}')

model = RandomForestRegressor().fit(x_train, y_train)
score_List.append(f'RandomForestRegressor: train_score: {model.score(x_train, y_train)}, val_score: {model.score(x_test, y_test)}')

model = GradientBoostingRegressor().fit(x_train, y_train)
score_List.append(f'GradientBoostingRegressor: train_score: {model.score(x_train, y_train)}, val_score: {model.score(x_test, y_test)}')

model = LGBMRegressor().fit(x_train, y_train)
score_List.append(f'LGBMRegressor: train_score: {model.score(x_train, y_train)}, val_score: {model.score(x_test, y_test)}')
# -

[print(i) for i in score_List]

# +
X = df_train_selected
Y = df_train_y

model_LR = LinearRegression().fit(X, Y)
model_RID = Ridge().fit(X, Y)
model_LA = Lasso().fit(X, Y)
model_GBR = GradientBoostingRegressor().fit(X,Y)
model_LGBM = LGBMRegressor().fit(X, Y)
# -

pred_LR = model_LR.predict(df_test_selected)
pred_RID = model_RID.predict(df_test_selected)
pred_LA = model_LA.predict(df_test_selected)
pred_LGBM = model_LGBM.predict(df_test_selected)
pred_GBR = model_RF.predict(df_test_selected)
pred_en = pred_KN*0.2 + pred_RID*0.2 + pred_LA*0.2 + pred_LGBM*0.2 + pred_RF*0.2
pred_en

df_pred = pd.DataFrame(pred, columns=["Body Mass (g)"])
df_pred.index.name = "id"
df_pred.to_csv("sub_en.csv")

df_train_target

df_test_target.to_csv("aaaaa.csv")

# +
import lightgbm as lgb

train_data_lgb = lgb.Dataset(df_train_target[:train_n].values, label=df_train_y[:train_n].values)
val_data_lgb = train_data_lgb.create_valid(df_train_target[train_n:].values, label=df_train_y[train_n:].values)

# +
param = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 110, 'random_state': 30,
         'objective': 'mse', 'device_type': 'gpu', 'verbosity' : -1}
n_round = 100

record = {}
record_eval = lgb.record_evaluation(record)
early = lgb.early_stopping(20)

bst = lgb.train(param, train_data_lgb, n_round, valid_sets=[val_data_lgb], callbacks=[record_eval, early])
bst.save_model('lgb_model.txt', num_iteration=bst.best_iteration)
print('best score: {}'.format(bst.best_score))
# -

lgb.plot_importance(bst)

bst.feature_importance()

df_train_target.columns[3]



df_train_target.columns[bst.feature_importance() > 0]

lgb.plot_metric(record)

df_train_target





def get_mse(pred, y):
    rmse = np.sqrt(np.multiply((y-pred), (y-pred))) / len(pred)
    return sum(rmse)


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
# -

cv_pred = cv_pred / (NFOLDS * num_seeds)
df_cv_pred = pd.DataFrame(cv_pred, columns=["Body Mass (g)"])
df_cv_pred.index.name = "id"
df_cv_pred


# ## Lasso

def lasso_model():
    
    model = keras.Sequential([
        layers.Dense(512, input_shape=[43], kernel_initializer=keras.initializers.he_normal(), kernel_regularizer='l1'),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, kernel_initializer=keras.initializers.he_normal(), kernel_regularizer='l1'),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='relu')

    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    
    return model


# +
from sklearn.model_selection import StratifiedKFold

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

num_seeds = 5
l1_cv_train = np.zeros(len(df_train_target))
l1_cv_pred = np.zeros(len(df_test_target))

early = keras.callbacks.EarlyStopping(patience=300, restore_best_weights=True)

for s in range(num_seeds):
    
    np.random.seed(s)

    for (tr_idx, te_idx) in kfold.split(df_train_target, df_train_y):
        ds_train = mk_Dataset(df_train_target.loc[tr_idx, :], df_train_y.loc[tr_idx])
        ds_test = mk_Dataset(df_train_target.loc[te_idx, :], df_train_y.loc[te_idx])
        
        model = lasso_model()
        
        model.fit(
            ds_train,
            validation_data=ds_test,
            verbose=0,
            callbacks=[early],
            epochs=2000
        )
        
        l1_cv_train[te_idx] += np.squeeze(model.predict(df_train_target.loc[te_idx, :].values.astype(float)))
        print("mse: {}".format(get_mse( l1_cv_train[te_idx] / (s + 1) , df_train_y[te_idx]) ))

        l1_cv_pred += np.squeeze(model.predict(df_test_target.values.astype(float)))

    print("-----------------")
    print("seed{}_mse: {}".format(s, get_mse(l1_cv_train / (s + 1), df_train_y)))
    print("-----------------")
# -

l1_cv_pred = l1_cv_pred / (NFOLDS * num_seeds)
df_l1_cv_pred = pd.DataFrame(l1_cv_pred, columns=["Body Mass (g)"])
df_l1_cv_pred.index.name = "id"
df_l1_cv_pred


# ## Ridge

def ridge_model():
    
    model = keras.Sequential([
        layers.Dense(512, input_shape=[43], kernel_initializer=keras.initializers.he_normal(), kernel_regularizer='l2'),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, kernel_initializer=keras.initializers.he_normal(), kernel_regularizer='l2'),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='relu')

    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    
    return model


# +
from sklearn.model_selection import StratifiedKFold

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

num_seeds = 5
l2_cv_train = np.zeros(len(df_train_target))
l2_cv_pred = np.zeros(len(df_test_target))

early = keras.callbacks.EarlyStopping(patience=300, restore_best_weights=True)

for s in range(num_seeds):
    
    np.random.seed(s)

    for (tr_idx, te_idx) in kfold.split(df_train_target, df_train_y):
        ds_train = mk_Dataset(df_train_target.loc[tr_idx, :], df_train_y.loc[tr_idx])
        ds_test = mk_Dataset(df_train_target.loc[te_idx, :], df_train_y.loc[te_idx])
        
        model = ridge_model()
        
        model.fit(
            ds_train,
            validation_data=ds_test,
            verbose=0,
            callbacks=[early],
            epochs=2000
        )
        
        l2_cv_train[te_idx] += np.squeeze(model.predict(df_train_target.loc[te_idx, :].values.astype(float)))
        print("mse: {}".format(get_mse( l2_cv_train[te_idx] / (s + 1) , df_train_y[te_idx]) ))

        l2_cv_pred += np.squeeze(model.predict(df_test_target.values.astype(float)))

    print("-----------------")
    print("seed{}_mse: {}".format(s, get_mse(l2_cv_train / (s + 1), df_train_y)))
    print("-----------------")
# -

l2_cv_pred = l2_cv_pred / (NFOLDS * num_seeds)
df_l2_cv_pred = pd.DataFrame(l2_cv_pred, columns=["Body Mass (g)"])
df_l2_cv_pred.index.name = "id"
df_l2_cv_pred


# ## Lasso & Ridge

def lr_model():
    
    model = keras.Sequential([
        layers.Dense(512, input_shape=[43], kernel_initializer=keras.initializers.he_normal() 
                     ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, kernel_initializer=keras.initializers.he_normal()
                     ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='relu')

    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    
    return model


# +
from sklearn.model_selection import StratifiedKFold

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

num_seeds = 5
lr_cv_train = np.zeros(len(df_train_target))
lr_cv_pred = np.zeros(len(df_test_target))

early = keras.callbacks.EarlyStopping(patience=300, restore_best_weights=True)

for s in range(num_seeds):
    
    np.random.seed(s)

    for (tr_idx, te_idx) in kfold.split(df_train_target, df_train_y):
        ds_train = mk_Dataset(df_train_target.loc[tr_idx, :], df_train_y.loc[tr_idx])
        ds_test = mk_Dataset(df_train_target.loc[te_idx, :], df_train_y.loc[te_idx])
        
        model = lr_model()
        
        model.fit(
            ds_train,
            validation_data=ds_test,
            verbose=0,
            callbacks=[early],
            epochs=2000
        )
        
        lr_cv_train[te_idx] += np.squeeze(model.predict(df_train_target.loc[te_idx, :].values.astype(float)))
        print("mse: {}".format(get_mse( lr_cv_train[te_idx] / (s + 1) , df_train_y[te_idx]) ))

        lr_cv_pred += np.squeeze(model.predict(df_test_target.values.astype(float)))

    print("-----------------")
    print("seed{}_mse: {}".format(s, get_mse(lr_cv_train / (s + 1), df_train_y)))
    print("-----------------")
# -

lr_cv_pred = lr_cv_pred / (NFOLDS * num_seeds)
df_lr_cv_pred = pd.DataFrame(lr_cv_pred, columns=["Body Mass (g)"])
df_lr_cv_pred.index.name = "id"
df_lr_cv_pred

# ## lgb

# +
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

param = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 31, 'random_state': 30,
         'objective': 'mse', 'device_type': 'gpu', 'verbosity' : -1, 'feature_fraction': 0.8}

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

cv_pred_mix = cv_pred * 0.2 + l1_cv_pred * 0.2 + l2_cv_pred * 0.2 + lr_cv_pred * 0.2 + cv_pred_lgb * 0.2
df_cv_mix = pd.DataFrame(cv_pred_mix, columns=["Body Mass (g)"])
df_cv_mix.index.name = "id"
df_cv_mix

df_cv_mix.to_csv("submission_cv_mix_lr.csv")







# +
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

X = df_train_target
y = df_train_y

selector = SequentialFeatureSelector(
    Ridge(), 
    scoring='neg_mean_squared_error', 
    k_features=20, 
    forward=False, 
    floating=True,
    cv=10
)

selector = selector.fit(X, y)
fig = plot_sfs(selector.get_metric_dict(), kind='std_err')

# -

selector.k_feature_names_

df_train_selected = df_train_target.loc[:, selector.k_feature_names_]
df_test_selected = df_test_target.loc[:, selector.k_feature_names_]
df_train_selected









# +
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

X = df_train_target
y = df_train_y

selector = SequentialFeatureSelector(
    Ridge(), 
    scoring='neg_mean_squared_error', 
    k_features=20, 
    forward=False, 
    floating=True,
    cv=10
)

selector = selector.fit(X, y)
fig = plot_sfs(selector.get_metric_dict(), kind='std_err')

# -

selector.k_feature_names_

list(range(7, 15, 2))

loss_list = []
for i in range(7, 15, 2):
    important_col = selector.k_feature_names_[:i]
    df_train_selected = df_train_target.loc[:, important_col]
    df_test_selected = df_test_target.loc[:, important_col]

    
    ds_train = mk_Dataset(df_train_selected[:train_n], df_train_y[:train_n])
    ds_val = mk_Dataset(df_train_selected[train_n:], df_train_y[train_n:])

    sel_model = keras.Sequential([
        layers.Dense(512, kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='relu')

    ])

    sel_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")

    sel_hist = sel_model.fit(
        ds_train,
        validation_data = ds_val,
        callbacks=[early],
        epochs=2000
    )
    
    loss_list.append((i, min(sel_hist.history["val_loss"])))
print(loss_list)

print(loss_list)

important_col = selector.k_feature_names_[:15]
df_train_selected = df_train_target.loc[:, important_col]
df_test_selected = df_test_target.loc[:, important_col]

# +
early = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

ds_train = mk_Dataset(df_train_target[:train_n], df_train_y_rescaled[:train_n])
ds_val = mk_Dataset(df_train_target[train_n:], df_train_y_rescaled[train_n:])

sel_model = keras.Sequential([
        layers.Dense(512, kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.75),

        layers.Dense(128, kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='sigmoid')

    ])

sel_model.compile(optimizer='adam', loss="mse")

sel_hist = sel_model.fit(
    ds_train,
    validation_data = ds_val,
    callbacks=[early],
    epochs=2000
)
# -

history_df = pd.DataFrame(sel_hist.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Lowest validation loss: {}".format(history_df.val_loss.min()))

get_mse(np.squeeze(sel_model.predict(df_train_selected)), df_train_y)

get_mse(np.squeeze(sel_model.predict(df_train_target)), df_train_y)

df_train_target



def get_mse(pred, y):
    pred = (pred * (body_max - body_min)) + body_min
    rmse = np.sqrt(np.multiply((y-pred), (y-pred))) / len(pred)
    return sum(rmse)


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
        layers.Dense(512, input_shape=[15], kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.75),

        layers.Dense(128, kernel_initializer=keras.initializers.he_normal()),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='sigmoid')

    ])
    
    model.compile(optimizer='adam', loss="mse")
    
    return model


# +
from sklearn.model_selection import StratifiedKFold

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

num_seeds = 5
cv_train = np.zeros(len(df_train_selected))
cv_pred = np.zeros(len(df_test_selected))

early = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

for s in range(num_seeds):
    
    np.random.seed(s)

    for (tr_idx, te_idx) in kfold.split(df_train_selected, df_train_y):
        ds_train = mk_Dataset(df_train_selected.loc[tr_idx, :], df_train_y_rescaled.loc[tr_idx])
        ds_test = mk_Dataset(df_train_selected.loc[te_idx, :], df_train_y_rescaled.loc[te_idx])
        
        model = Dense_model()
        
        model.fit(
            ds_train,
            validation_data=ds_test,
            verbose=0,
            callbacks=[early],
            epochs=2000
        )
        
        cv_train[te_idx] += np.squeeze(model.predict(df_train_selected.loc[te_idx, :].values.astype(float)))
        print("mse: {}".format(get_mse( cv_train[te_idx] / (s + 1) , df_train_y[te_idx]) ))

        cv_pred += np.squeeze(model.predict(df_test_selected.values.astype(float)))

    print("-----------------")
    print("seed{}_mse: {}".format(s, get_mse(cv_train / (s + 1), df_train_y)))
    print("-----------------")
# -

cv_pred = cv_pred / (NFOLDS * num_seeds)
cv_pred = (cv_pred * (body_max - body_min)) + body_min
df_cv_pred = pd.DataFrame(cv_pred, columns=["Body Mass (g)"])
df_cv_pred.index.name = "id"
df_cv_pred


# ## Lasso

def lasso_model():
    
    model = keras.Sequential([
        layers.Dense(512, input_shape=[15], kernel_initializer=keras.initializers.he_normal(), kernel_regularizer='l1'),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.75),

        layers.Dense(128, kernel_initializer=keras.initializers.he_normal(), kernel_regularizer='l1'),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='sigmoid')

    ])
    
    model.compile(optimizer='adam', loss="mse")
    
    return model


# +
from sklearn.model_selection import StratifiedKFold

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

num_seeds = 5
l1_cv_train = np.zeros(len(df_train_selected))
l1_cv_pred = np.zeros(len(df_test_selected))

early = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

for s in range(num_seeds):
    
    np.random.seed(s)

    for (tr_idx, te_idx) in kfold.split(df_train_selected, df_train_y):
        ds_train = mk_Dataset(df_train_selected.loc[tr_idx, :], df_train_y_rescaled.loc[tr_idx])
        ds_test = mk_Dataset(df_train_selected.loc[te_idx, :], df_train_y_rescaled.loc[te_idx])
        
        model = lasso_model()
        
        model.fit(
            ds_train,
            validation_data=ds_test,
            verbose=0,
            callbacks=[early],
            epochs=2000
        )
        
        l1_cv_train[te_idx] += np.squeeze(model.predict(df_train_selected.loc[te_idx, :].values.astype(float)))
        print("mse: {}".format(get_mse( l1_cv_train[te_idx] / (s + 1) , df_train_y[te_idx]) ))

        l1_cv_pred += np.squeeze(model.predict(df_test_selected.values.astype(float)))

    print("-----------------")
    print("seed{}_mse: {}".format(s, get_mse(l1_cv_train / (s + 1), df_train_y)))
    print("-----------------")
# -

l1_cv_pred = l1_cv_pred / (NFOLDS * num_seeds)
l1_cv_pred = (l1_cv_pred * (body_max - body_min)) + body_min
df_l1_cv_pred = pd.DataFrame(l1_cv_pred, columns=["Body Mass (g)"])
df_l1_cv_pred.index.name = "id"
df_l1_cv_pred


# ## Ridge

def ridge_model():
    
    model = keras.Sequential([
        layers.Dense(512, input_shape=[15], kernel_initializer=keras.initializers.he_normal(), kernel_regularizer='l2'),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.75),

        layers.Dense(128, kernel_initializer=keras.initializers.he_normal(), kernel_regularizer='l2'),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='sigmoid')

    ])
    
    model.compile(optimizer='adam', loss="mse")
    
    return model


# +
from sklearn.model_selection import StratifiedKFold

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

num_seeds = 5
l2_cv_train = np.zeros(len(df_train_selected))
l2_cv_pred = np.zeros(len(df_test_selected))

early = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

for s in range(num_seeds):
    
    np.random.seed(s)

    for (tr_idx, te_idx) in kfold.split(df_train_selected, df_train_y):
        ds_train = mk_Dataset(df_train_selected.loc[tr_idx, :], df_train_y_rescaled.loc[tr_idx])
        ds_test = mk_Dataset(df_train_selected.loc[te_idx, :], df_train_y_rescaled.loc[te_idx])
        
        model = ridge_model()
        
        model.fit(
            ds_train,
            validation_data=ds_test,
            verbose=0,
            callbacks=[early],
            epochs=2000
        )
        
        l2_cv_train[te_idx] += np.squeeze(model.predict(df_train_selected.loc[te_idx, :].values.astype(float)))
        print("mse: {}".format(get_mse( l2_cv_train[te_idx] / (s + 1) , df_train_y[te_idx]) ))

        l2_cv_pred += np.squeeze(model.predict(df_test_selected.values.astype(float)))

    print("-----------------")
    print("seed{}_mse: {}".format(s, get_mse(l2_cv_train / (s + 1), df_train_y)))
    print("-----------------")
# -

l2_cv_pred = l2_cv_pred / (NFOLDS * num_seeds)
l2_cv_pred = (l2_cv_pred * (body_max - body_min)) + body_min
df_l2_cv_pred = pd.DataFrame(l2_cv_pred, columns=["Body Mass (g)"])
df_l2_cv_pred.index.name = "id"
df_l2_cv_pred


# ## Lasso & Ridge

def lr_model():
    
    model = keras.Sequential([
        layers.Dense(512, input_shape=[15], kernel_initializer=keras.initializers.he_normal() 
                     ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.75),

        layers.Dense(128, kernel_initializer=keras.initializers.he_normal()
                     ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
        layers.PReLU(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(1, kernel_initializer=keras.initializers.he_normal(), activation='sigmoid')

    ])
    
    model.compile(optimizer='adam', loss="mse")
    
    return model


# +
from sklearn.model_selection import StratifiedKFold

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

num_seeds = 5
lr_cv_train = np.zeros(len(df_train_selected))
lr_cv_pred = np.zeros(len(df_test_selected))

early = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

for s in range(num_seeds):
    
    np.random.seed(s)

    for (tr_idx, te_idx) in kfold.split(df_train_selected, df_train_y):
        ds_train = mk_Dataset(df_train_selected.loc[tr_idx, :], df_train_y_rescaled.loc[tr_idx])
        ds_test = mk_Dataset(df_train_selected.loc[te_idx, :], df_train_y_rescaled.loc[te_idx])
        
        model = lr_model()
        
        model.fit(
            ds_train,
            validation_data=ds_test,
            verbose=0,
            callbacks=[early],
            epochs=2000
        )
        
        lr_cv_train[te_idx] += np.squeeze(model.predict(df_train_selected.loc[te_idx, :].values.astype(float)))
        print("mse: {}".format(get_mse( lr_cv_train[te_idx] / (s + 1) , df_train_y[te_idx]) ))

        lr_cv_pred += np.squeeze(model.predict(df_test_selected.values.astype(float)))

    print("-----------------")
    print("seed{}_mse: {}".format(s, get_mse(lr_cv_train / (s + 1), df_train_y)))
    print("-----------------")
# -

lr_cv_pred = lr_cv_pred / (NFOLDS * num_seeds)
lr_cv_pred = (lr_cv_pred * (body_max - body_min)) + body_min
df_lr_cv_pred = pd.DataFrame(lr_cv_pred, columns=["Body Mass (g)"])
df_lr_cv_pred.index.name = "id"
df_lr_cv_pred

# ## lgb

# +
from keras.wrappers.scikit_learn import KerasRegressor

model = lgb.LGBMRegressor()

X = df_train_target
y = df_train_y

selector = SequentialFeatureSelector(
    model, 
    scoring='neg_mean_squared_error', 
    k_features='best', 
    forward=True, 
    floating=True,
    cv=10
)

selector = selector.fit(X, y)
fig = plot_sfs(selector.get_metric_dict(), kind='std_err')

# -

selector.k_feature_names_

df_train_selected_lgb = df_train_target.loc[:, selector.k_feature_names_]
df_train_selected_lgb

# +
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = 102)

param = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 31, 'random_state': 30,
         'objective': 'mse', 'device_type': 'gpu', 'verbosity' : -1, 'feature_fraction': 0.8}

num_seeds = 16
n_round = 10000

cv_train_lgb = np.zeros(len(df_train_selected))
cv_pred_lgb = np.zeros(len(df_test_selected))

early = lgb.early_stopping(100)

for s in range(num_seeds):
    
    param['random_state'] = s

    for (tr_idx, te_idx) in kfold.split(df_train_selected, df_train_y):
        
        train_data_lgb = lgb.Dataset(df_train_selected.loc[tr_idx, :].values, label=df_train_y.loc[tr_idx].values)
        val_data_lgb = train_data_lgb.create_valid(df_train_selected.loc[te_idx, :].values, label=df_train_y.loc[te_idx].values)
        
        bst = lgb.train(param, train_data_lgb, n_round, valid_sets=[val_data_lgb], callbacks=[early])
        print('best score: {}'.format(bst.best_score))

        cv_train_lgb[te_idx] += np.squeeze(bst.predict(df_train_selected.loc[te_idx, :].values.astype(float)))
        print("mse: {}".format(get_mse( cv_train_lgb[te_idx] / (s + 1) , df_train_y.loc[te_idx]) ))
        
        cv_pred_lgb += np.squeeze(bst.predict(df_test_selected.values.astype(float)))

    print("-----------------")
    print("seed{}_mse: {}".format(s, get_mse( cv_train_lgb / (s + 1), df_train_y)))
    print("-----------------")
# -

cv_pred_lgb = cv_pred_lgb *(NFOLDS * num_seeds) *(NFOLDS * num_seeds)

cv_pred_lgb = cv_pred_lgb / (NFOLDS * num_seeds)
df_cv_pred_lgb = pd.DataFrame(cv_pred_lgb, columns=["Body Mass (g)"])
df_cv_pred_lgb.index.name = "id"
df_cv_pred_lgb

cv_pred_mix = cv_pred * 0.2 + l1_cv_pred * 0.2 + l2_cv_pred * 0.2 + lr_cv_pred * 0.2 + cv_pred_lgb * 0.2
cv_pred_mix = cv_pred_mix * 0.5 + pred_en * 0.5
df_cv_mix = pd.DataFrame(cv_pred_mix, columns=["Body Mass (g)"])
df_cv_mix.index.name = "id"
df_cv_mix

df_cv_mix.to_csv("submission_cv_mix_final.csv")
