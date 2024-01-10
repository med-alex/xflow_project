import pickle
import pandas as pd
from catboost import CatBoostRegressor, Pool

df_train = pd.read_csv('/home/ml-srv-admin/xflow_project/datasets/data_train.csv')

train_pool = Pool(data=df_train.id, label=df_train.norm, cat_features=['id'], feature_names=['id'])
model = CatBoostRegressor(max_depth=2, random_state=0)
model.fit(train_pool)

with open('/home/ml-srv-admin/xflow_project/models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
