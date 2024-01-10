from sklearn.metrics import mean_squared_error
import pickle
import pandas as pd
from catboost import CatBoostRegressor, Pool

df_test = pd.read_csv('/home/ml-srv-admin/xflow_project/datasets/data_test.csv')

test_pool = Pool(data=df_test.id, cat_features=['id'], feature_names=['id'])
model = CatBoostRegressor()
with open('/home/ml-srv-admin/xflow_project/models/model.pkl', 'rb') as f:
    model = pickle.load(f)

pred = model.predict(test_pool)
mse = mean_squared_error(df_test.norm.values, pred)
print("mse=", mse)
