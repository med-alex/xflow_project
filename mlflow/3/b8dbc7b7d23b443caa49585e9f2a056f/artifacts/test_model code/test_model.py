from sklearn.metrics import mean_squared_error
import pickle
import pandas as pd
from catboost import Pool
import os
import mlflow
from mlflow.tracking import MlflowClient


os.environ["MLFLOW_REGISTRY_URI"] = "/home/ml-srv-admin/xflow_project/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test_model")

df_test = pd.read_csv('/home/ml-srv-admin/xflow_project/datasets/data_test.csv')

test_pool = Pool(data=df_test.id, cat_features=['id'], feature_names=['id'])

with open('/home/ml-srv-admin/xflow_project/models/model.pkl', 'rb') as f:
    model = pickle.load(f)

pred = model.predict(test_pool)
mse = mean_squared_error(df_test.norm.values, pred)

with mlflow.start_run():
    mlflow.log_metric("mse", mse)
    mlflow.log_artifact(local_path="/home/ml-srv-admin/xflow_project/scripts/test_model.py",
                        artifact_path="test_model code")
    mlflow.end_run()

print("mse=", mse)
