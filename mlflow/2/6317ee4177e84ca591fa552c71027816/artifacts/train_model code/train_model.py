import pickle
import pandas as pd
from catboost import CatBoostRegressor, Pool
import os
import mlflow
from mlflow.tracking import MlflowClient


os.environ["MLFLOW_REGISTRY_URI"] = "/home/ml-srv-admin/xflow_project/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")

df_train = pd.read_csv('/home/ml-srv-admin/xflow_project/datasets/data_train.csv')

train_pool = Pool(data=df_train.id, label=df_train.norm, cat_features=['id'], feature_names=['id'])

max_depth_p = 2
random_state_p = 0
model = CatBoostRegressor(max_depth=max_depth_p, random_state=random_state_p)
model.fit(train_pool)

with mlflow.start_run():
    mlflow.catboost.log_model(model,
                             artifact_path="catboost",
                             registered_model_name="catboost")
    mlflow.log_param("max_depth", max_depth_p)
    mlflow.log_param("random_state", random_state_p)
    mlflow.log_artifact(local_path="/home/ml-srv-admin/xflow_project/scripts/train_model.py",
                        artifact_path="train_model code")
    mlflow.end_run()

with open('/home/ml-srv-admin/xflow_project/models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
