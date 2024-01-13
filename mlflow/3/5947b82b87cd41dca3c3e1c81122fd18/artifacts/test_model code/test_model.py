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

client = MlflowClient()
exp = client.get_experiment_by_name('train_model')
runs_info = client.search_runs(exp.experiment_id)

model = mlflow.catboost.load_model(f"runs:/{runs_info[0].info.run_id}/catboost")
model_params = model.get_params()
pred = model.predict(test_pool)
mse = mean_squared_error(df_test.norm.values, pred)

with mlflow.start_run():
    mlflow.log_metric("mse", mse)
    for param in model_params.keys():
        mlflow.log_param(param, model_params[param])
    mlflow.log_artifact(local_path="/home/ml-srv-admin/xflow_project/scripts/test_model.py",
                        artifact_path="test_model code")
    mlflow.end_run()

print("mse=", mse)
