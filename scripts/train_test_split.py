import pandas as pd


df = pd.read_csv('/home/ml-srv-admin/xflow_project/datasets/data_processed.csv')

l = int(len(df)*0.7)
train = df[:l]
test = df[l+1:]

test.columns = ['id', 'norm']
train.to_csv('/home/ml-srv-admin/xflow_project/datasets/data_train.csv', index=None)
test.to_csv('/home/ml-srv-admin/xflow_project/datasets/data_test.csv', index=None)
