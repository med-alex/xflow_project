import pandas as pd

df = pd.read_csv('/home/ml-srv-admin/xflow_project/datasets/data.csv')

for i in range(len(df.id)):
  max = df.counts.max()
  min =df.counts.min()
  df.counts[i] = (df.counts[i] - min)/(max - min)

df.to_csv('/home/ml-srv-admin/xflow_project/datasets/data_processed.csv', index=False)
