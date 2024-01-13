import pandas as pd

df = pd.read_csv('/home/ml-srv-admin/xflow_project/datasets/data.csv')

for i in df.index:
  df.at[i, 'norm'] = (df.counts[i] - df.counts.min())/(df.counts.max() - df.counts.min())

# df['id_index'] = df.index

df[['id', 'norm']].to_csv('/home/ml-srv-admin/xflow_project/datasets/data_processed.csv', index=False)
