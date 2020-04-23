import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./preprocessed.csv', dtype=np.float)

to_one_hot = ['srcIP', 'dstIP', 'srcPort', 'dstPort', 'protocol', 'flags', 'method', 'status', 'host',
              'user-agent']

unique_sum = 0
for col in df:
    if col in to_one_hot:
        print(col, ',', df[col].max())
        unique_sum += df[col].max() + 1
    else:
        unique_sum += 1

to_df = pd.DataFrame(np.zeros((df.values.shape[0], int(unique_sum))), dtype=np.float)

unique_sum = 0

for col in df:
    print(col)
    if col == 'class':
        to_df.iloc[:, -1] = df[col]
    if col in to_one_hot:
        ser = df[col]
        maxval = int(ser.max())
        tmpnp = np.zeros((df.values.shape[0], maxval + 1))
        for i in range(maxval + 1):
            tmpnp[np.where(ser == i)[0], :] = np.eye(maxval + 1)[i]
        to_df.iloc[:, unique_sum:unique_sum + maxval + 1] = tmpnp
        unique_sum += maxval + 1
    else:
        to_df.iloc[:, unique_sum] = MinMaxScaler().fit_transform(df[col].to_numpy().reshape((-1, 1)))
        unique_sum += 1

to_df.to_csv('finally_preprocessed.csv', index=False, header=False)
to_df = to_df.iloc[:2000, :]

to_df.to_csv('cracked.csv', index=False, header=False)
