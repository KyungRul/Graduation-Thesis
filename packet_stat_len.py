from os import listdir
import numpy as np
from scapy.all import *
import gc
import pandas as pd
from scipy import stats

dir_path = './packets_extracted_info'
target_dir_path = './stat'


def insert_into_df(normal, attack, type):
    dct = {}
    dct['Type'] = type
    dct['normal_len_mean'] = normal.mean()
    dct['normal_len_std'] = normal.std()
    dct['normal_count'] = normal.size
    dct['attack_len_mean'] = attack.mean()
    dct['attack_len_std'] = attack.std()
    dct['normal_count'] = attack.size

    tot = np.concatenate((normal, attack))
    dct['tot_len_mean'] = tot.mean()
    dct['tot_len_std'] = tot.std()
    dct['tot_count'] = tot.size

    return dct


def analyze_file(file, df_stat):
    df_src = pd.read_csv(dir_path + '/' + file, header=0)

    normal_len = df_src.loc[~df_src['class'], 'len']
    attack_len = df_src.loc[df_src['class'], 'len']

    normal_len = normal_len.to_numpy()
    attack_len = attack_len.to_numpy()

    df_stat = df_stat.append(insert_into_df(normal_len, attack_len, file.split('_')[0]), ignore_index=True)

    return df_stat, normal_len, attack_len


if __name__ == "__main__":
    files = listdir(dir_path)
    dtypes = np.dtype([('normal_len_mean', np.float), ('normal_len_std', np.float), ('attack_len_mean', np.float),
                       ('attack_len_std', np.float), ('tot_len_mean', np.float), ('tot_len_std', np.float),
                       ('normal_count', np.int), ('abnormal_count', np.int), ('tot_count', np.int)])
    df_stat = pd.DataFrame(np.empty((0), dtype=dtypes))

    normal_len_tot = np.empty([1], dtype=np.int)
    attack_len_tot = np.empty([1], dtype=np.int)

    for file in files:
        print('{} starts'.format(file))
        df_stat, normal_len, attack_len = analyze_file(file, df_stat)
        normal_len_tot = np.concatenate((normal_len_tot, normal_len))
        attack_len_tot = np.concatenate((attack_len_tot, attack_len))
        print('{} finished'.format(file))

    df_stat = df_stat.append(insert_into_df(normal_len_tot, attack_len_tot, 'total'), ignore_index=True)
    df_stat.to_csv(target_dir_path + '/len_stat.csv', index=False)
