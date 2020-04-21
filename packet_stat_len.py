from os import listdir
import numpy as np
import gc
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

dir_path = './packets_extracted_info'
target_dir_path = './stat'


def get_rel_freq(lst_len, title):
    res = stats.relfreq(lst_len, numbins=10)
    x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, res.frequency, width=res.binsize)
    ax.set_title(title)
    ax.set_xlim([x.min(), x.max()])
    plt.savefig(target_dir_path + '/' + title + '.jpg')


def insert_into_df(normal, attack, type):
    dct = {}
    dct['Type'] = type
    dct['normal_len_mean'] = normal.mean()
    dct['normal_len_std'] = normal.std()
    dct['normal_count'] = normal.size
    dct['attack_len_mean'] = attack.mean()
    dct['attack_len_std'] = attack.std()
    dct['abnormal_count'] = attack.size

    tot = np.concatenate((normal, attack))
    dct['tot_len_mean'] = tot.mean()
    dct['tot_len_std'] = tot.std()
    dct['tot_count'] = tot.size

    get_rel_freq(normal, type + '_' + 'normal_rel_freq')
    get_rel_freq(attack, type + '_' + 'attack_rel_freq')
    get_rel_freq(tot, type + '_' + 'total_rel_freq')

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

    normal_len_tot = np.zeros((0), dtype=np.int)
    attack_len_tot = np.zeros((0), dtype=np.int)

    for file in files:
        print('{} starts'.format(file))
        df_stat, normal_len, attack_len = analyze_file(file, df_stat)
        normal_len_tot = np.concatenate((normal_len_tot, normal_len))
        attack_len_tot = np.concatenate((attack_len_tot, attack_len))
        print('{} finished'.format(file))

    df_stat = df_stat.append(insert_into_df(normal_len_tot, attack_len_tot, 'total'), ignore_index=True)
    df_stat.to_csv(target_dir_path + '/len_stat.csv', index=False)
