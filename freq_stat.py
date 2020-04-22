from os import listdir
import numpy as np
from scapy.all import *
import gc
import pandas as pd
from scipy import stats

tot_proto_result = {}
tot_len_result = []
tot_attack_len_result = []

cols = ['normal_len_mean', 'normal_len_std', 'abnormal_len_mean', 'abnormal_len_std', 'tot_len_mean', 'tot_len_std',
        'normal_count', 'abnormal_count', 'tot_count']
df_stat = pd.DataFrame(np.zeros((38, len(cols))), columns=cols, dtype=np.float)

i = -1


def insert_df(lst_len, lst_attack_len):
    lst_len = np.array(lst_len)
    lst_attack_len = np.array(lst_attack_len)
    df_stat['normal_len_mean'][i] = lst_len.mean()
    df_stat['normal_len_std'][i] = lst_len.std()
    df_stat['normal_count'][i] = lst_len.size
    df_stat['abnormal_len_mean'][i] = lst_attack_len.mean()
    df_stat['abnormal_len_std'][i] = lst_attack_len.std()
    df_stat['abnormal_count'][i] = lst_attack_len.size

    lst_tot = np.array(lst_len.tolist() + lst_attack_len.tolist())
    df_stat['tot_len_mean'][i] = lst_tot.mean()
    df_stat['tot_len_std'][i] = lst_tot.std()
    df_stat['tot_count'][i] = lst_tot.size


def print_percentage(pcap, file):
    global tot_len_result
    global i
    global tot_attack_len_result
    i += 1
    dict_proto = {}
    lst_len = []
    lst_attack_len = []
    for packet in pcap:
        if packet.payload.name == 'ARP' or packet.payload.name == 'Raw':
            continue

        name = packet.payload.payload.name
        dict_proto[name] = dict_proto.get(name, 0) + 1
        tot_proto_result[name] = tot_proto_result.get(name, 0) + 1
        if packet.payload.dst == '210.89.164.90':
            lst_attack_len.append(packet.__len__())
        else:
            lst_len.append(packet.__len__())

    tot_len_result += lst_len
    tot_attack_len_result += lst_attack_len

    insert_df(lst_len, lst_attack_len)

    result = ''
    print(file, ' Total: ', len(pcap))
    for k, v in sorted(dict_proto.items(), key=lambda x: x[1]):
        result += 'num: {}, {}: {}\t'.format(v, k, v / len(pcap))
    print(result)
    gc.collect()


if __name__ == "__main__":
    dir_path = './RawData/iot-network-intrusion-dataset/Packets'
    files = listdir(dir_path)

    for file in files:
        if 'mirai' in file and 'hostbruteforce' not in file:
            pcap = rdpcap(dir_path + '/' + file)
            print_percentage(pcap, file)

    result = ''
    for k, v in sorted(tot_proto_result.items(), key=lambda x: x[1]):
        result += 'num: {}, {}: {}\t'.format(v, k, v / len(pcap))
    print(result)

    i += 1
    insert_df(tot_len_result, tot_attack_len_result)
    df_stat.to_csv(dir_path + '/' + 'len_stat.csv')
