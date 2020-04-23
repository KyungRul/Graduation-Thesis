import pandas as pd
import numpy as np


def preprocess(colname, df, dtype, to_df):
    print(colname, ' processing')
    ser = df[colname].astype(dtype)
    unique = np.unique(ser)
    new_np = np.zeros_like(ser, dtype=np.float)
    for i, u in enumerate(unique):
        new_np[np.where(ser == u)[0]] = i
    to_df[colname] = new_np
    return to_df


def categorize(df):
    dtypes = {'srcIP': np.str, 'dstIP': np.str, 'srcPort': np.str, 'dstPort': np.str, 'protocol': np.str,
              'seq': np.float,
              'ack': np.float, 'flags': np.str, 'method': np.str, 'uriLen': np.float, 'status': np.str, 'host': np.str,
              'user-agent': np.str, 'cookiesLen': np.float, 'len': np.float, 'class': np.float}
    to_df = pd.DataFrame(np.zeros_like(df.values), columns=dtypes.keys(), dtype=np.float)

    for k, v in dtypes.items():
        if v == np.str:
            to_df = preprocess(k, df, v, to_df)
        else:
            to_df[k] = df[k]
    return to_df


if __name__ == '__main__':
    file_path = './packets_extracted_info/'
    df_http = pd.read_csv(file_path + 'httpflooding_extract.csv')
    df_ack = pd.read_csv(file_path + 'ackflooding_extract.csv')
    df_udp = pd.read_csv(file_path + 'udpflooding_extract.csv')

    df = df_http
    df = df.append(df_ack, ignore_index=True)
    df = df.append(df_udp, ignore_index=True)

    df.fillna(value={'srcPort': 1, 'dstPort': 1, 'seq': 0, 'ack': -1, 'uriLen': 0, 'cookiesLen': 0}, inplace=True)

    src = df['srcIP'].to_numpy().astype(np.str)
    dst = df['dstIP'].to_numpy().astype(np.str)
    ip = np.unique(np.concatenate((src, dst)))
    freq = []
    freq_dst = []
    freq_src = []
    src_unique = np.unique(src)
    dst_unique = np.unique(dst)
    for i in src_unique:
        freq_src.append(np.where(src == i)[0].size)
    for i in dst_unique:
        freq_dst.append(np.where(dst == i)[0].size)

    freq_dst = np.array(freq_dst)
    freq_src = np.array(freq_src)

    less_ips = dst_unique[np.where(freq_dst <= 3000)]
    for i in less_ips:
        dst[np.where(dst == i)[0]] = 'small'

    less_ips = src_unique[np.where(freq_src <= 6)]
    for i in less_ips:
        src[np.where(src == i)[0]] = 'small'
    df['srcIP'] = src
    df['dstIP'] = dst

    ip = ip.size

    src = df['srcPort'].to_numpy().astype(np.int)
    dst = df['dstPort'].to_numpy().astype(np.int)

    src_unique = np.unique(src)
    dst_unique = np.unique(dst)

    count = np.bincount(src)
    # idx = count.nonzero()[0]
    # selected_ports = np.where(count.sum() // 100 >= count)
    # selected_ports = np.intersect1d(idx, selected_ports[0])
    selected_ports = np.where(count.sum() // 100 < count)[0]
    selected_ports = [a for a in src_unique if a not in selected_ports]
    for port in selected_ports:
        src[np.where(src == port)[0]] = -1

    count = np.bincount(dst)
    # idx = count.nonzero()[0]
    # selected_ports = np.where(count.sum() // 100 >= count)
    # selected_ports = np.intersect1d(idx, selected_ports)
    selected_ports = np.where(count.sum() // 100 < count)[0]
    selected_ports = [a for a in src_unique if a not in selected_ports]
    for port in selected_ports:
        dst[np.where(dst == port)[0]] = -1

    df['srcPort'] = src
    df['dstPort'] = dst
    #
    # flag = np.unique(df['flags'].to_numpy().astype(np.str)).size
    # method = np.unique(df['method'].to_numpy().astype(np.str)).size
    # status = np.unique(df['status'].to_numpy().astype(np.str)).size
    # host = np.unique(df['host'].to_numpy().astype(np.str)).size
    # user_agent = np.unique(df['user-agent'].to_numpy().astype(np.str)).size

    to_df = categorize(df)
    to_df.to_csv('./preprocessed.csv', index=False)
