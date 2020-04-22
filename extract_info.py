from scapy.all import *
from os import listdir
import pandas as pd
import numpy as np
import gc


def pcap2dict(target, label):
    p_list = []
    pcap = rdpcap(target)

    for i, packet in enumerate(pcap):
        layer = packet.payload
        p_dict = {}
        p_dict['class'] = 0
        while layer:
            layerName = layer.name

            if layerName == "IP":
                p_dict["srcIP"] = layer.src
                p_dict["dstIP"] = layer.dst
                if layer.dst == '210.89.164.90':
                    p_dict['class'] = label
            if layerName == "TCP" or layerName == "UDP":
                p_dict['protocol'] = layerName
                if layerName == 'TCP':
                    flags = ''
                    if layer.flags == 2:
                        flags = "SYN"
                    elif layer.flags == 16:
                        flags = "ACK"
                    elif layer.flags == 17:
                        flags = "FIN,ACK"
                    elif layer.flags == 18:
                        flags = "SYN,ACK"
                    elif layer.flags == 24:
                        flags = "PSH,ACK"
                    p_dict["flags"] = flags
                    p_dict["ack"] = int(layer.ack)
                    p_dict["seq"] = int(layer.seq)

                p_dict["srcPort"] = layer.sport
                p_dict["dstPort"] = layer.dport
            if layerName == "Raw":
                result = processHTTP(layer.load)
                for k, v in result.items():
                    p_dict[k] = v

            layer = layer.payload
        p_dict['len'] = packet.__len__()
        p_list.append(p_dict)
        if i % 500 == 0:
            print('{}/{} '.format(i, len(pcap)))
    return p_list


def processHTTP(data):
    info = dict()

    try:
        data.decode()
    except UnicodeDecodeError:
        return info

    headers = data.decode().splitlines()
    info['method'] = 'ETC'
    for header in headers:
        if header.startswith("GET") or header.startswith("POST"):
            info["method"] = header.split()[0]
            info["uriLen"] = len(header.split()[1])
        if header.startswith("HTTP"):
            info["method"] = "response"
            info["status"] = header.split()[1]
        if header.startswith("HOST"):
            info["host"] = header.split(":", 1)[1]
        if header.startswith("User-Agent"):
            info["user-agent"] = header.split(":", 1)[1]
        if header.startswith("Cookie"):
            info["cookiesLen"] = len(header.split(":", 1)[1])

    return info


def analyze_file(files, to_file, label):
    result_path = './packets_extracted_info/'
    dtypes = np.dtype(
        [('srcIP', np.str), ('dstIP', np.str), ('srcPort', np.str), ('dstPort', np.str), ('protocol', np.str),
         ('seq', np.int), ('ack', np.int), ('flags', np.str), ('method', np.str), ('uriLen', np.int),
         ('status', np.str),
         ('host', np.str), ('user-agent', np.str), ('cookiesLen', np.int), ('len', np.int),
         ('class', np.int)])

    df = pd.DataFrame(np.empty(0, dtype=dtypes))
    for file in files:
        print(file, ': starts')
        result = pcap2dict(file, label)
        df = df.append(result, ignore_index=True)
        print(file, ': finished')
        del result
        gc.collect()

    df.to_csv(result_path + to_file, index=False)

    del df
    gc.collect()


if __name__ == "__main__":
    dir_path = './RawData/iot-network-intrusion-dataset/Packets'
    files = listdir(dir_path)
    udpflood = [dir_path + '/' + file for file in files if "udpflooding" in file]
    ackflood = [dir_path + '/' + file for file in files if "ackflooding" in file]
    httpflood = [dir_path + '/' + file for file in files if "httpflooding" in file]
    analyze_file(httpflood, 'httpflooding_extract.csv', 2)
    analyze_file(udpflood, 'udpflooding_extract.csv', 3)
    analyze_file(ackflood, 'ackflooding_extract.csv', 1)
