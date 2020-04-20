from os import listdir
import fileinput
import sys

def trim_csv(file):
    for line in fileinput.input(file, inplace=True):
        line = line.replace('""', '\\"')
        sys.stdout.write(line)

directory = './RawData/iot-network-intrusion-dataset/Packets_csv'
files = listdir(directory)

for file in files:
    trim_csv(directory + '/' + file)
