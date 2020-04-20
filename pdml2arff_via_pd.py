#!/usr/bin/python
# change the path (above) to reflect where you have python installed
#
# this script will take a tshark generated pdml file and turn it
# into an arff formatted file, suitable for ingestment by weka
# here's how to create the pdml file from pcap:
# tshark -T pdml -r <infile> > <outfile>
# (adding -V gets you no more data)
# usage of this script: pdml2arff.py <outfile> (outfile is pdml from above)
# ./pdml2arff.py <input_file> -o <output_file(optional)> -n (convert all strings to numerics

import csv
from os import listdir, stat
import pandas as pd
import arff as af


class myDialect(csv.Dialect):
    delimiter = ','
    quotechar = '"'
    quoting = csv.QUOTE_NONNUMERIC
    lineterminator = "\n"
    doublequote = False
    skipinitialspace = False


#
# Define a simple class to wrap functions
#
class PdmlConvert:
    def __init__(self, templateString, numbers_only=False):
        self.template = templateString
        self.numbers_only = numbers_only
        self.headers = []
        self.results = []
        self.packet_count = 1
        self.df = None

    #
    # convert the given input to ARFF format
    #
    def convert_file(self, input_file, **kwargs):
        fname, ext = self.parse_filename(input_file)
        output_file = kwargs.get('output_file', fname + '.arff').replace('pdml', 'arff')
        self.parse_file(input_file)
        header = self.build_header(input_file)  # build the top section of output file
        # self.write_to_file(header, output_file)  # write top section to output file
        self.append_array_of_dict_to_csv(output_file)  # write data to output file

    #
    #  uses xml.dom.minidom to parse input xml file
    #  - reads each packet -> proto -> field
    #  - creates a key/value results dict {} for each field
    #  - new fields are added to headers array
    #
    def parse_file(self, file):
        from xml.dom import minidom  # load minidom
        # self.clean_file(file)  # found a parsing error in input data, see clean_file for info
        xmldoc = minidom.parse(file)  # use minidom to parse xml
        for packet in xmldoc.getElementsByTagName('packet'):  # for every packet -> proto -> field...
            self.parse_packet(packet)
            if 200 < len(self.results):
                return

    #
    #
    #
    def parse_packet(self, packet):
        id = self.packet_count
        self.packet_count += 1
        arf = self.create_arf(id)
        for proto in packet.getElementsByTagName('proto'):
            arf = self.parse_proto_into_arf(arf, proto)
        if arf.get('ip.dst', '') == 'd259a45a':
            arf['class'] = 1
        else:
            arf['class'] = 0
        self.results.append(arf)

    #
    #
    #
    def parse_proto_into_arf(self, arf, proto):
        proto_name = proto.getAttribute('name')
        for field in proto.getElementsByTagName('field'):
            arf = self.parse_field_into_arf(proto_name, arf, field)
        return arf

    #
    # parse_field_into_arf ( proto_name , arf , field )
    #                      Adds any field or subfields to arf {} if it has a value
    #
    def parse_field_into_arf(self, proto_name, arf, field):
        field_name = field.getAttribute('name')  # get name attribute of field
        name = self.build_name(field_name, proto_name)  # build name grand.parent.child
        arf = self.append_key_value(name, self.get_value_from_field(field), arf)  # append key/val to arf dict {}

        # Some fields have children subfields with values
        for subfield in field.getElementsByTagName('field'):
            sf_name = subfield.getAttribute('name')
            name = self.build_name(sf_name, field.getAttribute('name'), proto_name)
            arf = self.append_key_value(name, self.get_value_from_field(subfield), arf)
        return arf

    #
    #
    #
    def append_key_value(self, key, value, map):
        if value == '':
            return map
        if not key in self.headers:
            self.headers.append(key)
        map[key] = value
        return map

    #
    # Returns an unmaskedvalue or a vlue or '' from field attributes
    #
    def get_value_from_field(self, field):
        if field.hasAttribute('unmaskedvalue'):
            return field.getAttribute('unmaskedvalue')
        elif field.hasAttribute('value'):
            return field.getAttribute('value')
        else:
            return ''

    #
    #
    #
    def build_name(self, name, parent, grand=''):
        ret = name
        if not str(name).startswith(parent):
            ret = parent + '.' + ret
        if not grand == '':
            if not ret.startswith(grand):
                ret = grand + '.' + ret
        return ret

    #
    #
    #
    def create_arf(self, id):
        if not 'packet_id' in self.headers:
            self.headers.append('packet_id')
        return {'packet_id': id}

    #
    # This clean file is a simple xml cleaner of the <proto> </proto> element
    # In the input files I've seen, there is an extra </proto> which shows up
    # just before a '</packet>' in the data (often but not always).  So this function
    # counts each opening '<proto' and closing '</proto>' and whenever we see an extra
    # (count < 0) we do not output that extra one.  This seems to clean the file properly.
    #
    def clean_file(self, file):
        import re
        stack = 0
        output = []
        for line in open(file, 'r'):
            if re.search('<proto', line):
                stack += 1
            elif re.search('</proto>', line):
                stack -= 1

            if stack >= 0:
                output.append(line)
            else:
                stack += 1

        o = open(file, 'wb')
        for line in output:
            o.write(line.encode('UTF-8', 'replace'))

    #
    # Appends and Array of Dictionaries to given filename
    # - inserts headers at beginning (of where appending happens)
    #
    def append_array_of_dict_to_csv(self, filename):
        # csvfile = open(filename, 'a')  # open file for appending
        dialect = myDialect()

        self.df = pd.DataFrame(columns=self.headers)

        # csvw = csv.DictWriter(csvfile, self.headers, '?', dialect=dialect)  # instantiate DictWriter
        # for kvs in self.results:  # for every dict result, append dict to csv
        #     if self.numbers_only:
        #         kvs = self.map2num(kvs)
        #     # csvw.writerow(kvs)
        #     self.df = self.df.append(kvs, ignore_index=True)

        self.df = pd.DataFrame(self.results)
        self.df = self.df.iloc[:, [0, 300]]
        self.df.dropna(axis=1, how='all')
        self.df.fillna("?")

        af.dump(filename, self.df.values, relation=filename, names=self.df.columns)

    #
    # Writes text to filename
    #
    def write_to_filei(self, text, filename):
        f = open(filename, 'wb')
        f.write(text.encode('UTF-8', 'replace'))

    #
    # Build header/top section of output file
    #
    def build_header(self, filename):
        from string import Template
        text = Template(self.template)  # Template example:
        attr_str = ""  # temp = Template('this is a $INSERT')

        for attr in self.headers:  # print temp.substitute(INSERT='test')
            #			attr_str += "@attribute " + attr + " STRING" + "\n" # use this if outputting "string" data type
            attr_str += "@attribute " + attr + " STRING" + "\n"  # use this if outputting "numeric" data type
        return text.substitute(RELATION=filename, ATTRIBUTES=attr_str)

    #
    # Parse a filename into its base name and extension
    # returns [basename,ext] or 'Invalid Filename'
    #
    def parse_filename(self, name):
        import re
        r = re.search(r"(\S+)(\.\S{1,4})$", name)
        if r:
            return [r.group(1), r.group(2)]
        else:
            raise Exception('Invalid Filename')

    #
    #  converts each value of the given map/dict to an integer using str2num
    #
    def map2num(self, m):
        result = {}
        for k, v in m.iteritems():
            result[k] = self.str2num(v)
        return result

    #
    # Convert a string to a number (takes the ord value of each letter and
    # combines it then converts it to int)
    # i.e. str2num( 'abc' ); ord('a') = 97; "979899" => returns 979899 as int
    #
    def str2num(self, s):
        if type(s) is int:
            return s
        num = ''
        for letter in s:
            o = ord(letter)
            num += str(o)
        return int(num)

    #
    #  Write errors to log
    #
    def error_log(self, message):
        f = open('pdml.errors.log', 'wb')
        f.write(message)


# Template ARFF File
arff = '''
%
% This arff created by pdml2arff.py
% Written by Tim Stello with input from Charlie Fowler, spring 2013
% This script takes a pdml file created by tshark and converts it to arff
%
@relation $RELATION
%
%attributes
%
$ATTRIBUTES
%
@data
%
'''

#
# Main: this portion executes only when this file is executed
# from the command line.  If you 'import' this file, this section
# will not execute
#
if __name__ == '__main__':
    import sys

    usage = "./pdml2arffpy <input_file> -o <output_file (optional)> -n (convert all strings to numerics)\n"
    # numbers_only = False
    # if '-n' in sys.argv:
    #     numbers_only = True
    #     sys.argv.remove('-n')
    # pdmlc = PdmlConvert(arff, numbers_only)
    # l = len(sys.argv)
    # if l == 2:
    #     pdmlc.convert_file(sys.argv[1])
    # elif l == 4:
    #     pdmlc.convert_file(sys.argv[1], {'output_file': sys.argv[3]})
    # else:
    #     print(usage)
    #     # sys.exit

    numbers_only = False
    pdmlc = PdmlConvert(arff, numbers_only)
    files = listdir('./RawData/iot-network-intrusion-dataset/Packets_pdml')

    # for file in files:
    #     print('{} cleaning starts'.format(file))
    #     pdmlc.clean_file('./RawData/iot-network-intrusion-dataset/Packets_pdml/' + file)
    #     print('{} cleaning finished'.format(file))

    done_files = listdir("./RawData/iot-network-intrusion-dataset/Packets_arff")

    done_files = [file.replace('arff', 'pdml') for file in done_files]
    for file in files:
        if file not in done_files and 'mirai' in file:
            print('{} starts'.format(file))
            pdmlc.convert_file('./RawData/iot-network-intrusion-dataset/Packets_pdml/' + file)
            print('{} finished'.format(file))
