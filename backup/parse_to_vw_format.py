import collections;
import cPickle
import os;
import operator;
import math;
import scipy;
import scipy.linalg;
import sklearn
import sklearn.covariance
import sys
import time;
import theano
import theano.tensor
import types
import numpy
import numpy.random;

def parse_to_vw_format(data_directory, data_file):
    data_feature = numpy.load(os.path.join(data_directory, "%s.feature.npy" % data_file))
    data_label = numpy.load(os.path.join(data_directory, "%s.label.npy" % data_file))
    output_file = os.path.join(data_directory, "%s.txt" % data_file);
    output_stream = open(output_file, 'w');
    
    for x in xrange(len(data_label)):
        label_string = "%s" % data_label[x];
        feature_list = [];
        for y in xrange(len(data_feature[x, :])):
            if data_feature[x, y] == 0:
                continue;
            feature_list.append("%d:%g" % (y, data_feature[x, y]));
        feature_string = " ".join(feature_list)
        output_stream.write("%s |%s %s\n" % (label_string, "U1", feature_string));
        
if __name__ == '__main__':
    data_directory = sys.argv[1];
    parse_to_vw_format(data_directory, "train")
    parse_to_vw_format(data_directory, "test")
