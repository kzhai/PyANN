import collections;
import cPickle
import os;
import operator;
import math;
import numpy;
import numpy.random;
import scipy;
import scipy.linalg;
import sklearn
import sklearn.covariance
import sys
import time;
import theano
import theano.tensor
import matplotlib
import matplotlib.pyplot

def document_parser(input_directory, output_directory, cutoff_df_threshold=0, cutoff_tf_threshold=0):
    input_training_file = os.path.join(input_directory, "train.dat");
    input_testing_file = os.path.join(input_directory, "test.dat");
    
    label_information, type_information = collect_statistics(input_training_file);
    
    (label_to_index, index_to_label, label_frequency) = label_information
    output_label_file = os.path.join(output_directory, "label.info.dat")
    output_label_stream = open(output_label_file, 'w');
    for label_index in xrange(len(index_to_label)):
        output_label_stream.write("%s\t%d\n" % (index_to_label[label_index], label_frequency[index_to_label[label_index]]));
        #print "%s\t%d\n" % (index_to_label[label_index], label_frequency[index_to_label[label_index]])
    output_label_stream.close()
    
    (type_to_index, index_to_type, type_document_frequency, type_term_frequency) = type_information
    output_type_file = os.path.join(output_directory, "type.info.dat")
    output_type_stream = open(output_type_file, 'w');
    for type_index in xrange(len(index_to_type)):
        output_type_stream.write("%s\t%d\t%d\n" % (index_to_type[type_index], type_document_frequency[index_to_type[type_index]], type_term_frequency[index_to_type[type_index]]));
    output_type_stream.close();
    
    output_training_feature_file = os.path.join(output_directory, "train.feature.npy");
    output_training_label_file = os.path.join(output_directory, "train.label.npy");
    parse_data(label_information, type_information, input_training_file, output_training_feature_file, output_training_label_file, cutoff_df_threshold, cutoff_tf_threshold);
    
    output_testing_feature_file = os.path.join(output_directory, "test.feature.npy");
    output_testing_label_file = os.path.join(output_directory, "test.label.npy");
    parse_data(label_information, type_information, input_testing_file, output_testing_feature_file, output_testing_label_file, cutoff_df_threshold, cutoff_tf_threshold);

def parse_data(label_information, type_information, input_file, output_feature_file, output_label_file, cutoff_df_threshold=0, cutoff_tf_threshold=0):
    (label_to_index, index_to_label, label_frequency) = label_information
    (type_to_index, index_to_type, type_document_frequency, type_term_frequency) = type_information
    
    input_stream = open(input_file, 'r')
    data_features = numpy.zeros((0, len(type_to_index)))
    data_labels = numpy.zeros(0)
    for line in input_stream:
        line = line.strip();
        fields = line.split("\t");
        
        label = fields[0].strip();
        if (label in label_to_index):
            datum_label = label_to_index[label];
        else:
            sys.stderr.write("label not found: %s" % line)
            continue;
        
        datum_feature = numpy.zeros((1, len(type_to_index)))
        tokens = fields[1].split();
        for token in tokens:
            if token not in type_to_index:
                continue;
            if type_document_frequency[token] <= cutoff_df_threshold:
                continue;
            if type_term_frequency[token] <= cutoff_tf_threshold:
                continue;
            
            token_id = type_to_index[token];
            datum_feature[0, token_id] += 1;
        if numpy.sum(datum_feature) > 0:
            # print data_features.shape, datum_feature.shape
            data_features = numpy.vstack((data_features, datum_feature));
            data_labels = numpy.append(data_labels, datum_label);
        else:
            sys.stderr.write("document collapsed: %s" % line)
            continue;
        
    print data_features.dtype, data_features.shape, numpy.max(data_features), numpy.min(data_features)
    data_features = data_features.astype(numpy.uint8);
    print data_features.dtype, data_features.shape, numpy.max(data_features), numpy.min(data_features)
    data_labels = data_labels.astype(numpy.uint8);
    print data_labels.dtype, data_labels.shape, numpy.max(data_labels), numpy.min(data_labels)
     
    assert data_features.shape[0] == data_labels.shape[0]
    if data_features.shape[0] <= 0:
        sys.stderr.write("no feature/label extracted...")
        return;
    else:
        print "successfully generated %d data instances..." % data_labels.shape[0]
        numpy.save(output_feature_file, data_features)
        numpy.save(output_label_file, data_labels)
    
def collect_statistics(input_file):
    label_to_index = {};
    index_to_label = {};
    label_frequency = collections.defaultdict();
    
    # type_to_index = {};
    # index_to_type = {};
    type_document_frequency = collections.defaultdict();
    type_term_frequency = collections.defaultdict();
    
    input_stream = open(input_file, 'r')
    for line in input_stream:
        line = line.strip();
        fields = line.split("\t");
        
        label = fields[0].strip();
        if label not in label_to_index:
            label_to_index[label] = len(label_to_index)
            index_to_label[len(index_to_label)] = label;
            label_frequency[label] = 0;
        label_frequency[label] += 1;
        
        tokens = fields[1].split();
        for token in tokens:
            if token not in type_document_frequency:
                # type_to_index[token] = len(type_to_index);
                # index_to_type[len(index_to_type)] = token;
                type_document_frequency[token] = 0;
                type_term_frequency[token] = 0;
            type_document_frequency[token] += 1;
            type_term_frequency[token] += 1;
    
    type_to_index = {};
    index_to_type = {};
    for token, term_frequency in sorted(type_term_frequency.items(), key=operator.itemgetter(1), reverse=True):
        assert token not in type_to_index;
        type_to_index[token] = len(type_to_index);
        index_to_type[len(index_to_type)] = token;
    
    return (label_to_index, index_to_label, label_frequency), (type_to_index, index_to_type, type_document_frequency, type_term_frequency)
    
if __name__ == '__main__':
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    document_parser(input_directory, output_directory)
