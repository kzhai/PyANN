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
        # print "%s\t%d\n" % (index_to_label[label_index], label_frequency[index_to_label[label_index]])
    output_label_stream.close()
    print "successfully parsed label file..."
    
    (type_to_index, index_to_type, type_document_frequency, type_term_frequency) = type_information
    output_type_file = os.path.join(output_directory, "type.info.dat")
    output_type_stream = open(output_type_file, 'w');
    for type_index in xrange(len(index_to_type)):
        output_type_stream.write("%s\t%d\t%d\n" % (index_to_type[type_index], type_document_frequency[index_to_type[type_index]], type_term_frequency[index_to_type[type_index]]));
    output_type_stream.close();
    print "successfully parsed type file..."
    
    output_training_feature_file = os.path.join(output_directory, "train.feature.npy");
    output_training_label_file = os.path.join(output_directory, "train.label.npy");
    parse_data(label_information, type_information, input_training_file, output_training_feature_file, output_training_label_file, cutoff_df_threshold, cutoff_tf_threshold);
    print "successfully parsed training file..."
    
    output_testing_feature_file = os.path.join(output_directory, "test.feature.npy");
    output_testing_label_file = os.path.join(output_directory, "test.label.npy");
    parse_data(label_information, type_information, input_testing_file, output_testing_feature_file, output_testing_label_file, cutoff_df_threshold, cutoff_tf_threshold);
    print "successfully parsed testing file..."

def parse_data(label_information, type_information, input_file, output_feature_file, output_label_file, cutoff_df_threshold=0, cutoff_tf_threshold=0):
    (label_to_index, index_to_label, label_frequency) = label_information
    (type_to_index, index_to_type, type_document_frequency, type_term_frequency) = type_information
    
    input_stream = open(input_file, 'r')
    data_features = numpy.zeros((100000, len(type_to_index)));
    data_labels = numpy.zeros(100000);
    number_of_documents = 0;
    for line in input_stream:
        line = line.strip();
        fields = line.split("\t");
        if len(fields) != 2:
            sys.stderr.write("document collapsed: %s\n" % line)
            continue;
        
        label = fields[0].strip();
        if (label in label_to_index):
            datum_label = label_to_index[label];
        else:
            sys.stderr.write("label not found: %s\n" % line)
            continue;
        
        datum_feature = numpy.zeros(len(type_to_index))
        tokens = fields[1].split();
        for token in tokens:
            if token not in type_to_index:
                continue;
            if type_document_frequency[token] <= cutoff_df_threshold:
                continue;
            if type_term_frequency[token] <= cutoff_tf_threshold:
                continue;
            
            token_id = type_to_index[token];
            datum_feature[token_id] += 1;
            
        if numpy.sum(datum_feature) > 0:
            # datum_feature /= numpy.sum(datum_feature);
            data_features[number_of_documents, :] = datum_feature;
            data_labels[number_of_documents] = datum_label;
            number_of_documents += 1;
        else:
            sys.stderr.write("document collapsed: %s\n" % line)
            continue;
        
        if number_of_documents % 1000 == 0:
            print "successfully parsed %d data features..." % number_of_documents;
    
    data_features = data_features[:number_of_documents, :];
    data_labels = data_labels[:number_of_documents];
    
    print data_features.dtype, data_features.shape, numpy.max(data_features), numpy.min(data_features)
    # data_features = data_features.astype(numpy.float32);
    data_features = data_features.astype(numpy.uint16);
    print data_features.dtype, data_features.shape, numpy.max(data_features), numpy.min(data_features)
    data_labels = data_labels.astype(numpy.uint8);
    print data_labels.dtype, data_labels.shape, numpy.max(data_labels), numpy.min(data_labels)
     
    assert data_features.shape[0] == data_labels.shape[0]
    if data_features.shape[0] <= 0:
        sys.stderr.write("no feature/label extracted...\n")
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
        if len(fields) != 2:
            sys.stderr.write("document collapsed: %s\n" % line)
            continue;
        
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
