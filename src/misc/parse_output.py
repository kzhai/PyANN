import collections;
import cPickle
import heapq;
import os;
import operator;
import math;
import numpy;
import numpy.random;
import re;
import scipy;
import scipy.linalg;
import sklearn
import sklearn.covariance
import sys
import time;
import theano
import theano.tensor
#import matplotlib
#import matplotlib.pyplot

train_minibatch_pattern = re.compile(r'train result: epoch (?P<epoch>[\d]+?), minibatch (?P<minibatch>[\d]+?), loss (?P<loss>[\d.]+?), accuracy (?P<accuracy>[\d.]+?)%');
valid_minibatch_pattern = re.compile(r'validate result: epoch (?P<epoch>[\d]+?), minibatch (?P<minibatch>[\d]+?), loss (?P<loss>[\d.]+?), accuracy (?P<accuracy>[\d.]+?)%');
test_minibatch_pattern = re.compile(r'test result: epoch (?P<epoch>[\d]+?), minibatch (?P<minibatch>[\d]+?), loss (?P<loss>[\d.]+?), accuracy (?P<accuracy>[\d.]+?)%');

train_epoch_pattern = re.compile(r'train result: epoch (?P<epoch>[\d]+?), duration (?P<duration>[\d.]+?)s, loss (?P<loss>[\d.]+?), accuracy (?P<accuracy>[\d.]+?)%');
best_validate_minibatch_pattern = re.compile(r'best model found: epoch (?P<epoch>\d+?), minibatch (?P<minibatch>\d+?), accuracy (?P<accuracy>[\d.]+?)%');

def parse_output(input_file):
    input_stream = open(input_file, "r");

    train_minibatches = numpy.zeros((0, 4));
    valid_minibatches = numpy.zeros((0, 4));
    test_minibatches = numpy.zeros((0, 4));

    train_epochs = numpy.zeros((0, 4));
    best_models = numpy.zeros((0, 4));

    best_found = False;
    for line in input_stream:
        line = line.strip();

        if len(line) == 0:
            continue;

        matcher_found = False;

        train_minibatch = minibatch_pattern_match(train_minibatch_pattern, line);
        if train_minibatch!=None:
            train_minibatches = numpy.vstack((train_minibatches, train_minibatch));
            matcher_found = True;
        valid_minibatch = minibatch_pattern_match(valid_minibatch_pattern, line);
        if valid_minibatch != None:
            valid_minibatches = numpy.vstack((valid_minibatches, valid_minibatch));
            matcher_found = True;
        test_minibatch = minibatch_pattern_match(test_minibatch_pattern, line);
        if test_minibatch != None:
            test_minibatches = numpy.vstack((test_minibatches, test_minibatch));
            matcher_found = True;
            if best_found:
                best_models = numpy.vstack((best_models, test_minibatch));
                best_found = False;

        matcher = re.match(train_epoch_pattern, line);
        if matcher is not None:
            epoch = int(matcher.group("epoch"));
            duration = float(matcher.group("duration"));
            loss = float(matcher.group("loss"));
            accuracy = float(matcher.group("accuracy"));

            train_epoch = numpy.asarray([epoch, duration, loss, accuracy]);
            train_epochs = numpy.vstack((train_epochs, train_epoch));

            matcher_found = True;

        matcher = re.match(best_validate_minibatch_pattern, line);
        if matcher is not None:
            epoch = int(matcher.group("epoch"));
            minibatch = int(matcher.group("minibatch"));
            accuracy = float(matcher.group("accuracy"));

            best_found = True;

            matcher_found = True;

        if matcher_found:
            continue;
        else:
            print line;

    print train_minibatches.shape
    print valid_minibatches.shape
    print test_minibatches.shape

    print train_epochs.shape
    print best_models.shape

    print best_models[-1, -1], numpy.max(best_models[:, -1])

def minibatch_pattern_match(pattern, line):
    matcher = re.match(pattern, line);
    if matcher is not None:
        # print "check"
        epoch = int(matcher.group("epoch"));
        minibatch = int(matcher.group("minibatch"));
        loss = float(matcher.group("loss"));
        accuracy = float(matcher.group("accuracy"));

        temp_minibatch = numpy.asarray([epoch, minibatch, loss, accuracy]);
        return temp_minibatch;

    return None;

if __name__ == '__main__':
    input_file = sys.argv[1]

    parse_output(input_file)
