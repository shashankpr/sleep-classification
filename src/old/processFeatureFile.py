from sklearn import preprocessing

import matplotlib as plt
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import pickle

# from sknn.mlp import Classifier, Layer
import pandas as pd
import numpy as np
import csv
import datetime as dt


def set_classifier(feature_files, label_files):
    # read from csv instead
    print "Reading training data..."
    feature_list = []
    label_list = []
    time_list = []
    processedDict = {}
    for file_itr in range(0, len(feature_files)):
        feature_file = feature_files[file_itr]
        label_file = label_files[file_itr]
        print "----------------------------"
        print "Feature File:"
        print feature_file
        print "Label File:"
        print label_file
        print "----------------------------"
        feature_list_file = []
        label_list_file = []
        label_time_list_file = []

        # ff = open(feature_file, 'r')
        ff = pd.read_csv(feature_file)
        # print ff.info
        # print ff['TIME']
        # reader = csv.reader(ff)
        for count, row in enumerate(ff['TIME']):
            # print row
            feature_row_dict = {}
            try:
                # print str(row[1])
                timeObject = dt.datetime.strptime(str(row), '%Y-%m-%d %H:%M:%S.%f')
                # timeObject		= timeObject + dt.timedelta(hours=5, minutes=30)
            except ValueError:
                timeObject = dt.datetime.strptime(str(row), '%Y-%m-%d %H:%M:%S')
                # timeObject 		= timeObject + dt.timedelta(hours=5, minutes=30)

            feature_row_dict['TIME'] = timeObject
            float_features = []

            float_features.append(float(ff['HSIGNAL'][count]))
            float_features.append(float(ff['RSIGNAL'][count]))
            feature_row_dict['FEATURES'] = float_features
            # print np.asarray(float_features).shape
            feature_list_file.append(feature_row_dict)
        print "----------------------------"
        print "Number of Epochs for Dozee:"
        print len(feature_list_file)
        print "----------------------------"

        # lf = open(label_file, 'r')
        lf = pd.read_csv(label_file, header=None, names=['TIME', 'LABEL'])
        # label_reader = csv.reader(lf)
        # print lf.info
        # print lf.ix[0]
        for count, label_row in enumerate(lf['TIME']):
            # print label_row
            try:
                labeltimeObject = dt.datetime.strptime(str(label_row), '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                labeltimeObject = dt.datetime.strptime(str(label_row), '%Y-%m-%d %H:%M:%S')

            label_time_list_file.append(labeltimeObject)
            label = lf['LABEL'][count]
            # print label
            label_list_file.append(label)

        file_epoch_counter = 0
        diff = -30
        for itr in range(0, len(label_time_list_file)):
            if (itr != len(label_time_list_file) - 1):
                label_time = label_time_list_file[itr]
                label_end_time = label_time_list_file[itr + 1]
                for dict in feature_list_file:
                    dict_time = dict['TIME']
                    epoch_start_time = dict_time - dt.timedelta(seconds=15)
                    epoch_end_time = dict_time + dt.timedelta(seconds=15)

                    if (epoch_start_time > label_time and epoch_end_time < label_end_time):
                        # if(int(label_list_file[itr]) != 4):
                        time_val = label_time + dt.timedelta(seconds=diff + 30)
                        feature_list.append(dict['FEATURES'])
                        label_list.append(int(label_list_file[itr]))
                        time_list.append(time_val)
                        file_epoch_counter = file_epoch_counter + 1
                        diff = diff + 30
                    else:
                        diff = -30
            print "----------------------------"
        print "Number of Epochs matched:"
        print file_epoch_counter
        print "----------------------------"
    print "----------------------------"
    print "Number of Total Epochs:"
    print len(feature_list)
    print "Number of Total Labels:"
    print len(label_list)
    print "----------------------------"
    print "Number of Timestamps"
    print len(time_list)
    print "----------------------------"

    # df = pd.DataFrame(processedDict)
    # df.to_csv("processed.csv")

    num_deep_epochs = label_list.count(1)
    num_light_epochs = label_list.count(2)
    num_rem_epochs = label_list.count(3)
    num_wake_epochs = label_list.count(4)

    print "----------------------------"
    print "Number of Deep Epochs:"
    print num_deep_epochs
    print "Number of Light Labels:"
    print num_light_epochs
    print "Number of REM Epochs:"
    print num_rem_epochs
    print "Number of Wake Labels:"
    print num_wake_epochs
    print "----------------------------"

    # X_train = np.array(feature_list)
    # y_train = np.array(label_list)
    X_train = feature_list
    y_train = label_list

    processedDict["TIME"] = time_list
    processedDict["FEATURES"] = X_train
    processedDict["LABELS"] = y_train

    # df = pd.DataFrame(processedDict)
    # df.to_csv("processed.csv")

    return processedDict


def set_classifierTest(feature_files, label_files):
    # read from csv instead
    print "Reading test data..."
    feature_list = []
    label_list = []
    time_list = []
    processedDictTest = {}
    for file_itr in range(0, len(feature_files)):
        feature_file = feature_files[file_itr]
        label_file = label_files[file_itr]
        print "----------------------------"
        print "Feature File:"
        print feature_file
        print "Label File:"
        print label_file
        print "----------------------------"
        feature_list_file = []
        label_list_file = []
        label_time_list_file = []

        ff = open(feature_file, 'r')
        reader = csv.reader(ff)
        # print reader[-1]
        for row in reader:
            feature_row_dict = {}
            try:
                timeObject = dt.datetime.strptime(str(row[0]), '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                timeObject = dt.datetime.strptime(str(row[0]), '%Y-%m-%d %H:%M:%S')

            feature_row_dict['TIME'] = timeObject
            float_features = []

            for feature in row[1:]:
                float_features.append(float(feature))
            feature_row_dict['FEATURES'] = float_features
            feature_list_file.append(feature_row_dict)
        print "----------------------------"
        print "Number of Epochs for Dozee:"
        print len(feature_list_file)
        print "----------------------------"
        lf = open(label_file, 'r')
        label_reader = csv.reader(lf)
        for label_row in label_reader:
            try:
                labeltimeObject = dt.datetime.strptime(str(label_row[0]), '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                labeltimeObject = dt.datetime.strptime(str(label_row[0]), '%Y-%m-%d %H:%M:%S')

            label_time_list_file.append(labeltimeObject)
            label_list_file.append(label_row[1])

        file_epoch_counter = 0
        diff = -30
        for itr in range(0, len(label_time_list_file)):
            if (itr != len(label_time_list_file) - 1):
                label_time = label_time_list_file[itr]
                label_end_time = label_time_list_file[itr + 1]
                for dict in feature_list_file:
                    dict_time = dict['TIME']
                    epoch_start_time = dict_time - dt.timedelta(seconds=15)
                    epoch_end_time = dict_time + dt.timedelta(seconds=15)

                    if (epoch_start_time > label_time and epoch_end_time < label_end_time):

                        time_val = label_time + dt.timedelta(seconds=diff + 30)
                        feature_list.append(dict['FEATURES'])
                        label_list.append(int(label_list_file[itr]))
                        time_list.append(time_val)
                        file_epoch_counter = file_epoch_counter + 1
                        diff = diff + 30
                    else:
                        diff = -30
        print "----------------------------"
        print "Number of Epochs matched:"
        print file_epoch_counter
        print "----------------------------"
    print "----------------------------"
    print "Number of Total TEST Epochs:"
    print len(feature_list)
    print "Number of Total TEST Labels:"
    print len(label_list)
    print "----------------------------"

    # X_train = feature_list
    # y_train = label_list
    y_test = []
    for x in label_list:
        y_test.append(int(x))

    num_deep_epochs = label_list.count(1)
    num_light_epochs = label_list.count(2)
    num_rem_epochs = label_list.count(3)
    num_wake_epochs = label_list.count(4)

    print "----------------------------"
    print "Number of Deep Epochs:"
    print num_deep_epochs
    print "Number of Light Labels:"
    print num_light_epochs
    print "Number of REM Epochs:"
    print num_rem_epochs
    print "Number of Wake Labels:"
    print num_wake_epochs
    print "----------------------------"

    X_test = np.asarray(feature_list)
    y_test = np.asarray(y_test)

    processedDictTest["TIME"] = time_list
    processedDictTest["FEATURES"] = X_test
    processedDictTest["LABELS"] = y_test
    return processedDictTest


def set_classifierValidation(feature_files, label_files):
    # read from csv instead
    print "Reading Validation data..."
    feature_list = []
    label_list = []
    time_list = []
    processedDictValidation = {}
    for file_itr in range(0, len(feature_files)):
        feature_file = feature_files[file_itr]
        label_file = label_files[file_itr]
        print "----------------------------"
        print "Feature File:"
        print feature_file
        print "Label File:"
        print label_file
        print "----------------------------"
        feature_list_file = []
        label_list_file = []
        label_time_list_file = []

        ff = open(feature_file, 'r')
        reader = csv.reader(ff)
        # print reader[-1]
        for row in reader:
            feature_row_dict = {}
            try:
                timeObject = dt.datetime.strptime(str(row[0]), '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                timeObject = dt.datetime.strptime(str(row[0]), '%Y-%m-%d %H:%M:%S')

            feature_row_dict['TIME'] = timeObject
            float_features = []

            for feature in row[1:]:
                float_features.append(float(feature))
            feature_row_dict['FEATURES'] = float_features
            feature_list_file.append(feature_row_dict)
        print "----------------------------"
        print "Number of Epochs for Dozee:"
        print len(feature_list_file)
        print "----------------------------"
        lf = open(label_file, 'r')
        label_reader = csv.reader(lf)
        for label_row in label_reader:
            try:
                labeltimeObject = dt.datetime.strptime(str(label_row[0]), '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                labeltimeObject = dt.datetime.strptime(str(label_row[0]), '%Y-%m-%d %H:%M:%S')

            label_time_list_file.append(labeltimeObject)
            label_list_file.append(label_row[1])

        file_epoch_counter = 0
        diff = -30
        for itr in range(0, len(label_time_list_file)):
            if (itr != len(label_time_list_file) - 1):
                label_time = label_time_list_file[itr]
                label_end_time = label_time_list_file[itr + 1]
                for dict in feature_list_file:
                    dict_time = dict['TIME']
                    epoch_start_time = dict_time - dt.timedelta(seconds=15)
                    epoch_end_time = dict_time + dt.timedelta(seconds=15)

                    if (epoch_start_time > label_time and epoch_end_time < label_end_time):

                        time_val = label_time + dt.timedelta(seconds=diff + 30)
                        feature_list.append(dict['FEATURES'])
                        label_list.append(int(label_list_file[itr]))
                        time_list.append(time_val)
                        file_epoch_counter = file_epoch_counter + 1
                        diff = diff + 30
                    else:
                        diff = -30
        print "----------------------------"
        print "Number of Epochs matched:"
        print file_epoch_counter
        print "----------------------------"
    print "----------------------------"
    print "Number of Total TEST Epochs:"
    print len(feature_list)
    print "Number of Total TEST Labels:"
    print len(label_list)
    print "----------------------------"

    # X_train = feature_list
    # y_train = label_list
    y_test = []
    for x in label_list:
        y_test.append(int(x))

    num_deep_epochs = label_list.count(1)
    num_light_epochs = label_list.count(2)
    num_rem_epochs = label_list.count(3)
    num_wake_epochs = label_list.count(4)

    print "----------------------------"
    print "Number of Deep Epochs:"
    print num_deep_epochs
    print "Number of Light Labels:"
    print num_light_epochs
    print "Number of REM Epochs:"
    print num_rem_epochs
    print "Number of Wake Labels:"
    print num_wake_epochs
    print "----------------------------"

    X_test = np.asarray(feature_list)
    y_test = np.asarray(y_test)

    processedDictValidation["TIME"] = time_list
    processedDictValidation["FEATURES"] = X_test
    processedDictValidation["LABELS"] = y_test
    return processedDictValidation


def set_classifierNew(feature_files):
    # read from csv instead
    print "Reading test data..."
    feature_list = []
    label_list = []
    time_list = []
    processedNew = {}
    feature_list_file = []
    for file_itr in range(0, len(feature_files)):
        feature_file = feature_files[file_itr]
        # label_file   = label_files[file_itr]
        print "----------------------------"
        print "Feature File:"
        print feature_file
        print "----------------------------"
        feature_list_file = []
        # label_list_file   = []
        # label_time_list_file   = []

        ff = open(feature_file, 'r')
        reader = csv.reader(ff)
        # print reader[-1]
        for row in reader:
            feature_row_dict = {}
            try:
                timeObject = dt.datetime.strptime(str(row[0]), '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                timeObject = dt.datetime.strptime(str(row[0]), '%Y-%m-%d %H:%M:%S')

            feature_row_dict['TIME'] = timeObject
            float_features = []

            for feature in row[1:]:
                float_features.append(float(feature))
            feature_row_dict['FEATURES'] = float_features
            feature_list_file.append(feature_row_dict)

        for dict in feature_list_file:
            dict_time = dict['TIME']
            dict_features = dict["FEATURES"]
            feature_list.append(dict_features)
            time_list.append(dict_time)
        print "----------------------------"
        print "Number of Epochs for Dozee:"
        print len(feature_list)
        print "----------------------------"

        X_test = np.asarray(feature_list)
        processedNew['FEATURES'] = X_test
        processedNew['TIME'] = time_list
    # pd.DataFrame(processedNew).to_csv("new_file.csv")
    return processedNew


def set_classifier_epoch_gen(feature_files, label_files):
    # read from csv instead
    print "Reading training data..."
    feature_list = []
    label_list = []
    time_list = []
    processedDict = {}
    for file_itr in range(0, len(feature_files)):
        feature_file = feature_files[file_itr]
        label_file = label_files[file_itr]
        print "----------------------------"
        print "Feature File:"
        print feature_file
        print "Label File:"
        print label_file
        print "----------------------------"
        feature_list_file = []
        label_list_file = []
        label_time_list_file = []

        ff = open(feature_file, 'r')
        reader = csv.reader(ff)
        for row in reader:
            feature_row_dict = {}
            try:
                timeObject = dt.datetime.strptime(str(row[0]), '%Y-%m-%d %H:%M:%S.%f')
            # timeObject		= timeObject + dt.timedelta(hours=5, minutes=30)
            except ValueError:
                timeObject = dt.datetime.strptime(str(row[0]), '%Y-%m-%d %H:%M:%S')
            # timeObject 		= timeObject + dt.timedelta(hours=5, minutes=30)

            feature_row_dict['TIME'] = timeObject
            float_features = []
            for feature in row[1:]:
                float_features.append(float(feature))
            feature_row_dict['FEATURES'] = float_features
            feature_list_file.append(feature_row_dict)
        print "----------------------------"
        print "Number of Epochs for Dozee:"
        print len(feature_list_file)
        print "----------------------------"

        lf = open(label_file, 'r')
        label_reader = csv.reader(lf)
        for label_row in label_reader:
            try:
                labeltimeObject = dt.datetime.strptime(str(label_row[0]), '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                labeltimeObject = dt.datetime.strptime(str(label_row[0]), '%Y-%m-%d %H:%M:%S')

            label_time_list_file.append(labeltimeObject)
            label_list_file.append(label_row[1])

        file_epoch_counter = 0
        diff = -30
        for itr in range(0, len(label_time_list_file)):
            if (itr != len(label_time_list_file) - 1):
                label_time = label_time_list_file[itr]
                label_end_time = label_time_list_file[itr + 1]
                for dict in feature_list_file:
                    dict_time = dict['TIME']
                    epoch_start_time = dict_time - dt.timedelta(seconds=15)
                    epoch_end_time = dict_time + dt.timedelta(seconds=15)

                    if (epoch_start_time > label_time and epoch_end_time < label_end_time):
                        # if(int(label_list_file[itr]) != 4):
                        time_val = label_time + dt.timedelta(seconds=diff + 30)
                        feature_list.append(dict['FEATURES'])
                        label_list.append(int(label_list_file[itr]))
                        time_list.append(time_val)
                        file_epoch_counter = file_epoch_counter + 1
                        diff = diff + 30
                    else:
                        diff = -30
            print "----------------------------"
        print "Number of Epochs matched:"
        print file_epoch_counter
        print "----------------------------"
    print "----------------------------"
    print "Number of Total Epochs:"
    print len(feature_list)
    print "Number of Total Labels:"
    print len(label_list)
    print "----------------------------"
    print "Number of Timestamps"
    print len(time_list)
    print "----------------------------"

    processedDict["TIME"] = time_list
    processedDict["FEATURES"] = feature_list
    processedDict["LABELS"] = label_list
    # df = pd.DataFrame(processedDict)
    # df.to_csv("processed.csv")

    num_deep_epochs = label_list.count(1)
    num_light_epochs = label_list.count(2)
    num_rem_epochs = label_list.count(3)
    num_wake_epochs = label_list.count(4)

    print "----------------------------"
    print "Number of Deep Epochs:"
    print num_deep_epochs
    print "Number of Light Labels:"
    print num_light_epochs
    print "Number of REM Epochs:"
    print num_rem_epochs
    print "Number of Wake Labels:"
    print num_wake_epochs
    print "----------------------------"

    X_train = np.array(feature_list)
    y_train = np.array(label_list)
    return processedDict
