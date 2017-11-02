import numpy as np
import logging
import h5py as h5

from utils import threadsafe_generator

class DataGenerator(object):
    def __init__(self, file_path, BATCH_SIZE = 32, SEQ_LEN = 64, n_classes = 4, split_criterion = 0.33,
                 shuffle_opt = False, FEATURES_DIM = 2):
        """

        Args:
            file_path:
            BATCH_SIZE:
            SEQ_LEN:
            n_classes:
            split_criterion:
            shuffle_opt:
            FEATURES_DIM:
        """
        self.file_path = file_path
        self.BATCH_SIZE = BATCH_SIZE
        self.SEQ_LEN = SEQ_LEN
        self.n_classes = n_classes
        self.shuffle = shuffle_opt
        self.split_criterion = split_criterion
        self.FEATURES_DIM = FEATURES_DIM
        self.logger = logging.getLogger(__name__)

        self.hrv_feature_list = ['Heart Rate', 'Breathing Rate', 'Movement', 'VLF', 'LF', 'HF', 'Relative HF']

        self.sleep_data = h5.File(self.file_path, 'a')
        self.logger.info("File: {}".format(self.sleep_data))
        masterFileKeys = self.sleep_data.keys()
        validation_split = int(len(masterFileKeys) * self.split_criterion)

        self.validation_subject_keys = masterFileKeys[0: validation_split]
        self.training_subject_keys = masterFileKeys[validation_split:]

        self.logger.debug("Training : {}".format(self.training_subject_keys))
        self.logger.debug("Validation: {}".format(self.validation_subject_keys))

    @threadsafe_generator
    def generate_training_sequences(self, epochs=4):
        """

        Args:
            epochs:

        Returns:

        """
        # Infinite loop
        while 1:
            if self.FEATURES_DIM > 2:
                self.logger.info("Generating Training Data - HRV features")
                for filteredSignal, labels in self.__load_hrv_features(subject_keys=self.training_subject_keys):
                    # self.logger.debug("Received data from HDF : {} -- {}".format(labels.shape, filteredSignal.shape))

                    # Generate Sequences
                    for filteredSignal_seq, labels_seq in self.__create_subsequence(filteredSignal, labels):
                        self.logger.debug("New shapes : {} --- {}".format(filteredSignal_seq.shape, labels_seq.shape))

                        # Generate order of exploration of dataset
                        indexes = self.__get_exploration_order(filteredSignal_seq)

                        # Generate batches
                        max_batches = int(len(indexes) / self.BATCH_SIZE)
                        remainder_samples = len(indexes) % self.BATCH_SIZE
                        if remainder_samples:
                            max_batches = max_batches + 1

                        self.logger.info("Total batches : {}".format(max_batches))
                        for i in range(max_batches):
                            # Get data in batches of batch_size (here, 32)

                            if i == max_batches - 1:
                                filteredSignal_in_batches = [filteredSignal_seq[k] for k in
                                                             indexes[i * self.BATCH_SIZE:]]
                                labels_in_batches = [labels_seq[k] for k in indexes[i * self.BATCH_SIZE:]]
                            else:
                                filteredSignal_in_batches = [filteredSignal_seq[k] for k in
                                                             indexes[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]]
                                labels_in_batches = [labels_seq[k] for k in
                                                     indexes[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]]

                            labels_in_batches_encoded = self.__categorical(np.array(labels_in_batches))

                            self.logger.info("Sending data of shape : {} and {}"
                                             .format(np.array(filteredSignal_in_batches).shape,
                                                     np.array(labels_in_batches_encoded).shape))
                            yield np.array(filteredSignal_in_batches), np.array(labels_in_batches_encoded)

            # If number of features == 2, then use the default generator method.
            else:
                self.logger.info("Generating Training Data - 2 features")
                for filteredSignal, labels in self.__load_hdf_generator(subject_keys=self.training_subject_keys,
                                                                        epochs=epochs):
                    # self.logger.debug("Received data from HDF : {} -- {}".format(labels.shape, filteredSignal.shape))

                    # Generate Sequences
                    for filteredSignal_seq, labels_seq in self.__create_subsequence(filteredSignal, labels):
                        self.logger.debug("New shapes : {} --- {}".format(filteredSignal_seq.shape, labels_seq.shape))

                        # Generate order of exploration of dataset
                        indexes = self.__get_exploration_order(filteredSignal_seq)

                        # Generate batches
                        max_batches = int(len(indexes) / self.BATCH_SIZE)
                        remainder_samples = len(indexes)%self.BATCH_SIZE
                        if remainder_samples:
                            max_batches = max_batches + 1

                        self.logger.info("Total batches : {}".format(max_batches))
                        for i in range(max_batches):
                            # Get data in batches of batch_size (here, 32)

                            if i == max_batches - 1:
                                filteredSignal_in_batches = [filteredSignal_seq[k] for k in indexes[i * self.BATCH_SIZE:]]
                                labels_in_batches = [labels_seq[k] for k in indexes[i * self.BATCH_SIZE:]]
                            else:
                                filteredSignal_in_batches = [filteredSignal_seq[k] for k in
                                                             indexes[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]]
                                labels_in_batches = [labels_seq[k] for k in
                                                     indexes[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]]

                            labels_in_batches_encoded = self.__categorical(np.array(labels_in_batches))

                            self.logger.info("Sending data of shape : {} and {}"
                                             .format(np.array(filteredSignal_in_batches).shape,
                                                     np.array(labels_in_batches_encoded).shape))
                            yield np.array(filteredSignal_in_batches), np.array(labels_in_batches_encoded)

    @threadsafe_generator
    def generate_validation_sequences(self, epochs = 4):
        """

        Args:
            epochs:

        Returns:

        """
        while 1:
            if self.FEATURES_DIM > 2:
                self.logger.info("Generating Validation Data - HRV Features")
                for filteredSignal, labels in self.__load_hrv_features(subject_keys=self.validation_subject_keys):
                    # self.logger.debug("Received data from HDF : {} -- {}".format(labels.shape, filteredSignal.shape))

                    # Generate Sequences
                    for filteredSignal_seq, labels_seq in self.__create_subsequence(filteredSignal, labels):
                        self.logger.debug("New shapes : {} --- {}".format(filteredSignal_seq.shape, labels_seq.shape))

                        # Generate order of exploration of dataset
                        indexes = self.__get_exploration_order(filteredSignal_seq)

                        # Generate batches
                        max_batches = int(len(indexes) / self.BATCH_SIZE)
                        remainder_samples = len(indexes) % self.BATCH_SIZE
                        if remainder_samples:
                            max_batches = max_batches + 1

                        self.logger.info("Total batches : {}".format(max_batches))
                        for i in range(max_batches):
                            # Get data in batches of batch_size (here, 32)

                            if i == max_batches - 1:
                                filteredSignal_in_batches = [filteredSignal_seq[k] for k in
                                                             indexes[i * self.BATCH_SIZE:]]
                                labels_in_batches = [labels_seq[k] for k in indexes[i * self.BATCH_SIZE:]]
                            else:
                                filteredSignal_in_batches = [filteredSignal_seq[k] for k in
                                                             indexes[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]]
                                labels_in_batches = [labels_seq[k] for k in
                                                     indexes[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]]

                            labels_in_batches_encoded = self.__categorical(np.array(labels_in_batches))

                            self.logger.info("Sending data of shape : {} and {}"
                                             .format(np.array(filteredSignal_in_batches).shape,
                                                     np.array(labels_in_batches_encoded).shape))
                            yield np.array(filteredSignal_in_batches), np.array(labels_in_batches_encoded)

            # If number of features == 2, then use the default generator method.
            else:
                self.logger.info("Generating Validation Data - 2 Features")
                for filteredSignal, labels in self.__load_hdf_generator(subject_keys=self.validation_subject_keys,
                                                                        epochs=epochs):
                    # self.logger.debug("Received data from HDF : {} -- {}".format(labels.shape, filteredSignal.shape))

                    # Generate Sequences
                    for filteredSignal_seq, labels_seq in self.__create_subsequence(filteredSignal, labels):
                        self.logger.debug("New shapes : {} --- {}".format(filteredSignal_seq.shape, labels_seq.shape))

                        # Generate order of exploration of dataset
                        indexes = self.__get_exploration_order(filteredSignal_seq)

                        # Generate batches
                        max_batches = int(len(indexes) / self.BATCH_SIZE)
                        remainder_samples = len(indexes)%self.BATCH_SIZE
                        if remainder_samples:
                            max_batches = max_batches + 1

                        self.logger.info("Total batches : {}".format(max_batches))
                        for i in range(max_batches):
                            # Get data in batches of batch_size (here, 32)

                            if i == max_batches - 1:
                                filteredSignal_in_batches = [filteredSignal_seq[k] for k in indexes[i * self.BATCH_SIZE:]]
                                labels_in_batches = [labels_seq[k] for k in indexes[i * self.BATCH_SIZE:]]
                            else:
                                filteredSignal_in_batches = [filteredSignal_seq[k] for k in
                                                             indexes[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]]
                                labels_in_batches = [labels_seq[k] for k in
                                                     indexes[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]]


                            labels_in_batches_encoded = self.__categorical(np.array(labels_in_batches))

                            self.logger.info("Sending data of shape : {} and {}"
                                             .format(np.array(filteredSignal_in_batches).shape,
                                                     np.array(labels_in_batches_encoded).shape))
                            yield np.array(filteredSignal_in_batches), np.array(labels_in_batches_encoded)

    #TODO : Need to come up with better test_samples generation method
    def generate_test_sequences(self):
        """

        Returns:

        """
        single_signal = []
        labels_list = []

        result_feat = []
        result_lab = []

        # If using HRV features then use this method
        if self.FEATURES_DIM > 2:
            self.logger.info("Generating HRV data")
            # self.logger.info("Validation Files : {}".format(self.validation_subject_keys))
            for subject in self.validation_subject_keys:
                subjectGroupHandle = self.sleep_data[subject]
                subjectGroupKeys = subjectGroupHandle.keys()

                # nimhansLabels = []
                for dateValue in subjectGroupKeys:
                    dateGroupHandle = subjectGroupHandle[dateValue]
                    dateGroupKeys = dateGroupHandle.keys()

                    processedGroupHandle = dateGroupHandle['PROCESSED']
                    filteredGroupHandle = dateGroupHandle['FILTERED']

                    filteredGroupKeys = filteredGroupHandle.keys()
                    processedGroupKeys = processedGroupHandle.keys()
                    HRVdatasetHandle = processedGroupHandle['PROCESSED_VALUES_WITH_HRV']

                    labels_list = []

                    single_signal = np.asarray((HRVdatasetHandle[self.hrv_feature_list[0]],
                                             HRVdatasetHandle[self.hrv_feature_list[1]],
                                             HRVdatasetHandle[self.hrv_feature_list[2]],
                                             HRVdatasetHandle[self.hrv_feature_list[3]],
                                             HRVdatasetHandle[self.hrv_feature_list[4]],
                                             HRVdatasetHandle[self.hrv_feature_list[5]],
                                             HRVdatasetHandle[self.hrv_feature_list[6]]
                                             ))

                    self.logger.info("File name : {}, Date : {}".format(subject, dateValue))
                    single_signal = single_signal.transpose()
                    self.logger.debug("HRV Signal Transpose shape: {}".format(single_signal.shape))

                    for count, filteredValue in enumerate(filteredGroupKeys):
                        datasetHandle = filteredGroupHandle[filteredValue]
                        if count < len(HRVdatasetHandle):
                            labels_list.append(datasetHandle.attrs['label'])
                        else:
                            break

                    self.logger.debug("Label list shape : {}".format(np.array(labels_list).shape))

        # If number of features == 2, then use the default generator method.
        else:
            self.logger.info("Loading Test Data")
            for subject in self.validation_subject_keys:
                subjectGroupHandle = self.sleep_data[subject]
                subjectGroupKeys = subjectGroupHandle.keys()

                dateValue = subjectGroupKeys[0]
                dateGroupHandle = subjectGroupHandle[dateValue]

                filteredGroupHandle = dateGroupHandle['FILTERED']
                filteredGroupKeys = filteredGroupHandle.keys()

                self.logger.info("Total epochs : {}".format(len(filteredGroupKeys)))

                for epoch_count, filteredValue in enumerate(filteredGroupKeys):
                    self.logger.debug("Filtered values : {}".format(filteredValue))
                    dataSetHandle = filteredGroupHandle[filteredValue]

                    for dataSet in range(len(dataSetHandle)):
                        single_signal.append([float(dataSetHandle[dataSet][1]), float(dataSetHandle[dataSet][2])])
                        labels_list.append(dataSetHandle.attrs['label'])

        for index in range(len(single_signal) - self.SEQ_LEN):
            # result.append(hdata[index: index + sequence_length])
            result_feat.append(single_signal[index: index + self.SEQ_LEN])
            result_lab.append(labels_list[index])

        single_signal = []
        labels_list = []

        # labels_encoded = self.__categorical(np.array(result_lab))
        self.logger.debug("Test X shape : {}, Test Y Shape : {}".format(np.array(result_feat).shape,
                                                                   np.array(result_lab).shape))

        return np.array(result_feat), np.array(result_lab)


    def __get_exploration_order(self, filteredSignal):
        """

        Args:
            filteredSignal:

        Returns:

        """
        # Find exploration order
        indexes = np.arange(len(filteredSignal))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __create_subsequence(self, filteredSignal, labels):
        """

        Args:
            filteredSignal:
            labels:

        Returns:

        """
        result_feat = []
        result_lab = []

        if self.FEATURES_DIM > 2:
            for index in range(filteredSignal.shape[0] - self.SEQ_LEN):
                result_feat.append(filteredSignal[index: index + self.SEQ_LEN])
                result_lab.append(labels[index])
        else:
            for index in range(len(filteredSignal) - self.SEQ_LEN):
                result_feat.append(filteredSignal[index: index + self.SEQ_LEN])
                result_lab.append(labels[index])

        yield np.array(result_feat), np.array(result_lab)

    def __categorical(self, labels):
        """

        Args:
            labels:

        Returns:

        """
        # As labels start from 1, we use j+1
        return np.array([[1 if labels[i] == j+1 else 0 for j in range(self.n_classes)]
                         for i in range(labels.shape[0])])


    def __load_hdf_generator(self, subject_keys, epochs):
        """

        Args:
            subject_keys:
            epochs:

        Returns:

        """
        for subject in subject_keys:
            subjectGroupHandle = self.sleep_data[subject]
            subjectGroupKeys = subjectGroupHandle.keys()

            # nimhansLabels = []
            for dateValue in subjectGroupKeys:
                dateGroupHandle = subjectGroupHandle[dateValue]
                dateGroupKeys = dateGroupHandle.keys()

                filteredGroupHandle = dateGroupHandle['FILTERED']
                filteredGroupKeys = filteredGroupHandle.keys()

                single_signal = []
                labels_list = []

                self.logger.info("Total epochs : {}".format(len(filteredGroupKeys)))
                for epoch_count, filteredValue in enumerate(filteredGroupKeys):

                    self.logger.debug("Filtered values : {}".format(filteredValue))

                    dataSetHandle = filteredGroupHandle[filteredValue]

                    for dataSet in range(len(dataSetHandle)):
                        single_signal.append([float(dataSetHandle[dataSet][1]), float(dataSetHandle[dataSet][2])])
                        labels_list.append(dataSetHandle.attrs['label'])

                    if (epoch_count+1) % epochs == 0:
                        self.logger.info("Collecting and sending samples ...")
                        self.logger.debug("Single signal len : {}".format(len(single_signal)))
                        self.logger.debug('Single Signal Shape :{}'.format(np.array(single_signal).shape))
                        # self.logger.debug('Padded signal shape : {}'.format(single_signal_pad.shape))

                        # self.logger.debug("Labels : {}".format(labels_list))
                        self.logger.debug("Label len : {}".format(len(labels_list)))
                        self.logger.debug("Label shape : {}".format(np.array(labels_list).shape))

                        yield single_signal, labels_list

                        single_signal = []
                        labels_list = []
                    else:
                        continue

    def __load_hrv_features(self, subject_keys):
        """

        Args:
            subject_keys:

        Returns:

        """
        self.logger.info("Generating HRV data")
        # self.logger.info("Validation Files : {}".format(self.validation_subject_keys))
        for subject in subject_keys:
            subjectGroupHandle = self.sleep_data[subject]
            subjectGroupKeys = subjectGroupHandle.keys()

            # nimhansLabels = []
            for dateValue in subjectGroupKeys:
                dateGroupHandle = subjectGroupHandle[dateValue]
                dateGroupKeys = dateGroupHandle.keys()

                processedGroupHandle = dateGroupHandle['PROCESSED']
                filteredGroupHandle = dateGroupHandle['FILTERED']

                filteredGroupKeys = filteredGroupHandle.keys()
                processedGroupKeys = processedGroupHandle.keys()
                HRVdatasetHandle = processedGroupHandle['PROCESSED_VALUES_WITH_HRV']

                labels_list = []

                hrv_signal = np.asarray((HRVdatasetHandle[self.hrv_feature_list[0]],
                                         HRVdatasetHandle[self.hrv_feature_list[1]],
                                         HRVdatasetHandle[self.hrv_feature_list[2]],
                                         HRVdatasetHandle[self.hrv_feature_list[3]],
                                         HRVdatasetHandle[self.hrv_feature_list[4]],
                                         HRVdatasetHandle[self.hrv_feature_list[5]],
                                         HRVdatasetHandle[self.hrv_feature_list[6]]
                                         ))

                # Replace NaN values with 0 and inf (-inf) with largest (smallest) values.
                hrv_signal = np.nan_to_num(hrv_signal, copy=False)

                self.logger.info("File name : {}, Date : {}".format(subject, dateValue))

                hrv_signal_transpose = hrv_signal.transpose()
                self.logger.debug("HRV Signal Transpose shape: {}".format(hrv_signal_transpose.shape))

                for count, filteredValue in enumerate(filteredGroupKeys):
                    datasetHandle = filteredGroupHandle[filteredValue]
                    if count < len(HRVdatasetHandle):
                        labels_list.append(datasetHandle.attrs['label'])
                    else:
                        break

                self.logger.debug("Label list shape : {}".format(np.array(labels_list).shape))

                yield hrv_signal_transpose, labels_list

    def training_sample_count(self):
        """

        Returns:

        """
        training_samples = 0

        if self.FEATURES_DIM > 2:
            for subject in self.training_subject_keys:
                subjectGroupHandle = self.sleep_data[subject]
                training_samples = training_samples + subjectGroupHandle.attrs['hrv_sample_count']

            self.logger.info("Training samples : {}".format(training_samples))

        else:
            for subject in self.training_subject_keys:
                subjectGroupHandle = self.sleep_data[subject]
                training_samples = training_samples + subjectGroupHandle.attrs['sample_count']

            self.logger.info("Training samples : {}".format(training_samples))
        return int(training_samples)

    def validation_sample_count(self):
        """

        Returns:

        """
        validation_samples = 0

        # Access the `hrv_sample_count` if number of features > 2
        if self.FEATURES_DIM > 2:
            for subject in self.validation_subject_keys:
                subjectGroupHandle = self.sleep_data[subject]
                validation_samples= validation_samples+ subjectGroupHandle.attrs['hrv_sample_count']

        # Access the `sample_count` if number of features == 2
        else:
            for subject in self.validation_subject_keys:
                subjectGroupHandle = self.sleep_data[subject]
                validation_samples= validation_samples+ subjectGroupHandle.attrs['sample_count']

        self.logger.info("Validation samples : {}".format(validation_samples))
        return int(validation_samples)

    def get_total_sample_count(self):
        """

        Returns:

        """

        if self.FEATURES_DIM > 2:
            total_sample_count = self.sleep_data.attrs['total_hrv_samples']
        else:
            total_sample_count = self.sleep_data.attrs['total_sample_count']

        return int(total_sample_count)

    def check_HDF_attrs(self):
        if len(self.sleep_data.attrs.keys()) < 2 :
            self.__gen_sample_count()
        else:
            return

    def __gen_sample_count(self):
        """

        Returns:

        """
        subject_sample_size = 0
        total_sample_count = 0
        hrv_sample_count = 0
        total_hrv_samples = 0

        self.logger.info("File : {}".format(self.sleep_data))
        masterFileKeys = self.sleep_data.keys()
        for subject in masterFileKeys:
            subjectGroupHandle = self.sleep_data[subject]
            subjectGroupKeys = subjectGroupHandle.keys()

            for dateValue in subjectGroupKeys:
                dateGroupHandle = subjectGroupHandle[dateValue]
                dateGroupKeys = dateGroupHandle.keys()

                filteredGroupHandle = dateGroupHandle['FILTERED']
                processedGroupHandle = dateGroupHandle['PROCESSED']
                processedGroupKeys = processedGroupHandle.keys()
                filteredGroupKeys = filteredGroupHandle.keys()
                HRVdatasetHandle = processedGroupHandle['PROCESSED_VALUES_WITH_HRV']

                hrv_sample_count = hrv_sample_count + HRVdatasetHandle.size

                for epoch_count, filteredValue in enumerate(filteredGroupKeys):
                    dataSetHandle = filteredGroupHandle[filteredValue]
                    subject_sample_size = subject_sample_size + dataSetHandle.size

            total_sample_count = total_sample_count + subject_sample_size
            total_hrv_samples = total_hrv_samples + hrv_sample_count
            self.logger.debug("subject handle : {}".format(subjectGroupHandle))

            subjectGroupHandle.attrs.create('sample_count', subject_sample_size, dtype=int)
            subjectGroupHandle.attrs.create('hrv_sample_count', hrv_sample_count, dtype=int)
            self.logger.info(
                "Keys : {}, Values : {}".format(subjectGroupHandle.attrs.keys(), subjectGroupHandle.attrs.values()))

            subject_sample_size = 0
            hrv_sample_count = 0

        self.sleep_data.attrs.create('total_sample_count', data=total_sample_count, dtype=int)
        self.sleep_data.attrs.create('total_hrv_samples', data=total_hrv_samples, dtype=int)

        self.logger.info("Total number of samples : {}".format(total_sample_count))
        self.logger.info("Total number of HRVsamples : {}".format(total_hrv_samples))

    def get_data(self):
        """

        Returns:

        """
        return list(self.generate_training_sequences(epochs=4))