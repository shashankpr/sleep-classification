from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM

import numpy as np
import time
import logging

import model_callbacks
from data_generate import DataGenerator
from utils import Metrics

seed = 7
np.random.seed(seed=seed)

class RunLSTM(object):
    def __init__(self, file_path, BATCH_SIZE = 32, SEQ_LEN = 64, n_classes = 4, FEATURES_DIM = 7, EPOCHS = 10,
                 shuffle_opt = False, nb_workers = 2):
        """Initializing the hyper-parameters

        Args:
            file_path        : Path to the HDF file
            BATCH_SIZE  (int): Number of batches of data to be supplied to the model
            SEQ_LEN     (int): Number of sequences to be visited/remembered before processing the next sample point.
            n_classes   (int): Number of classes/outputs of the training data.
            FEATURES_DIM (int): Total number of features to consider for training. 2 = Heart & Breathing, 7 = HRV features
            EPOCHS      (int): Total number of epochs to train the model.
            shuffle_opt (Bool): Whether to shuffle the data or not.
            nb_workers  (int): Number of workers/threads to spawn for fit_generator()
        """

        self.file_path = file_path
        self.SEQ_LEN = SEQ_LEN
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.FEATURES_DIM = FEATURES_DIM
        self.n_classes = n_classes
        self.shuffle = shuffle_opt
        self.nb_workers = nb_workers
        self.logger = logging.getLogger(__name__)

    def lstm_model(self):
        """

        Returns:

        """
        layers = [self.FEATURES_DIM, 50, 100, self.n_classes]
        model = Sequential()

        model.add(LSTM(
            units=layers[1],
            input_shape=(self.SEQ_LEN, self.FEATURES_DIM),
            return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(
            layers[2],
            return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(
            units=self.n_classes,
            activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
        return model

    def run_gen_model(self):
        """

        Returns:

        """

        start = time.time()
        model = self.lstm_model()

        self.logger.info("Entering generator")
        mygenerator = DataGenerator(file_path=self.file_path, BATCH_SIZE=self.BATCH_SIZE,
                                    SEQ_LEN=self.SEQ_LEN, n_classes=self.n_classes, FEATURES_DIM=self.FEATURES_DIM)

        # Check HDF for sample_count attribute
        self.logger.info("Checking HDF for sample_count attributes ...")
        mygenerator.check_HDF_attrs()

        # Prepare the callbacks
        history = model_callbacks.Histories()
        checkpointer = model_callbacks.Checkpoints()

        self.logger.info("Training on the Data")

        model.fit_generator(generator = mygenerator.generate_training_sequences(),
                            steps_per_epoch = mygenerator.training_sample_count()//self.BATCH_SIZE,
                            validation_data = mygenerator.generate_validation_sequences(),
                            validation_steps = mygenerator.validation_sample_count()//self.BATCH_SIZE,
                            epochs = self.EPOCHS, callbacks=[history],
                            verbose = 1,
                            workers = self.nb_workers)

        self.logger.info("> Compilation Time : {}".format(time.time() - start))

        self.logger.debug("Epoch Loss = {}".format(history.losses))
        self.logger.debug("Epoch Accuracy = {}".format(history.acc))

        # Prediction and Confusion matrix
        test_samples, test_labels = mygenerator.generate_test_sequences()
        predicted_vals = model.predict(test_samples, batch_size=self.BATCH_SIZE, verbose=1)

        model_metrics = Metrics(predicted_vals, test_labels)
        model_metrics.build_conf_matrix()