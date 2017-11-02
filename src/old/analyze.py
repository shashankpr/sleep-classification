import logging
import pandas as pd
import numpy as np
import itertools

from process_signals import ExtractHeartSignal
from process_signals import ExtractRespSignal

from data_writer import DataWriter

logging.getLogger(__name__)
logging.basicConfig(filename="logs/epochs.log", filemode='w', level=logging.DEBUG)


class ProcessPiezoData(object):
    def __init__(self, piezo_data):
        self.piezo_data = piezo_data
        self.epochs = 30
        self.epoch_freq = str(self.epochs) + 'S'

        self.time_list = [x['TIME'] for x in piezo_data]

    def _convert_to_epochs(self):
        first_timestamp = pd.to_datetime(self.time_list[0])
        last_timestamp = pd.to_datetime(self.time_list[-1])

        logging.info("First Timestamp = {}".format(first_timestamp))
        logging.info("Last Timestamp = {}".format(last_timestamp))

        epoch_range = pd.date_range(start=first_timestamp, end=last_timestamp, freq=self.epoch_freq)
        logging.debug("Epoch range = {}".format(len(epoch_range)))

        return epoch_range

        # The below code is Pythonic way to get 30s epoch data.
        # epoch_dict = {}

        # for ind in range(0, len(epoch_range)-1):
        #     epoch_list = []
        #     epoch_start_time = epoch_range[0]
        #     epoch_end_time = epoch_range[1]
        #     logging.debug("Start time : {}".format(epoch_start_time))
        #     logging.debug("End Time : {}".format(epoch_end_time))
        #
        #     for data in self.piezo_data:
        #         if epoch_start_time <= data['TIME'] < epoch_end_time:
        #             # epoch_list.append(data['TIME'])
        #             # logging.info(epoch_list)
        #             # logging.debug("Timestamp : {}".format(data['TIME']))
        #             epoch_list.append(data)
        #         else:
        #             if data['TIME'] == epoch_end_time:
        #                 epoch_list.append(data)
        #
        #             epoch_dict[str(ind)] = epoch_list
        #             ind = ind + 1
        #             epoch_start_time = epoch_range[ind]
        #             epoch_end_time = epoch_range[ind+1]
        #             logging.debug("New Start time : {}".format(epoch_start_time))
        #             logging.debug("New End time : {}".format(epoch_end_time))
        #             epoch_list = []
        #
        #     break
        #
        # logging.info(epoch_dict)
        # return epoch_dict

    def _get_epoch_data(self, epoch_range, epoch_index):
        self.piezo_data = pd.DataFrame(self.piezo_data)
        time_frame = self.piezo_data['TIME']

        try:
            piezo_epoch_data = self.piezo_data[(epoch_range[epoch_index] <= time_frame) & (time_frame < epoch_range[epoch_index + 1])]
        except:
            # print self.time_list[-1]
            piezo_epoch_data = self.piezo_data[(epoch_range[epoch_index] <= time_frame) & (time_frame <= self.time_list[-1])]

        logging.info("EPOCH {}".format(epoch_index))
        logging.debug(piezo_epoch_data)

        return piezo_epoch_data


class GetSignalData(ProcessPiezoData):
    def __init__(self, piezo_data):
        super(GetSignalData, self).__init__(piezo_data)
        self.piezo_dict = {}
        self.heart_dict = {}
        self.resp_dict = {}
        self.feature_dict = {}

    def get_epoch_piezo_signal(self):
        """
        Get the whole Raw piezo data in a 30 sec epoch interval.
        :return: 
        """

        print "Getting Raw Epoch Piezo Signal"

        epoch_range = self._convert_to_epochs()

        for ind in range(len(epoch_range)):
            epoch_piezo_data = self._get_epoch_data(epoch_range, ind)

    def _extract_heart_signal(self):
        """
        Get filtered heart signal in 30sec epoch interval.
        :return: dict of timestamps and filtered signal
        """
        print "Heart Signal here"

        epoch_range = self._convert_to_epochs()

        timestamp = []
        # timestamp = np.array(timestamp)
        heart_signal = []
        heart_signal = np.array(heart_signal)

        for ind in range(len(epoch_range)):
            epoch_piezo_data = self._get_epoch_data(epoch_range, ind)

            ts = epoch_piezo_data['TIME']
            # print ts
            heart_class_obj = ExtractHeartSignal(epoch_piezo_data)
            filtered_heart_signal = heart_class_obj.get_heart_signal()

            # timestamp = np.concatenate((timestamp, ts))
            timestamp.append(ts)
            heart_signal = np.concatenate((heart_signal, filtered_heart_signal))

            logging.info("Heart Signal Extracted for Epoch {}".format(ind))

        self.heart_dict['TIME'] = list(itertools.chain.from_iterable(timestamp))  # Making a flat list - fast
        self.heart_dict['HSIGNAL'] = heart_signal
        # logging.debug("TIME = {}, Heart = {}".format(len(timestamp), len(heart_signal)))

    def _extract_respiratory_signal(self):
        """
        Get respiratory signal in 30sec epoch interval
        :return: dict of timestamps and filtered signal
        """
        print "Resp Signal here"

        epoch_range = self._convert_to_epochs()

        # timestamp = np.array([])
        timestamp_resp = []
        resp_signal = []
        resp_signal = np.array(resp_signal)

        for ind in range(len(epoch_range)):
            epoch_piezo_data = self._get_epoch_data(epoch_range, ind)

            # ts = np.asarray(epoch_piezo_data['TIME'])
            ts = epoch_piezo_data['TIME']

            resp_class_obj = ExtractRespSignal(epoch_piezo_data)
            filtered_resp_signal, sample_rate = resp_class_obj.get_respiratory_signal()

            # timestamp = np.concatenate((timestamp, ts))
            # timestamp_resp.append(ts[(sample_rate/2) - 1:])
            timestamp_resp.append(ts)
            try:
                resp_signal = np.concatenate((resp_signal, filtered_resp_signal))
            except Exception as e:
                logging.warning(e)
                repeat = np.ones(sample_rate*self.epochs)
                filtered_resp_signal = repeat * np.asarray(filtered_resp_signal)
                resp_signal = np.concatenate((resp_signal, filtered_resp_signal))
            logging.info("Resp Signal Extracted for Epoch {}".format(ind))

        self.resp_dict['TIME'] = list(itertools.chain.from_iterable(timestamp_resp))
        self.resp_dict['RSIGNAL'] = resp_signal
        logging.debug("Time = {}, Resp = {} ".format(len(self.resp_dict['TIME']), len(resp_signal)))

    def get_filtered_heart_signal(self, file_count, write_file=False):
        self._extract_heart_signal()

        logging.debug("Length of HS : {}".format(len(self.heart_dict['HSIGNAL'])))

        if write_file:
            write_to_csv = DataWriter()
            write_to_csv.csv_writer(self.heart_dict, file_count, "features/heartS")

        return self.heart_dict

    def get_filtered_resp_signal(self, file_count, write_file=False):
        self._extract_respiratory_signal()

        logging.debug("Length of RS : {}".format(len(self.resp_dict['RSIGNAL'])))
        if write_file:
            write_to_csv = DataWriter()
            write_to_csv.csv_writer(self.resp_dict, file_count, "features/respS")

        return self.resp_dict

    def get_combined_filtered_signal(self, file_count, write_file=False):
        heart_dictionary = self.get_filtered_heart_signal(file_count, write_file=False)
        resp_dictionary = self.get_filtered_resp_signal(file_count, write_file=False)
        self.feature_dict['TIME'] = heart_dictionary['TIME']
        self.feature_dict['HSIGNAL'] = heart_dictionary['HSIGNAL']
        self.feature_dict['RSIGNAL'] = resp_dictionary['RSIGNAL']

        logging.debug("Length of RS : {}, Length of HS : {}, Length of Time : {}".format(len(self.resp_dict['RSIGNAL']),
                                                                                         len(self.heart_dict['HSIGNAL']),
                                                                                         len(self.resp_dict['TIME'])))
        if write_file:
            write_to_csv = DataWriter()
            write_to_csv.csv_writer(self.feature_dict, file_count, "features/feature")

        return self.feature_dict
