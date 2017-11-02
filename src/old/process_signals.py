import numpy as np
import pandas as pd
import logging
from biosppy.signals import resp as rr
from biosppy.signals import ecg as hr


class ExtractSignals(object):
    def __init__(self, epoch_piezo_data):
        self.piezo_data = np.array(epoch_piezo_data['PIEZO'])
        self.sample_rate = int(len(epoch_piezo_data) / 30)


class ExtractHeartSignal(ExtractSignals):
    def __init__(self, epoch_piezo_data):
        super(ExtractHeartSignal, self).__init__(epoch_piezo_data)
        self.filtered_heart_signal = []

    def _get_heart_signal_from_biosppy(self, graph_option=0):

        sample_rate = self.sample_rate
        epoch_piezo_data = self.piezo_data

        logging.debug("Sample Rate : {}".format(self.sample_rate))

        if graph_option == 0:
            try:
                self.filtered_heart_signal = hr.ecg(epoch_piezo_data, sample_rate, False)
            except:
                self.filtered_heart_signal = -1

        else:
            try:
                self.filtered_heart_signal = hr.ecg(epoch_piezo_data, sample_rate, True)

            except:
                self.filtered_heart_signal = -1

    def get_heart_signal(self):
        self._get_heart_signal_from_biosppy()

        logging.info("Length of Heart Signal : {}".format(len(self.filtered_heart_signal[1])))
        try:
            print len(self.filtered_heart_signal[1])
            return self.filtered_heart_signal[1]
        except TypeError:
            return [[-1000]]

    def get_heart_rate(self):
        self._get_heart_signal_from_biosppy()
        return self.filtered_heart_signal[-1]


class ExtractRespSignal(ExtractSignals):
    def __init__(self, epoch_piezo_data):
        super(ExtractRespSignal, self).__init__(epoch_piezo_data)
        self.filtered_resp_signal = []

    def _get_respiratory_signal_from_biosppy(self, graph_option=0):

        sample_rate = self.sample_rate
        epoch_piezo_data = self.piezo_data

        epoch_piezo_rolling_mean = pd.rolling_mean(epoch_piezo_data, 1)
        # print epoch_piezo_rolling_mean

        epoch_piezo_rolling_mean_clean = []
        for ind in range(len(epoch_piezo_rolling_mean)):
            if (np.isnan(epoch_piezo_rolling_mean[ind]) != 1):
                epoch_piezo_rolling_mean_clean.append(epoch_piezo_rolling_mean[ind])

        if graph_option == 0:
            try:
                self.filtered_resp_signal = rr.resp(epoch_piezo_rolling_mean_clean, sample_rate, False)
            except Exception as e:
                logging.warning(e)
                self.filtered_resp_signal = -1
        else:
            try:
                self.filtered_resp_signal = rr.resp(epoch_piezo_rolling_mean_clean, sample_rate, True)
            except Exception as e:
                logging.warning(e)
                self.filtered_resp_signal = -1

    def get_respiratory_signal(self):
        self._get_respiratory_signal_from_biosppy()
        try:
            return self.filtered_resp_signal[1], self.sample_rate
        except:
            return self.filtered_resp_signal, self.sample_rate

    def get_respiratory_rate(self):
        self._get_respiratory_signal_from_biosppy()
        return self.filtered_resp_signal[-1]
