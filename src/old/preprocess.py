import os
import pandas as pd
import numpy as np

from sklearn import preprocessing


class Preprocess(object):
    def __init__(self, file_name):
        self.heart_scaled = {}
        self.resp_scaled  = {}
        self.file_name = file_name
        os.chdir('dataset/' + file_name)

    def _read_processed_file(self):
        heart_data = pd.read_csv('heartS.csv')
        resp_data = pd.read_csv('respS.csv')

        return heart_data, resp_data

    def scale_data(self):
        heart_data, resp_data = self._read_processed_file()
        heart_val_scaled = preprocessing.scale(heart_data['HSIGNAL'])
        resp_val_scaled = preprocessing.scale(resp_data['RSIGNAL'])

        self.heart_scaled['HSIGNAL'] = heart_val_scaled
        self.heart_scaled['TIME'] = heart_data['TIME']

        self.resp_scaled['RSIGNAL'] = resp_val_scaled
        self.resp_scaled['TIME'] = resp_data['TIME']

        pd.DataFrame(self.heart_scaled).to_csv('heartS_scaled.csv', index=False)
        pd.DataFrame(self.resp_scaled).to_csv('respS_scaled.csv', index=False)

        # print np.mean(heart_val_scaled), np.mean(resp_val_scaled)
        # print np.std(heart_val_scaled), np.std(resp_val_scaled)

        return heart_val_scaled, resp_val_scaled


if __name__ == '__main__':
    filename = '10012017'
    p = Preprocess(filename)
    p.scale_data()
