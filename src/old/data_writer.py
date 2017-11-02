import pandas as pd
import logging


class DataWriter(object):

    def csv_writer(self, data, file_count, file_name):
        if file_count == 0:
            pd.DataFrame(data).to_csv(file_name + '.csv', mode='a', index=False)
        else:
            pd.DataFrame(data).to_csv(file_name + '.csv', mode='a', index=False, header=None)
