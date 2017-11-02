import os
from os import path
import sys
import glob
import logging

# from memory_profiler import profile
# from guppy import hpy

from process_file import ConvertFromAscii
from analyze import GetSignalData

logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, filename='logs/')

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# h = hpy()


# fp = open('logs/memory_profile.log', 'w')
# @profile(stream=fp)
def read_dataset(file_name):
    os.chdir('dataset/' + file_name)
    logging.debug("CWD: {}".format(os.system('pwd')))

    file_paths = []
    for path_names in glob.glob("*"):
        file_paths.append(path_names)

    # write_data = DataWriter()

    file_paths.sort()
    # h.setref()
    for count, names in enumerate(file_paths):
        logging.debug("File Name : {}".format(str(names)))

        process_file_object = ConvertFromAscii()
        process_file_object.ascii_reader(names)
        # logging.debug(process_file_object.get_full_data())
        piezo_data = process_file_object.get_piezo_data()

        process_epoch_obj = GetSignalData(piezo_data)
        # process_epoch_obj.get_filtered_heart_signal(count, write_file=True)
        # process_epoch_obj.get_filtered_resp_signal(count, write_file=True)
        process_epoch_obj.get_combined_filtered_signal(count, write_file=True)

        # write_data.csv_writer(data=piezo_data, file_name=file_name)

# fop = open('logs/heappy_mem_usage.log', 'w')
# fop.write(str(h.heap()))
# fop.close()


if __name__ == '__main__':
    read_dataset('10012017')
