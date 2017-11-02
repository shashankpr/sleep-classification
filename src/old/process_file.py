import logging
import base64
import datetime as dt


logging.basicConfig(filemode='w', filename='logs/process.log', level=logging.DEBUG)


class DataReader(object):
    def __init__(self):
        self.userid = 0
        self.deviceid = 0
        self.header_timestamp = 0
        self.data = []

    def get_piezo_data(self):
        # logging.debug(self.data)

        piezo_data = []
        for lists in self.data:
            piezo_data = lists["PIEZODATA"]

        return piezo_data

    def get_conditions_data(self):

        conditions_data = []
        for lists in self.data:
            conditions_data = lists["CONDITIONS"]

        return conditions_data

    def get_full_data(self):
        return self.data


class ConvertFromAscii(DataReader):
    def __init__(self):
        super(ConvertFromAscii, self).__init__()

        self.numberofHeaders = 0
        self.time_code = 248
        self.dht_code = 249
        self.light_code = 242

    def ascii_reader(self, filename):
        ascii_file = open(filename, 'r')
        raw_lines = ascii_file.readlines()
        raw_sensor_string = ''

        for lines in raw_lines:
            raw_sensor_string = raw_sensor_string + lines

        encoded_string = base64.b64decode(raw_sensor_string)

        headerpos = [i for i, ltr in enumerate(encoded_string) if ord(ltr) == 252]

        self.numberofHeaders = len(headerpos)

        for i in range(len(headerpos)):
            sensordata = []
            dataforonefile = {}
            conditionslist = []
            piezolist = []

            if i == len(headerpos) - 1:
                newencoded_string = encoded_string[headerpos[i]:len(encoded_string)]
            else:
                newencoded_string = encoded_string[headerpos[i]:headerpos[i + 1]]

            for item in range(0, len(newencoded_string)):
                if item > 27:
                    sensordata.append(ord(newencoded_string[item]))

            # store the header
            dataforonefile['HEADER'] = encoded_string[headerpos[i]:headerpos[i] + 28]
            # hardcoded for now, change later
            utc_header_time = newencoded_string[1:11]
            # we lose the last digit in the millisecond here, check this later
            utc_header_time_milli = (float(utc_header_time) / 1.0)
            absolute_header_time = dt.datetime.fromtimestamp(utc_header_time_milli).strftime('%Y-%m-%d %H:%M:%S.%f')
            absolute_header_time_obj = dt.datetime.strptime(absolute_header_time, '%Y-%m-%d %H:%M:%S.%f')
            absolute_header_time_obj = absolute_header_time_obj
            # print absolute_header_time_obj
            # print 'Absolute Header'
            flag_nextcharacter_time = 0
            flag_nextcharacter_ldr = 0
            flag_nextcharacter_dht = 0
            flag_nextcharacter_piezo = 0
            one_sec_piezo_list = []
            relative_time_stamp = 0
            temperature = -1
            ldr = -1
            humidity = -1

            for decoded_value in sensordata:

                if decoded_value == self.time_code:

                    flag_nextcharacter_time = 1
                    if len(one_sec_piezo_list) != 0:

                        sample_rate = len(one_sec_piezo_list)
                        for piezo_itr in range(0, len(one_sec_piezo_list)):
                            piezo_time_secs = relative_time_stamp + piezo_itr * float(1.0 / sample_rate)
                            piezo_time_obj = absolute_header_time_obj + dt.timedelta(seconds=piezo_time_secs)

                            piezo_dict = {}
                            piezo_dict['TIME'] = piezo_time_obj
                            piezo_dict['PIEZO'] = one_sec_piezo_list[piezo_itr]
                            piezo_dict['PIEZO_ID'] = 207
                            # piezo_dict['PIEZO_ID'] = one_sec_piezo_code_list[piezo_itr]
                            piezolist.append(piezo_dict)
                        one_sec_piezo_list = []

                elif decoded_value == 240:
                    flag_nextcharacter_piezo = 1

                elif flag_nextcharacter_piezo == 1:
                    flag_nextcharacter_piezo = 0

                elif decoded_value == self.light_code:
                    flag_nextcharacter_ldr = 1

                elif decoded_value == self.dht_code:
                    flag_nextcharacter_dht = 2

                elif flag_nextcharacter_ldr == 1:

                    ldr = decoded_value
                    flag_nextcharacter_ldr = 0

                elif flag_nextcharacter_dht == 2:

                    temperature = decoded_value
                    flag_nextcharacter_dht = 1

                elif flag_nextcharacter_dht == 1:

                    humidity = decoded_value
                    flag_nextcharacter_dht = 0

                elif flag_nextcharacter_time == 1:
                    relative_time_stamp = decoded_value
                    flag_nextcharacter_time = 0
                else:
                    one_sec_piezo_list.append(decoded_value)

            if len(one_sec_piezo_list) != 0:

                sample_rate = len(one_sec_piezo_list)
                for piezo_itr in range(0, len(one_sec_piezo_list)):
                    piezo_time_secs = relative_time_stamp + piezo_itr * float(1.0 / sample_rate)
                    piezo_time_obj = absolute_header_time_obj + dt.timedelta(seconds=piezo_time_secs)

                    piezo_dict = {}
                    piezo_dict['TIME'] = piezo_time_obj
                    piezo_dict['PIEZO'] = one_sec_piezo_list[piezo_itr]
                    piezo_dict['PIEZO_ID'] = 207
                    piezolist.append(piezo_dict)

            conditions_dict = {}
            conditions_dict['TIME'] = absolute_header_time_obj
            conditions_dict['TEMPERATURE'] = temperature
            conditions_dict['HUMIDITY'] = humidity
            conditions_dict['LIGHT'] = ldr

            conditionslist.append(conditions_dict)

            dataforonefile["PIEZODATA"] = piezolist
            dataforonefile["CONDITIONS"] = conditionslist
            dataforonefile["HEADERTIME"] = absolute_header_time_obj

            self.data.append(dataforonefile)

            # There are special characters, timestamps and piezo values
            # 1,2,3,4,5,6,7 are designated special characters and they signify that
            # the next values till a special character again comes would belong to
            # the piezo numbered same as the special character. Similarly the special
            # character 9 would signify that the next character is the relative time stamp.

            # The following code will check each character
            # If the character is speacial, do nothing
            # If the character itself is not special and the previous one is store accordingly
            # If the character and the previous one is also not special then it is piezo value
            # self.piezoDictionaryList = piezoDictionaryListUnstructured
            # self.structurePiezoDictFromAsciiReader(startTimeStamp)
