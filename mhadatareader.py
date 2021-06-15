from os import listdir
import numpy as np
import utils as ut
import classes as cs

RECORD_STATUS_MSG = 'status'
PROB_TO_TRACKER_REC = 'probetotracker'
REF_TO_TRACKER_REC = 'referencetotracker'
TIMESTAMP_REC = 'timestamp'


def parse_timestamp_line(line):
    """
        Parses lines with timestamp and returns timestamp
        Example: Seq_Frame0000_Timestamp = 8151.50942857143\n'
    """

    timestamp_str = ut.remove_spec_char(line.split(' = ')[1])
    return float(timestamp_str)


def parse_transform(line):
    """
        Function parses the line and returns a transformation matrix as a 4x4 array
    """
    # matrix dimension
    dim = 4
    # split the line name from its data and leave only numbers
    transform = ut.remove_spec_char(line.split('=')[1])

    # split numbers to array
    split = transform.split(' ')
    to_array = np.zeros((dim, dim))

    row_count = 0
    while len(split) != 0:
        row = split[:dim]
        for i in range(len(row)):
            row[i] = float(row[i])

        to_array[row_count] = np.array([row])
        split = split[dim:]

        row_count = row_count + 1

    return to_array


def parse_file_name(file_name):
    """
        Parses file name and returns participant and scanned region
    """
    fn = file_name[file_name.rfind('/') + 1:]
    fn = fn.split('_')
    return fn[1], fn[2].split('.')[0]


class MhaDataReader:
    """
        Parses transformation data by proficiency from all mha files in a directory
    """

    def __init__(self):
        self.__end_of_file = 'ElementDataFile = LOCAL'
        self.__transform_line = 'Transform'
        self.__extension = '.mha'

    def read_data(self, dir_name):
        """
            Function parses mha data file and returns a dictionary
            with all transformations as numpy matrices
        """

        assert dir_name is not None and str(dir_name).strip() != ""

        data = self.__read_dir(dir_name)

        return data

    def __read_dir(self, dir_name):
        """
            Reads all mha files in the dictionary
        """
        data = cs.ParticipantsData()

        # filter out non mha files
        files = [f'{dir_name}{fn}' for fn in listdir(dir_name) if str(fn[-4:]).lower() == self.__extension]
        files.sort()

        for file_name in files:
            part_nm, scan_nm = parse_file_name(file_name)
            scan_nm = cs.Scan(scan_nm)
            if part_nm not in data:
                data.add_participant(cs.ParticipantScan(part_nm))

            time_for_region = self.__parse_region(file_name, data[part_nm], scan_nm)
            data[part_nm].add_reg_time(scan_nm, time_for_region)
            data[part_nm].add_time(time_for_region)

        return data

    def __parse_region(self, file_name: str, part: cs.ParticipantScan, scan_nm: cs.Scan) -> float:
        """
            Function reads the file for a scanned region, adds transformations of each type
            to the corresponding key in the dictionary

            Returns: time spent scanning the region
        """
        prob_to_tracker = np.zeros((4, 4))
        ref_to_tracker = np.zeros((4, 4))
        prev_prob_to_tracker = np.zeros((4, 4))
        prev_ref_to_tracker = np.zeros((4, 4))
        time_delta = 0
        prev_time_delta = 0
        prev_time = 0
        with open(file_name, 'rb') as f:
            for line in f:
                line = str(line)
                # The end of transformations block
                if self.__end_of_file in line:
                    break

                if RECORD_STATUS_MSG in line.lower():
                    continue

                if PROB_TO_TRACKER_REC in line.lower():
                    prob_to_tracker = parse_transform(line)
                elif REF_TO_TRACKER_REC in line.lower():
                    ref_to_tracker = parse_transform(line)
                elif TIMESTAMP_REC in line.lower():
                    if len(part.get_transforms(scan_nm)) == 0:
                        prev_time = parse_timestamp_line(line)

                    timestamp = parse_timestamp_line(line)
                    time_delta = timestamp - prev_time

                    # skip identical transformations with same time stamps
                    if np.allclose(prev_ref_to_tracker, ref_to_tracker) \
                            and np.allclose(prev_prob_to_tracker, prob_to_tracker) \
                            and time_delta == prev_time_delta:
                        continue

                    # computing reference-to-tracker transformations
                    ref_to_tracker_inv = np.linalg.inv(ref_to_tracker)
                    prob_to_ref = ref_to_tracker_inv.dot(prob_to_tracker)

                    rec = cs.TransformationRecord(prob_to_ref, time_delta)
                    part.add_transform(scan_nm, rec)
                    part.add_transform(cs.Scan.ALL, rec)

                    # update previous transformations
                    prev_prob_to_tracker = prob_to_tracker
                    prev_ref_to_tracker = ref_to_tracker
                    prev_time_delta = time_delta

            f.close()

        return time_delta
