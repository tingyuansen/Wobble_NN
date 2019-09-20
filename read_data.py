

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from json import JSONDecoder, JSONDecodeError  # for reading the JSON data files
import re  # for regular expressions
import os  # for os related operations

#We can list all files in this directory:
print(os.listdir("../input"))

#To be able to read the data in json format, we need to have a decoder as follows:
def decode_obj(line, pos=0, decoder=JSONDecoder()):
    no_white_space_regex = re.compile(r'[^\s]')
    while True:
        match = no_white_space_regex.search(line, pos)
        if not match:
            return
        pos = match.start()
        try:
            obj, pos = decoder.raw_decode(line, pos)
        except JSONDecodeError as err:
            print('Oops! something went wrong. Error: {}'.format(err))
        yield obj

def get_obj_with_last_n_val(line, n):
    obj = next(decode_obj(line))  # type:dict
    id = obj['id']
    class_label = obj['classNum']
    #
    data = pd.DataFrame.from_dict(obj['values'])  # type:pd.DataFrame
    data.set_index(data.index.astype(int), inplace=True)
    last_n_indices = np.arange(0, 60)[-n:]
    data = data.loc[last_n_indices]
    #
    return {'id': id, 'classType': class_label, 'values': data}
    

def convert_json_data_to_numpy(data_dir: str, file_name: str):
    """
    :param data_dir: the path to the data directory.
    :param file_name: name of the file to be read, with the extension.
    :return: the generated dataframe.
    """
    fname = os.path.join(data_dir, file_name)
    #
    all_df, labels, ids = [], [], []
    with open(fname, 'r') as infile: # Open the file for reading
        for line in infile:  # Each 'line' is one MVTS with its single label (0 or 1).
            obj = get_obj_with_last_n_val(line, 60)
            all_df.append(obj['values'])
            labels.append(obj['classType'])
            ids.append(obj['id'])
    #
    data = np.stack(all_df, axis=-1)
    data = np.einsum('ijk->kij', A)
    labels = np.asarray(labels)
    ids = np.asarray(ids)
    return data, labels, ids


path_to_data = "../input/"
file_name = "fold3Training.json"
data, labels, ids = convert_json_data_to_numpy(path_to_data, file_name)  # shape: 27006 X 60 X 25
print('df.shape = {}'.format(data.shape))
