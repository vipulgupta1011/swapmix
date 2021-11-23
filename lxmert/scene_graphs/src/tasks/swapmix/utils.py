import json
import numpy as np

def load_json(fname) :
    json_dict = json.load(open(fname))

    return json_dict

def np_load(fname) :
    np_array = np.load(fname)

    return np_array
