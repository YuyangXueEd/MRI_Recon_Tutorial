import os
import h5py
import numpy as np


def loadmat(filename, variable_names=None):
    """
    load mat file from h5py files
    :param filename: mat filename
    :param variable_names: list of variable names that should be loaded
    :return: dictionary of loaded data
    """
    data = {}

    matfile = h5py.File(filename, 'r')

    if variable_names is None:
        for key in matfile.keys():
            data.update({key: matfile[key][()]})
    else:
        for key in variable_names:
            if not key in matfile.keys():
                raise RuntimeError('Variable: "' + key + '" is not in file: ' + filename)
            data.update({key: matfile[key][()]})

    return data

if __name__ == '__main__':
    a = loadmat('file1000082_v2.h5')
    print(a)