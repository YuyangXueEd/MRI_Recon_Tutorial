import os
import h5py
import numpy as np

def judge_key(matfile, key):
    data = {}
    if isinstance(matfile[key], str) or \
            isinstance(matfile[key], list) or \
            isinstance(matfile[key], dict) or \
            key == '__header__' or key == '__globals__' or key == '__version__':
        data.update({key: matfile[key]})
    elif matfile[key].dtype.names is not None and 'imag' in matfile[key].dtype.names:
        data.update({key: np.transpose(np.asarray(matfile[key].value.view(np.complex), dtype='complex128'))})
    else:
        data.update({key: np.transpose(np.asarray(matfile[key].value, dtype=matfile[key].dtype))})

    return data


def loadmat(filename, variable_names=None):
    """
    load mat file from h5py files
    :param filename: mat filename
    :param variable_names: list of variable names that should be loaded
    :return: dictionary of loaded data
    """
    data = {}

    matfile = h5py.File(filename, 'r')

    if variable_names == None:
        for key in matfile.keys():
            data.update(judge_key(matfile, None))
    else:
        for key in variable_names:
            if not key in matfile.keys():
                raise RuntimeError('Variable: "' + key + '" is not in file: ' + filename)
            data.update(judge_key(matfile, variable_names))

    return data


if __name__ == '__main__':
    a = loadmat("FastMRI/Dataset/multicoil_test_v2/file1001700_v2.h5")
    print(a)