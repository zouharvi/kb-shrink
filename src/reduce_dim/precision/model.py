import numpy as np


def _transform_to_16(x):
    return np.array(x, dtype=np.float16)


def _transform_to_8(x):
    arr16 = np.array(x, dtype=np.float16)
    arr8 = arr16.tobytes()[1::2]
    # back to float 16
    return np.frombuffer(
        np.array(np.frombuffer(arr8, dtype='u1'), dtype='>u2').tobytes(),
        dtype=np.float16
    )

def _transform_to_1(x, offset):
    return (np.array(x) > 0)*1.0 + offset
    # return [1 if el > 0 else -1 for el in x]

def transform_to_16(array):
    return [_transform_to_16(x) for x in array]

def transform_to_8(array):
    return [_transform_to_8(x) for x in array]

def transform_to_1(array, offset=-0.5):
    return [_transform_to_1(x, offset) for x in array]

def transform_to_32(array):
    return np.array(array, dtype=np.float32)