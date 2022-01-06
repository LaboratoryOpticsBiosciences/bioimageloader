from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from PIL import Image


def imread_array(p: Path, dtype=None) -> np.ndarray:
    '''Read an image using PIL then convert it into numpy array'''
    img = np.asarray(Image.open(p), dtype=dtype)
    return img


def rle_decoding_inseg(
    size: Union[tuple, list],
    run_lengths: Union[pd.DataFrame, list]
) -> np.ndarray:
    """Decoding RLE (Run Length Encoding). Output binary mask. If you want each
    instance have different values, use `rle_decoding_inseg(), instead.`

    Parameters
    ----------
    size : list or tuple
        Shape of the original image array (height, width)
    run_lengths : list
        List of run length encodings
    val : int or float
        Constant value for all encoded pixels

    Returns
    -------
    decoded : numpy.ndarray
        Decoded image array

    """
    # #--- Prep ---# #
    # if `run_lengths` is pd.Dataframe, convert it to a list of numpy.ndarray
    if isinstance(run_lengths, pd.DataFrame):
        _run_lengths = []
        for e in run_lengths['EncodedPixels']:
            if not isinstance(e, np.ndarray):
                _run_lengths.append(np.array(e.split(sep=' '), dtype=int))
            else:
                _run_lengths.append(e)
        run_lengths = _run_lengths

    # #--- Draw canvas ---# #
    h, w = size[0], size[1]
    decoded = np.zeros(h * w, dtype=np.uint8)
    # num_objs = len(run_lengths)

    for i, rle in enumerate(run_lengths):
        for p, l in zip(rle[0::2], rle[1::2]):
            for dp in range(l):
                decoded[(p-1)+dp] = i + 1  # 0 is bg
    return decoded.reshape((w, h)).T
