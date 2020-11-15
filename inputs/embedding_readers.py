import os
import gzip
import logging

import numpy as np

logger = logging.getLogger(__name__)


def glove_reader(file_path, embedding_dim):
    """This function returns glove embedding dict from glove file

    Arguments:
        file_path {str} -- glove file path
        embedding_dim {int} -- embedding dimension

    Raises:
        ValueError: glove file doesn't match glove file

    Returns:
        dict -- glove embedding dict
    """

    if not os.path.exists(file_path):
        logger.error("golve file {} not exits.".format(file_path))
        raise ValueError("golve file {} not exits.".format(file_path))

    glove = {}

    if file_path[-3:] == '.gz':
        fin = gzip.open(file_path, 'rt')
    else:
        fin = open(file_path, 'r')

    for line in fin:
        words = line.strip().split()
        assert len(words) == embedding_dim + 1, "the dim of word `{}` is not correct".format(
            words[0])
        glove[words[0]] = list(map(float, words[1:]))

    fin.close()

    logger.info("glove size: {}.".format(len(glove)))
    return glove
