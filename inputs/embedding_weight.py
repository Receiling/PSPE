import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_embeddimg_weight(vocab, namespace, pretrained_vec, embedding_dims):
    """This function returns embedding weight from pretrained embedding vector

    Arguments:
        vocab {Vocabulary} -- vocabulary
        namespace {str} -- namespace
        pretrained_vec {dict} -- pretrained embedding vector
        embedding_dims {int} -- embedding dimension

    Returns:
        np.array -- weight
    """

    weight = np.random.uniform(-0.25, 0.25, (vocab.get_vocab_size(namespace), embedding_dims))
    padding_idx = vocab.get_padding_index(namespace)
    unknown_idx = vocab.get_unknown_index(namespace)
    weight[padding_idx, :] = 0.0

    total = 0
    found = 0
    for word, vec in pretrained_vec.items():
        total += 1
        idx = vocab.get_token_index(word, namespace)
        if idx != unknown_idx:
            found += 1
            weight[idx, :] = np.array(vec)

    logger.info("Found {} [{}%] words in pretrained embedding with {} words".format(
        found, 100.0 * found / total, total))

    return weight
