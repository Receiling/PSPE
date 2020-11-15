import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TokenEncoder(nn.Module):
    """This class encodes token information in multi-view,
    such as various word embeddings, char embedding and so on.
    """
    def __init__(self,
                 word_embedding,
                 char_embedding,
                 char_batch_size,
                 dropout=0.0,
                 aux_word_embeddings=None):
        """This function sets `TokenEncoder` model parameters

        Arguments:
            word_embedding {Module} -- word embedding model
            char_embedding {Module} -- char embedding model
            char_batch_size {int} -- char embedding batch size

        Keyword Arguments:
            dropout {float} -- dropout rate (default: {0.0})
            aux_word_embeddings {dict} -- aux word embedding dict (default: {None})
        """

        super().__init__()
        self.word_embedding = word_embedding
        self.char_embedding = char_embedding
        self.char_batch_size = char_batch_size

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

        self.aux_word_embeddings = dict(
            aux_word_embeddings) if aux_word_embeddings is not None else dict()

        for id, aux_word_embedding in self.aux_word_embeddings.items():
            self.add_module('aux_word_embedding_{}'.format(id), aux_word_embedding)

    def add_aux_word_embedding(self, aux_word_embeddings):
        """This function adds aux word embedding

        Arguments:
            aux_word_embeddings {dict} -- aux word embedding dict
        """

        for id, aux_word_embedding in aux_word_embeddings.items():
            if id in self.aux_word_embeddings:
                logger.error("duplicate aux_word_embedding id {}.".format(id))
                continue
            self.add_module('aux_word_embedding_{}'.format(id), aux_word_embedding)
            self.aux_word_embeddings[id] = aux_word_embedding

    def forward(self, seq_inputs, char_seq_inputs=None, aux_seq_inputs=None):
        """This function propagetes forwardly
        
        Arguments:
            seq_inputs {tensor} -- seq input data
        
        Keyword Arguments:
            char_seq_inputs {tensor} -- char seq input data (default: {None})
            aux_seq_inputs {dict} -- aux seq input data dict (default: {None})
        
        Returns:
            tensor -- token embeding in multi-view
        """

        seq_vecs = self.dropout(self.word_embedding(seq_inputs))
        token_vecs = [seq_vecs]

        if char_seq_inputs is not None:
            char_seq_vecs = []
            for id in range(0, char_seq_inputs.size(0), self.char_batch_size):
                batch_char_seq_inputs = char_seq_inputs[id:id + self.char_batch_size]
                char_seq_vecs.append(self.dropout(self.char_embedding(batch_char_seq_inputs)))
            char_seq_vecs = torch.cat(char_seq_vecs, dim=0)
            token_vecs.append(char_seq_vecs)

        if aux_seq_inputs is not None:
            for id, aux_seq_input in aux_seq_inputs.items():
                token_vecs.append(
                    self.dropout(getattr(self, 'aux_word_embedding_{}'.format(id))(aux_seq_input)))

        return torch.cat(token_vecs, dim=2)
