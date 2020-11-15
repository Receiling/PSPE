import logging

import torch
import torch.nn as nn

from inputs.embedding_readers import glove_reader
from inputs.embedding_weight import load_embeddimg_weight
from modules.token_embedders.token_encoder import TokenEncoder
from modules.token_embedders.embedding import Embedding
from modules.token_embedders.char_token_encoder import CharTokenEncoder
from modules.seq2vec_encoders.cnn_encoder import CNNEncoder

logger = logging.getLogger(__name__)


class WordCharEmbedModel(nn.Module):
    """This class acts as an embedding layer
    which contanins word embedding and cnn char embedding.
    """
    def __init__(self, cfg, vocab):
        """This function constructs `WordCharEmbedModel` components and
        sets `WordCharEmbedModel` parameters

        Arguments:
            cfg {dict} -- config parameters for constructing multiple models
            vocab {Vocabulary} -- vocabulary
        """

        super().__init__()
        self.vocab = vocab

        glove = glove_reader(cfg.pretrained_embeddings_file, cfg.embedding_dims)
        weight = load_embeddimg_weight(vocab, 'tokens', glove, cfg.embedding_dims)
        weight = torch.from_numpy(weight).float()
        logger.info(
            "`tokens` size: {}, `glove` size: {}, intersetion `tokens` and `glove` of size: {}".
            format(vocab.get_vocab_size('tokens'), len(glove),
                   len(set(glove)
                       & set(vocab.get_namespace_tokens('tokens')))))

        self.word_embedding = Embedding(vocab_size=self.vocab.get_vocab_size('tokens'),
                                        embedding_dim=cfg.embedding_dims,
                                        padding_idx=self.vocab.get_padding_index('tokens'),
                                        weight=weight,
                                        dropout=cfg.dropout)
        self.char_embedding = Embedding(vocab_size=self.vocab.get_vocab_size('char_tokens'),
                                        embedding_dim=cfg.char_dims,
                                        padding_idx=self.vocab.get_padding_index('char_tokens'),
                                        dropout=cfg.dropout)
        self.char_encoder = CNNEncoder(input_size=cfg.char_dims,
                                       num_filters=cfg.char_output_channels,
                                       ngram_filter_sizes=cfg.char_kernel_sizes,
                                       dropout=cfg.dropout)
        char_token_encoder = CharTokenEncoder(char_embedding=self.char_embedding,
                                              encoder=self.char_encoder)

        self.token_encoder = TokenEncoder(word_embedding=self.word_embedding,
                                          char_embedding=char_token_encoder,
                                          char_batch_size=cfg.char_batch_size)

        self.embedding_dims = cfg.word_dims + self.char_encoder.get_output_dims()

    def forward(self, batch_inputs):
        """This function propagetes forwardly

        Arguments:
            batch_inputs {dict} -- batch input data
        """

        batch_inputs['seq_encoder_reprs'] = self.token_encoder(batch_inputs['tokens'],
                                                               batch_inputs['char_tokens'])

    def get_hidden_size(self):
        """This function returns embedding dimensions
        
        Returns:
            int -- embedding dimensitons
        """

        return self.embedding_dims
