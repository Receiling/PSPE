import torch.nn as nn


class Embedding(nn.Module):
    """This class is embedding model using pre-trained feature-based word embeddings.
    """
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 padding_idx,
                 weight=None,
                 fix_embedding=False,
                 dropout=0.0):
        """This function sets `Embedding` model parameters

        Arguments:
            vocab_size {int} -- the size of vacabulary
            embedding_dim {int} -- embedding dim
            padding_idx {int} -- the index of padding token

        Keyword Arguments:
            weight {tensor} -- pretrained weight (default: {None})
            fix_embedding {bool} -- fix weight or not (default: {False})
            dropout {float} -- dropout rate (default: {0.0})
        """

        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_idx,
                                      _weight=weight)
        if fix_embedding:
            self.embedding.weight.requires_grad = False

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

    def get_vocab_size(self):
        return self.vocab_size

    def get_embedding_dim(self):
        return self.embedding_dim

    def get_padding_idx(self):
        return self.padding_idx

    def forward(self, inputs):
        """This function propagetes forwardly

        Arguments:
            inputs {tensor} -- input data

        Returns:
            tensor -- output after embedding
        """

        return self.dropout(self.embedding(inputs))
