import torch.nn as nn


class CharTokenEncoder(nn.Module):
    """This calss is token's character-level encoder.
    """
    def __init__(self, char_embedding, encoder, dropout=0.0):
        """This function sets `CharTokenEncoder` parameters

        Arguments:
            char_embedding {Module} -- char embedding
            encoder {Module} -- encoder

        Keyword Arguments:
            dropout {float} -- dropout rate (default: {0.0})
        """

        super().__init__()
        self.char_embedding = char_embedding
        self.encoder = encoder

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

    def forward(self, inputs):
        """This function propagates forwardly

        Arguments:
            inputs {tensor} -- input data

        Returns:
            tensor -- output after CharTokenEncoder
        """

        batch_size, sent_size, char_seq_size = inputs.size()

        # (batch_size * sent_size, char_seq_size)
        batchsent_char_inputs = inputs.view(-1, char_seq_size)

        # (batch_size * sent_size, char_seq_size, char_dim)
        batchsent_char_embeding = self.dropout(self.char_embedding(batchsent_char_inputs))

        # (batch_size, sent_size, output_size)
        batchsent_char_outputs = self.dropout(self.encoder(batchsent_char_embeding))
        batch_sent_char_outputs = batchsent_char_outputs.view(batch_size, sent_size, -1)
        return batch_sent_char_outputs
