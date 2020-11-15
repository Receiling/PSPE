import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    """This class is bidirectional lstm encoder
    """

    def __init__(self, input_size, hidden_size, num_layers=1, is_bidirectional=True, dropout=0.0):
        """This function sets bidirectional lstm neural network parameters

        Arguments:
            input_size {int} -- input dimensions
            hidden_size {int} -- hidden unit (output) dimensions

        Keyword Arguments:
            num_layers {int} -- the number of layers (default: {1})
            is_bidirectional {bool} -- bidirection or not (default: {True})
            dropout {float} -- dropout rate (default: {0.5})
        """

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_size,
                              hidden_size // 2 if is_bidirectional else hidden_size,
                              num_layers=num_layers,
                              bidirectional=is_bidirectional,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

    def get_input_dims(self):
        return self.input_size

    def get_output_dims(self):
        return self.hidden_size

    def get_num_layers(self):
        return self.num_layers

    def forward(self, inputs, seq_lens):
        """This function propagetes forwardly

        Arguments:
            inputs {tensor} -- input data, shape: (batch_size, sequence_len, input_size)
            seq_lens {tensor} -- the length of each sequence

        Returns:
            tensor -- output after bilstm
        """

        batch_size, sequence_len, input_size = inputs.size()

        assert batch_size == len(seq_lens), \
            "batch size is not euqal the size of sequence length list"
        assert input_size == self.input_size, \
            "input size of input data is not equal to `BiLSTMEncode` input size"

        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
        packed_outputs, _ = self.bilstm(packed_inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        return self.dropout(outputs)
