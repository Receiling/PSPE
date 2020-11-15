import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """This class is Convolution Neural Network Encoder
    """
    def __init__(self,
                 input_size,
                 num_filters,
                 ngram_filter_sizes=(2, 3, 4, 5),
                 output_size=0,
                 conv_layer_activation=nn.GELU(),
                 dropout=0.0):
        """This funciton sets `CNNEncoder` parameters

        Arguments:
            input_size {int} -- input dim
            num_filters {int} -- the number of filters

        Keyword Arguments:
            ngram_filter_sizes {tuple} -- filter size list (default: {(2, 3, 4, 5)})
            output_size {[int]} -- output dim (default: {0})
            conv_layer_activation {nn.Module} -- activation function (default: {nn.GELU()})
            dropout {float} -- drop out rate
        """

        super().__init__()
        self.input_size = input_size
        self.num_filters = num_filters
        self.ngram_filter_sizes = ngram_filter_sizes
        self.activation = conv_layer_activation
        self.output_size = output_size

        self.conv_layers = [
            nn.Conv1d(in_channels=self.input_size,
                      out_channels=self.num_filters,
                      kernel_size=ngram_filter_size,
                      padding=(ngram_filter_size, )) for ngram_filter_size in ngram_filter_sizes
        ]

        for id, conv_layer in enumerate(self.conv_layers):
            self.add_module('conv_layer_{}'.format(id), conv_layer)

        maxpool_output_size = self.num_filters * len(self.ngram_filter_sizes)

        if self.output_size > 0:
            self.projection_layer = nn.Sequential(nn.Linear(maxpool_output_size, self.output_size),
                                                  self.activation)
        else:
            self.output_size = maxpool_output_size
            self.projection_layer = lambda x: x

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

        self.apply(self.init_weights)

    def init_weights(self, module):
        """init_weight initialize the weights of parameters.
        """

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_input_dims(self):
        return self.input_size

    def get_output_dims(self):
        return self.output_size

    def forward(self, inputs, mask=None):
        """This function propagetes forwardly

        Arguments:
            inputs {tensor} -- input data, shape: (batch_size, sequence_len, input_size)

        Keyword Arguments:
            mask {tensor} -- mask martrix (default: {None})

        Returns:
            tensor -- output after cnn
        """

        if mask is not None:
            inputs = inputs * mask.unsqueeze(-1).float()

        # Our input is expected to have shape `(batch_size, num_tokens, embedding_dim)`.
        # The convolution layers expect input of shape `(batch_size, in_channels, sequence_length)`, 
        # where the conv layer `in_channels` is our `embedding_dim`.
        # We thus need to transpose the tensor first.

        inputs.transpose_(1, 2)

        filter_outputs = []
        for id in range(len(self.conv_layers)):
            conv_layer = getattr(self, 'conv_layer_{}'.format(id))
            filter_outputs.append(self.activation(conv_layer(inputs)).max(dim=2)[0])

        maxpool_output = torch.cat(filter_outputs,
                                   dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        output = self.projection_layer(maxpool_output)

        return self.dropout(output)
