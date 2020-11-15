import torch.nn as nn
import torch.nn.functional as F


class PosAwareAttEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 query_size,
                 feature_size,
                 att_size,
                 activation=nn.GELU(),
                 dropout=0.0):
        """PosAwareAttEncoder encoders different size vector into same representation size with attention mechanism.

        Arguments:
            input_size {int} -- input size
            query_size {int} -- query vector size
            feature_size {int} -- augmented feature size
            att_size {int} -- attention vector size

        Keyword Arguments:
            activation {nn.Module} -- activation (default: {nn.GELU()})
            dropout {float} -- dropout rate (default: {0.0})
        """

        super().__init__()

        self.input_size = input_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.att_size = att_size

        self.input_linear = nn.Linear(self.input_size, self.att_size, bias=False)
        self.query_linear = nn.Linear(self.query_size, self.att_size, bias=False)

        if self.feature_size > 0:
            self.feature_linear = nn.Linear(self.feature_size, self.att_size, bias=False)

        self.score_linear = nn.Linear(self.att_size, 1, bias=False)
        self.activation = activation

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

    def forward(self, inputs, query, mask=None, feature=None):
        """forward propagate forwardly

        Arguments:
            inputs {tensor} -- input tensor (K & V) shape: (batch_size, seq_len, input_size)
            query {tensor} -- query tensor (Q) shape: (batch_size, query_size)

        Keyword Arguments:
            mask {tensor} -- mask tensor for padding shape: (batch_size, seq_len)(default: {None})
            feature {tensor} -- argumented feature tensor (K) shape: (batch_size, seq_len, feature_size) (default: {None})
        """

        batch_size, seq_len, _ = inputs.size()

        inputs_proj = self.input_linear(inputs.view(-1, self.input_size)).view(
            batch_size, seq_len, self.att_size)
        query_proj = self.query_linear(query.view(-1, self.query_size)).view(
            batch_size, self.att_size).unsqueeze(1).expand(batch_size, seq_len, self.att_size)

        projs = [inputs_proj, query_proj]

        if feature is not None and self.feature_size > 0:
            feature_proj = self.feature_linear(feature.view(-1, self.feature_size)).view(
                batch_size, seq_len, self.att_size)
            projs.append(feature_proj)

        score = self.score_linear(self.activation(sum(projs)).view(-1, self.att_size)).view(
            batch_size, seq_len)

        # 0 represents mask while 1 represents no mask
        if mask is not None:
            score.data.masked_fill_(mask == 0, -float('inf'))

        attention = F.softmax(score, dim=1)

        outputs = attention.unsqueeze(1).bmm(inputs).squeeze(1)
        return self.dropout(outputs)

    def get_input_dims(self):
        return self.input_size

    def get_output_dims(self):
        return self.input_size
