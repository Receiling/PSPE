import torch.nn as nn
import torch.nn.functional as F

from modules.seq2vec_encoders.cnn_encoder import CNNEncoder
from utils.nn_utils import get_device_of, get_range_vector, batched_index_select


class CNNSpanExtractor(nn.Module):
    """This class computes all span representations by running CnnEncoder for all spans in the sentence parallel.
    """
    def __init__(self,
                 input_size,
                 num_filters,
                 ngram_filter_sizes=(2, 3, 4, 5),
                 conv_layer_activation=nn.GELU(),
                 dropout=0.0,
                 output_size=0):
        """This function sets `CNNSpanExtractor` parameters.

        Arguments:
            input_size {int} -- input size
            num_filters {int} -- the number of filters
    
        Keyword Arguments:
            ngram_filter_sizes {tuple} -- filter size list (default: {(2, 3, 4, 5)})
            conv_layer_activation {Module} -- convolution layer activation (default: {gelu})
            dropout {float} -- dropout rate (default: {0.0})
            output_size {int} -- output size (default: {None})
        """

        super().__init__()
        self.input_size = input_size
        self.cnn_encoder = CNNEncoder(input_size=input_size,
                                      num_filters=num_filters,
                                      ngram_filter_sizes=ngram_filter_sizes,
                                      conv_layer_activation=conv_layer_activation,
                                      dropout=dropout,
                                      output_size=output_size)
        self.output_size = self.cnn_encoder.get_output_dims()

    def get_input_dims(self):
        return self.input_size

    def get_output_dims(self):
        return self.output_size

    def forward(self, sequence_tensor, span_indices):
        """This function propagates forwardly.

        Arguments:
            sequence_tensor {tensor} -- sequence tensor
            span_indices {tensor} -- span index tensor

        Returns:
            tensor -- span convolution embedding
        """

        # both of tensors' shape: (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)

        # shape: (batch_size, num_spans, 1)
        # These span widths are off by 1, because the span ends are `inclusive`.
        span_widths = span_ends - span_starts

        # We need to know the maximum span width so we can
        # generate indices to extract the spans from the sequence tensor.
        # These indices will then get masked below, such that if the length
        # of a given span is smaller than the max, the rest of the values
        # are masked.
        max_batch_span_width = span_widths.max().item() + 1

        # shape: (1, 1, max_batch_span_width)
        max_span_range_indices = get_range_vector(max_batch_span_width,
                                                  get_device_of(sequence_tensor)).view(1, 1, -1)

        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        #
        # We're using < here (and for the mask below) because the span ends are
        # not inclusive, so we want to include indices which are equal to span_widths rather
        # than using it as a non-inclusive upper bound.
        span_indices_mask = (max_span_range_indices < span_widths).long()

        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This operation just like reversing the arragement of (continually) span indices
        raw_span_indices = (span_ends - 1 - max_span_range_indices)
        # Using RElU function remove these elements which are smaller than zero
        span_indices = F.relu(raw_span_indices.float()).long()

        # Shape: (batch_size, num_spans, max_batch_span_width, embeding_dim)
        # Firstly call flatten_and_batch_shift_indices transforms span_indices
        # Then selects indexed embedding
        span_embedding = batched_index_select(sequence_tensor, span_indices)

        batch_size, num_spans, _, _ = span_embedding.size()
        span_conv_embedding = self.cnn_encoder(
            inputs=span_embedding.view(batch_size * num_spans, max_batch_span_width, -1),
            mask=span_indices_mask.view(batch_size * num_spans, max_batch_span_width))
        span_conv_embedding = span_conv_embedding.view(batch_size, num_spans, -1)
        return span_conv_embedding
