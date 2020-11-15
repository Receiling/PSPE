import logging

import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SeqSoftmaxDecoder(nn.Module):
    """This class decodes sequence hidden unit
    """
    def __init__(self, hidden_size, label_size, bias=True, reduction='mean'):
        """This function sets SeqSoftmaxDecoder input/output size

        Arguments:
            hidden_size {int} -- the size of hidden unit
            label_size {int} -- the size of label

        Keyword Arguments:
            bias {bool} -- adding bias or not (default: {False})
            reduction {str} -- crossentropy loss recduction (default: {mean})
        """

        super().__init__()
        self.hidden_size = hidden_size
        self.label_size = label_size

        self.hidden2label = nn.Linear(hidden_size, label_size, bias)
        self.hidden2label.weight.data.normal_(mean=0.0, std=0.02)
        if self.hidden2label.bias is not None:
            self.hidden2label.bias.data.zero_()

        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def get_input_dim(self):
        return self.hidden_size

    def get_output_dim(self):
        return self.label_size

    def forward(self, seq_inputs, seq_mask=None, seq_labels=None):
        """This function propagetes forwardly

        Arguments:
            seq_inputs {tensor} -- input data, shape: (batch_size, seq_size, input_size)
        
        Keyword Arguments:
            seq_mask {tensor} -- mask tensor, shape: (batch_size, seq_size) (default: {None})
            seq_labels {tensor} -- label data, shape: (batch_size, seq_size) (default: {None})
        
        Returns:
            dict -- resutls: loss, predict, log_probs
        """

        batch_size, seq_size, input_size = seq_inputs.size()

        assert input_size == self.hidden_size, "input size is not equal to hidden size"

        results = {}

        seq_outpus = self.hidden2label(seq_inputs)
        seq_log_probs = F.log_softmax(seq_outpus, dim=2)
        seq_preds = seq_log_probs.argmax(dim=2)
        results['predict'] = seq_preds
        # results['log_probs'] = seq_log_probs

        if seq_labels is not None:
            if seq_mask is not None:
                active_loss = seq_mask.view(-1) == 1
                active_outputs = seq_outpus.view(-1, self.label_size)[active_loss]
                active_labels = seq_labels.view(-1)[active_loss]
                no_pad_avg_loss = self.loss(active_outputs, active_labels)
                results['loss'] = no_pad_avg_loss
            else:
                avg_loss = self.loss(seq_outpus.view(-1, self.label_size), seq_labels.view(-1))
                results['loss'] = avg_loss

        return results
