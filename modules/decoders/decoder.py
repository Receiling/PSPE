import torch.nn as nn
import torch.nn.functional as F


class VanillaSoftmaxDecoder(nn.Module):
    """This decoder firstly applies linear transformation then softmax
    """
    def __init__(self, hidden_size, label_size, bias=True, reduction='mean'):
        """This function sets vanilla softmax decoder input/output size

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

    def forward(self, inputs, labels=None):
        """This function propagetes forwardly

        Arguments:
            inputs {tensor} -- input data
        
        Keyword Arguments:
            labels {tensor} -- label data (default: {None})

        Returns:
            dict -- result: loss, predict, softmax_prob
        """

        batch_size, input_size = inputs.size()

        assert input_size == self.hidden_size, "input size is not equal to hidden size"

        results = {}
        outputs = self.hidden2label(inputs)
        log_probs = F.log_softmax(outputs, dim=1)
        preds = log_probs.argmax(dim=1)
        results["predict"] = preds
        # results["log_probs"] = F.log_softmax(outputs, dim=1)

        if labels is not None:
            avg_loss = self.loss(outputs, labels)
            results["loss"] = avg_loss

        return results
