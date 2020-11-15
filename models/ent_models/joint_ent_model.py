import logging

import torch.nn as nn

from models.embedding_models.bert_embedding_model import BertEmbedModel
from models.embedding_models.word_char_embedding_model import WordCharEmbedModel
from modules.seq2seq_encoders.seq2seq_bilstm_encoder import BiLSTMEncoder
from modules.decoders.seq_decoder import SeqSoftmaxDecoder
from utils.entity_chunking import get_entity_span

logger = logging.getLogger(__name__)


class JointEntModel(nn.Module):
    """This class regrads entity recognition task as a sequence labeling task, and utilizes bilstm model to handle it
    """
    def __init__(self, cfg, vocab):
        """This function constructs `JointEntModel` components and
        sets `JointEntModel` parameters

        Arguments:
            cfg {dict} -- config parameters for constructing multiple models
            vocab {Vocabulary} -- vocabulary
        """

        super().__init__()
        self.vocab = vocab
        self.lstm_layers = cfg.lstm_layers

        if cfg.embedding_model == 'word_char':
            self.embedding_model = WordCharEmbedModel(cfg, vocab)
        else:
            self.embedding_model = BertEmbedModel(cfg, vocab)

        self.encoder_output_size = self.embedding_model.get_hidden_size()

        if self.lstm_layers > 0:
            self.seq_encoder = BiLSTMEncoder(input_size=self.encoder_output_size,
                                             hidden_size=cfg.lstm_hidden_unit_dims,
                                             num_layers=cfg.lstm_layers,
                                             dropout=cfg.dropout)
            self.encoder_output_size = self.seq_encoder.get_output_dims()

        self.ent_decoder = SeqSoftmaxDecoder(hidden_size=self.encoder_output_size,
                                             label_size=self.vocab.get_vocab_size('entity_labels'))

    def forward(self, batch_inputs):
        """This function propagetes forwardly

        Arguments:
            batch_inputs {dict} -- batch input data

        Returns:
            dict -- results: ent_loss, ent_pred
        """

        results = {}

        batch_seq_entity_labels = batch_inputs['entity_labels']
        batch_seq_tokens_lens = batch_inputs['tokens_lens']
        batch_seq_tokens_mask = batch_inputs['tokens_mask']

        self.embedding_model(batch_inputs)
        batch_seq_tokens_encoder_repr = batch_inputs['seq_encoder_reprs']

        if self.lstm_layers > 0:
            batch_seq_encoder_repr = self.seq_encoder(batch_seq_tokens_encoder_repr,
                                                      batch_seq_tokens_lens).contiguous()
        else:
            batch_seq_encoder_repr = batch_seq_tokens_encoder_repr

        batch_inputs['seq_encoder_reprs'] = batch_seq_encoder_repr

        ent_outputs = self.ent_decoder(batch_seq_encoder_repr, batch_seq_tokens_mask,
                                       batch_seq_entity_labels)

        batch_inputs['ent_label_preds'] = ent_outputs['predict']

        ent_preds = self.get_ent_preds(batch_inputs)

        results['sequence_label_preds'] = ent_outputs['predict']
        results['ent_loss'] = ent_outputs['loss']
        results['all_ent_preds'] = ent_preds

        return results

    def get_ent_preds(self, batch_inputs):
        """This funtion gets entity predictions from entity decoder outputs
        
        Arguments:
            batch_inputs {dict} -- batch input data
        
        Returns:
            list -- entity predictions
        """

        ent_preds = []
        for idx, seq_len in enumerate(batch_inputs['tokens_lens']):
            ent_span_label = [
                self.vocab.get_token_from_index(label.item(), 'entity_labels')
                for label in batch_inputs['ent_label_preds'][idx][:seq_len]
            ]
            span2ent = get_entity_span(ent_span_label)
            ent_preds.append(span2ent)

        return ent_preds

    def get_hidden_size(self):
        """This function returns sentence encoder representation tensor size
        
        Returns:
            int -- sequence encoder output size
        """

        return self.encoder_output_size
