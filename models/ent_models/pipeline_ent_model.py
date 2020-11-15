import logging

import torch.nn as nn

from models.embedding_models.bert_embedding_model import BertEmbedModel
from models.embedding_models.word_char_embedding_model import WordCharEmbedModel
from modules.seq2seq_encoders.seq2seq_bilstm_encoder import BiLSTMEncoder
from modules.decoders.seq_decoder import SeqSoftmaxDecoder
from models.ent_models.cnn_ent_model import CNNEntModel

logger = logging.getLogger(__name__)


class PipelineEntModel(nn.Module):
    """This class utilizes pipeline method to handle
    entity recognition task, firstly detecting entity
    spans with sequence labeling then using CNN for entity spans typing
    """
    def __init__(self, cfg, vocab):
        """This function constructs `PipelineEntModel` components and
        sets `PipelineEntModel` parameters

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

        self.ent_span_decoder = SeqSoftmaxDecoder(
            hidden_size=self.encoder_output_size,
            label_size=self.vocab.get_vocab_size('entity_span_labels'))

        self.cnn_ent_model = CNNEntModel(cfg, vocab, self.encoder_output_size)

    def forward(self, batch_inputs):
        """This function propagetes forwardly

        Arguments:
            batch_inputs {dict} -- batch input data

        Returns:
            dict -- results: ent_loss, ent_pred
        """

        results = {}

        batch_seq_entity_span_labels = batch_inputs['entity_span_labels']
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

        entity_span_ouputs = self.ent_span_decoder(batch_seq_encoder_repr, batch_seq_tokens_mask,
                                                   batch_seq_entity_span_labels)

        batch_inputs['ent_span_preds'] = entity_span_ouputs['predict']

        results['ent_span_loss'] = entity_span_ouputs['loss']
        results['sequence_label_preds'] = entity_span_ouputs['predict']

        ent_model_outputs = self.cnn_ent_model(batch_inputs)
        ent_preds = self.get_ent_preds(batch_inputs, ent_model_outputs)

        results['all_ent_span_preds'] = batch_inputs['all_candi_ents']
        results['ent_loss'] = entity_span_ouputs['loss'] + ent_model_outputs['ent_loss']
        results['all_ent_preds'] = ent_preds

        return results

    def get_ent_preds(self, batch_inputs, ent_model_outputs):
        """This funtion gets entity predictions from entity model outputs
        
        Arguments:
            batch_inputs {dict} -- batch input data
            ent_model_outputs {dict} -- entity model outputs
        
        Returns:
            list -- entity predictions
        """

        ent_preds = []
        candi_ent_cnt = 0
        for ents in batch_inputs['all_candi_ents']:
            cur_ents_num = len(ents)
            ent_pred = {}
            for ent, pred in zip(
                    ents,
                    ent_model_outputs['ent_preds'][candi_ent_cnt:candi_ent_cnt + cur_ents_num]):
                ent_pred_label = self.vocab.get_token_from_index(pred.item(), 'span2ent')
                if ent_pred_label != 'None':
                    ent_pred[ent] = ent_pred_label
            ent_preds.append(ent_pred)
            candi_ent_cnt += cur_ents_num

        return ent_preds

    def get_hidden_size(self):
        """This function returns sentence encoder representation tensor size
        
        Returns:
            int -- sequence encoder output size
        """

        return self.encoder_output_size

    def get_ent_span_feature_size(self):
        """This funtitoin returns entity span feature size
        
        Returns:
            int -- entity span feature size
        """
        return self.cnn_ent_model.get_ent_span_feature_size()
