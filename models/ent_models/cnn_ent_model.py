from collections import defaultdict
import logging

import torch
import torch.nn as nn
import numpy as np

from modules.decoders.decoder import VanillaSoftmaxDecoder
from modules.span_extractors.cnn_span_extractor import CNNSpanExtractor
from modules.token_embedders.bert_encoder import BertLinear
from utils.entity_chunking import get_entity_span

logger = logging.getLogger(__name__)


class CNNEntModel(nn.Module):
    """This class predicts entities using CNN.
    """
    def __init__(self, cfg, vocab, seq_encoder_output_size):
        """This function constructs `CNNEntModel` components and
        sets `CNNEntModel` parameters

        Arguments:
            cfg {dict} -- config parameters for constructing multiple models
            vocab {Vocabulary} -- vocabulary
            seq_encoder_output_size {int} -- sequence encoder output size
        """

        super().__init__()
        self.vocab = vocab
        self.span_batch_size = cfg.span_batch_size
        self.ent_output_size = cfg.ent_output_size
        self.activation = nn.GELU()
        self.schedule_k = cfg.schedule_k
        self.device = cfg.device
        self.seq_encoder_output_size = seq_encoder_output_size
        self.pretrain_epoches = cfg.pretrain_epoches

        self.entity_span_extractor = CNNSpanExtractor(
            input_size=self.seq_encoder_output_size,
            num_filters=cfg.entity_cnn_output_channels,
            ngram_filter_sizes=cfg.entity_cnn_kernel_sizes,
            dropout=cfg.dropout)

        if self.ent_output_size > 0:
            self.ent2hidden = BertLinear(input_size=self.entity_span_extractor.get_output_dims(),
                                         output_size=self.ent_output_size,
                                         activation=self.activation,
                                         dropout=cfg.dropout)
        else:
            self.ent_output_size = self.entity_span_extractor.get_output_dims()
            self.ent2hidden = lambda x: x

        self.entity_decoder = VanillaSoftmaxDecoder(
            hidden_size=self.ent_output_size, label_size=self.vocab.get_vocab_size('span2ent'))

    def forward(self, batch_inputs):
        """This function propagetes forwardly

        Arguments:
            batch_inputs {dict} -- batch input data

        Returns:
            dict -- results: ent_loss, ent_pred
        """

        batch_seq_encoder_reprs = batch_inputs['seq_encoder_reprs']
        batch_ent_span_label_inputs = batch_inputs['entity_span_labels']
        batch_ent_span_preds = batch_inputs['ent_span_preds']
        seq_lens = batch_inputs['tokens_lens']

        results = {}

        if self.training and self.schedule_k > 0 and 'epoch' in batch_inputs:
            if batch_inputs['epoch'] > self.pretrain_epoches:
                schedule_p = self.schedule_k / (self.schedule_k + np.exp(
                    (batch_inputs['epoch'] - self.pretrain_epoches) / self.schedule_k))
                ent_span_preds = [
                    gold if np.random.random() < schedule_p else pred
                    for gold, pred in zip(batch_ent_span_label_inputs, batch_ent_span_preds)
                ]
            else:
                ent_span_preds = [gold for gold in batch_ent_span_label_inputs]
            ent_span_preds = torch.stack(ent_span_preds)
        else:
            ent_span_preds = batch_ent_span_preds

        all_candi_ents, all_candi_ent_labels = self.generate_all_candi_ents(
            batch_inputs, ent_span_preds)

        batch_inputs['all_candi_ents'] = all_candi_ents
        batch_inputs['all_candi_ent_labels'] = all_candi_ent_labels

        if sum(len(candi_ents) for candi_ents in all_candi_ents) == 0:
            zero_loss = torch.Tensor([0])
            zero_loss.requires_grad = True
            if self.device > -1:
                zero_loss = zero_loss.cuda(device=self.device, non_blocking=True)
            batch_size = len(batch_inputs['tokens'])
            ent_labels = [[] for _ in range(batch_size)]
            results['ent_loss'] = zero_loss
            results['ent_preds'] = ent_labels
            return results

        batch_ent_spans_feature = self.cache_ent_spans_feature(all_candi_ents, seq_lens,
                                                               batch_seq_encoder_reprs,
                                                               self.entity_span_extractor,
                                                               self.span_batch_size, self.device)

        batch_inputs['ent_spans_feature'] = batch_ent_spans_feature

        batch_ents = self.create_batch_ents(batch_inputs, batch_ent_spans_feature)

        entity_outputs = self.entity_decoder(batch_ents['ent_inputs'],
                                             batch_ents['all_candi_ent_labels'])

        results['ent_loss'] = entity_outputs['loss']
        results['ent_preds'] = entity_outputs['predict']

        return results

    def get_ent_span_feature_size(self):
        """This funtitoin returns entity span feature size
        
        Returns:
            int -- entity span feature size
        """
        return self.ent_output_size

    def create_batch_ents(self, batch_inputs, batch_ent_spans_feature):
        """This function creates batch entity inputs

        Arguments:
            batch_inputs {dict} -- batch inputs
            batch_ent_spans_feature {list} -- entity spans feature list

        Returns:
            dict -- batch entity inputs
        """

        batch_ents = defaultdict(list)

        for idx, _ in enumerate(batch_inputs['tokens_lens']):
            batch_ents['ent_inputs'].extend(batch_ent_spans_feature[idx][ent_span]
                                            for ent_span in batch_inputs['all_candi_ents'][idx])
            batch_ents['all_candi_ent_labels'].extend(batch_inputs['all_candi_ent_labels'][idx])

        batch_ents['ent_inputs'] = torch.stack(batch_ents['ent_inputs'])

        batch_ents['all_candi_ent_labels'] = torch.LongTensor(batch_ents['all_candi_ent_labels'])
        if self.device > -1:
            batch_ents['all_candi_ent_labels'] = batch_ents['all_candi_ent_labels'].cuda(
                device=self.device, non_blocking=True)

        return batch_ents

    def cache_ent_spans_feature(self, batch_ent_spans, seq_lens, batch_seq_encoder_reprs,
                                ent_sapn_extractor, ent_batch_size, device):
        """This function extracts all entity spans feature for caching

        Arguments:
            batch_ent_spans {list} -- batch entity spans
            seq_lens {list} -- batch sequence length
            batch_seq_encoder_reprs {list} -- batch sequence encoder reprentations
            ent_sapn_extractor {nn.Module} -- entity extractor model
            ent_batch_size {int} -- entity batch size
            device {int} -- device {int} -- device id: cpu: -1, gpu: >= 0 (default: {-1})

        Returns:
            list -- batch caching spans feature
        """

        assert len(batch_ent_spans) == len(
            batch_seq_encoder_reprs), "batch spans' size is not correct."

        all_spans = []
        all_seq_encoder_reprs = []
        for ent_spans, seq_encoder_reprs in zip(batch_ent_spans, batch_seq_encoder_reprs):
            all_spans.extend(ent_spans)
            all_seq_encoder_reprs.extend([seq_encoder_reprs for _ in range(len(ent_spans))])

        ent_spans_feature = [{} for _ in range(len(seq_lens))]

        if len(all_spans) == 0:
            return ent_spans_feature

        if ent_batch_size > 0:
            all_spans_feature = []
            for idx in range(0, len(all_spans), ent_batch_size):
                batch_spans_tensor = torch.LongTensor(all_spans[idx:idx +
                                                                ent_batch_size]).unsqueeze(1)
                if self.device > -1:
                    batch_spans_tensor = batch_spans_tensor.cuda(device=device, non_blocking=True)
                batch_seq_encoder_reprs = torch.stack(all_seq_encoder_reprs[idx:idx +
                                                                            ent_batch_size])

                all_spans_feature.append(
                    ent_sapn_extractor(batch_seq_encoder_reprs, batch_spans_tensor).squeeze(1))
            all_spans_feature = torch.cat(all_spans_feature, dim=0)
        else:
            all_spans_tensor = torch.LongTensor(all_spans).unsqueeze(1)
            if self.device > -1:
                all_spans_tensor = all_spans_tensor.cuda(device=device, non_blocking=True)
            all_seq_encoder_reprs = torch.stack(all_seq_encoder_reprs)
            all_spans_feature = ent_sapn_extractor(all_seq_encoder_reprs,
                                                   all_spans_tensor).squeeze(1)

        all_spans_feature = self.ent2hidden(all_spans_feature)

        idx = 0
        for i, ent_spans in enumerate(batch_ent_spans):
            for ent_span in ent_spans:
                ent_spans_feature[i][ent_span] = all_spans_feature[idx]
                idx += 1

        return ent_spans_feature

    def generate_all_candi_ents(self, batch_inputs, ent_span_labels):
        """This funtion generate all candidate entities
        
        Arguments:
            batch_inputs {dict} -- batch input data
            ent_span_labels {list} -- entity span labels list
        
        Returns:
            tuple -- all candidate entities, all candidate entities label
        """

        all_candi_ents = []
        all_candi_ent_labels = []
        for idx, seq_len in enumerate(batch_inputs['tokens_lens']):
            ent_span_label = [
                self.vocab.get_token_from_index(label.item(), 'entity_span_labels')
                for label in ent_span_labels[idx][:seq_len]
            ]
            ent_span_label = [item if item == 'O' else item + '-ENT' for item in ent_span_label]
            span2ent = get_entity_span(ent_span_label)
            candi_ents = set(span2ent.keys())
            if self.training:
                candi_ents.update(batch_inputs['span2ent'][idx].keys())

            candi_ents = list(candi_ents)
            candi_ent_labels = []
            for ent in candi_ents:
                if ent in batch_inputs['span2ent'][idx]:
                    candi_ent_labels.append(batch_inputs['span2ent'][idx][ent])
                else:
                    candi_ent_labels.append(self.vocab.get_token_index('None', 'span2ent'))

            all_candi_ents.append(candi_ents)
            all_candi_ent_labels.append(candi_ent_labels)

        return all_candi_ents, all_candi_ent_labels
