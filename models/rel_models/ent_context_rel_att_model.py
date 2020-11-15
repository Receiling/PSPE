from collections import defaultdict
from os import terminal_size

import torch
import torch.nn as nn

from modules.span_extractors.cnn_span_extractor import CNNSpanExtractor
from modules.token_embedders.bert_encoder import BertLinear
from modules.decoders.decoder import VanillaSoftmaxDecoder
from modules.seq2vec_encoders.position_away_attention_encoder import PosAwareAttEncoder


class EntConRelAttModel(nn.Module):
    """This class predicts relation between two candidate entity with attention,
    and extracts entity span representations.
    """
    def __init__(self, cfg, vocab, input_size, reduction='mean'):
        """This function sets `EntConRelAttModel` parameters

        Arguments:
            cfg {dict} -- config parameters for constructing multiple models
            vocab {dict} -- vocabulary
            input_size {int} -- input size

        Keyword Arguments:
            reduction {str} -- crossentropy loss recduction (default: {mean})
        """

        super().__init__()

        self.span_batch_size = cfg.span_batch_size
        self.ent_output_size = cfg.ent_output_size
        self.context_output_size = cfg.context_output_size
        self.output_size = cfg.ent_mention_output_size
        self.att_size = cfg.att_size
        self.position_embedding_dims = cfg.position_embedding_dims
        self.activation = nn.GELU()
        self.dropout = cfg.dropout
        self.device = cfg.device

        self.entity_span_extractor = CNNSpanExtractor(
            input_size=input_size,
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

        self.context_span_extractor = CNNSpanExtractor(
            input_size=input_size,
            num_filters=cfg.context_cnn_output_channels,
            ngram_filter_sizes=cfg.context_cnn_kernel_sizes,
            dropout=cfg.dropout)

        if self.context_output_size > 0:
            self.context2hidden = BertLinear(
                input_size=self.context_span_extractor.get_output_dims(),
                output_size=self.context_output_size,
                activation=self.activation,
                dropout=self.dropout)
        else:
            self.context_output_size = self.context_span_extractor.get_output_dims()
            self.context2hidden = lambda x: x

        assert self.ent_output_size == self.context_output_size

        self.position_embedding = nn.Embedding(7, self.position_embedding_dims)
        self.position_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.attention_encoder = PosAwareAttEncoder(self.ent_output_size,
                                                    input_size,
                                                    2 * self.position_embedding_dims,
                                                    self.att_size,
                                                    activation=self.activation,
                                                    dropout=cfg.dropout)
        self.ent_mention_mlp = BertLinear(input_size=self.ent_output_size,
                                          output_size=self.ent_output_size,
                                          activation=self.activation,
                                          dropout=self.dropout)

        if self.output_size > 0:
            self.mlp = BertLinear(input_size=self.ent_output_size,
                                  output_size=self.output_size,
                                  activation=self.activation,
                                  dropout=self.dropout)
        else:
            self.output_size = self.ent_output_size
            self.mlp = lambda x: x

        self.relation_decoder = VanillaSoftmaxDecoder(hidden_size=self.output_size,
                                                      label_size=vocab.get_vocab_size('span2rel'),
                                                      reduction=reduction)

        self.subj_pos = torch.LongTensor([-1, 0, 1, 2, 3]) + 3
        self.obj_pos = torch.LongTensor([-3, -2, -1, 0, 1]) + 3
        self.context_zero_feat = torch.zeros(self.entity_span_extractor.get_output_dims())
        if self.device > -1:
            self.subj_pos = self.subj_pos.cuda(device=self.device, non_blocking=True)
            self.obj_pos = self.obj_pos.cuda(device=self.device, non_blocking=True)
            self.context_zero_feat = self.context_zero_feat.cuda(device=self.device,
                                                                 non_blocking=True)

    def forward(self, batch_inputs):
        """This function propagetes forwardly

        Arguments:
            batch_inputs {dict} -- batch input data

        Returns:
            dict -- outputs: rel_inputs, all_candi_rel_labels
        """

        all_candi_rels = batch_inputs['all_candi_rels']
        seq_lens = batch_inputs['tokens_lens']
        batch_seq_encoder_reprs = batch_inputs['seq_encoder_reprs']

        batch_entity_spans, batch_context_spans = self.generate_all_spans(all_candi_rels, seq_lens)

        batch_entity_spans_feature = self.cache_spans_feature(batch_entity_spans,
                                                              batch_seq_encoder_reprs,
                                                              self.entity_span_extractor,
                                                              self.span_batch_size, self.device)
        batch_context_spans_feature = self.cache_spans_feature(batch_context_spans,
                                                               batch_seq_encoder_reprs,
                                                               self.context_span_extractor,
                                                               self.span_batch_size, self.device)

        batch_rels = self.create_batch_rels(batch_inputs, batch_entity_spans_feature,
                                            batch_context_spans_feature)

        relation_outputs = self.relation_decoder(batch_rels['rel_inputs'],
                                                 batch_rels['all_candi_rel_labels'])

        results = {}
        results['rel_loss'] = relation_outputs['loss']
        results['rel_preds'] = relation_outputs['predict']

        return results

    def create_batch_rels(self, batch_inputs, batch_entity_spans_feature,
                          batch_context_spans_feature):
        """This function creates batch relation inputs

        Arguments:
            batch_inputs {dict} -- batch inputs
            batch_entity_spans_feature {list} -- entity span feature list
            batch_context_spans_feature {list} -- context spans feature list

        Returns:
            dict -- batch realtion inputs
        """

        batch_rels = defaultdict(list)

        for idx, seq_len in enumerate(batch_inputs['tokens_lens']):
            batch_rels['all_candi_rel_labels'].extend(batch_inputs['all_candi_rel_labels'][idx])
            for e1, e2 in batch_inputs['all_candi_rels'][idx]:
                L = (0, e1[0])
                E1 = (e1[0], e1[1])
                M = (e1[1], e2[0])
                E2 = (e2[0], e2[1])
                R = (e2[1], seq_len)

                if L[0] >= L[1]:
                    batch_rels['L'].append(self.context_zero_feat)
                else:
                    batch_rels['L'].append(batch_context_spans_feature[idx][L])

                if M[0] >= M[1]:
                    batch_rels['M'].append(self.context_zero_feat)
                else:
                    batch_rels['M'].append(batch_context_spans_feature[idx][M])

                if R[0] >= R[1]:
                    batch_rels['R'].append(self.context_zero_feat)
                else:
                    batch_rels['R'].append(batch_context_spans_feature[idx][R])

                batch_rels['E1'].append(batch_entity_spans_feature[idx][E1])
                batch_rels['E2'].append(batch_entity_spans_feature[idx][E2])

                batch_rels['cls'].append(batch_inputs['seq_cls_repr'][idx])

        batch_rels['E1'] = self.ent2hidden(torch.stack(batch_rels['E1'])).unsqueeze(1)
        batch_rels['E2'] = self.ent2hidden(torch.stack(batch_rels['E2'])).unsqueeze(1)

        batch_rels['L'] = self.ent2hidden(torch.stack(batch_rels['L'])).unsqueeze(1)
        batch_rels['M'] = self.ent2hidden(torch.stack(batch_rels['M'])).unsqueeze(1)
        batch_rels['R'] = self.ent2hidden(torch.stack(batch_rels['R'])).unsqueeze(1)

        batch_rels['cls'] = torch.stack(batch_rels['cls'])

        rel_feature = torch.cat(
            [batch_rels['L'], batch_rels['E1'], batch_rels['M'], batch_rels['E2'], batch_rels['R']],
            dim=1)

        subj_pos_emb = self.position_embedding(self.subj_pos)
        obj_pos_emb = self.position_embedding(self.obj_pos)
        pos_emb = torch.cat([subj_pos_emb, obj_pos_emb],
                            dim=1).unsqueeze(0).repeat(rel_feature.size()[0], 1, 1)

        rel_feature = self.ent_mention_mlp(
            self.attention_encoder(inputs=rel_feature, query=batch_rels['cls'], feature=pos_emb))

        batch_rels['rel_inputs'] = self.mlp(rel_feature)

        batch_rels['all_candi_rel_labels'] = torch.LongTensor(batch_rels['all_candi_rel_labels'])
        if self.device > -1:
            batch_rels['all_candi_rel_labels'] = batch_rels['all_candi_rel_labels'].cuda(
                device=self.device, non_blocking=True)

        return batch_rels

    def cache_spans_feature(self, batch_spans, batch_seq_encoder_reprs, span_extractor,
                            span_batch_size, device):
        """This function calculates spans feature for caching

        Arguments:
            spans {list} -- spans
            batch_seq_encoder_reprs {list} -- batch sequence encoder reprentations
            span_extractor {nn.Module} -- span extractor model
            span_batch_size {int} -- span batch size
            device {int} -- device {int} -- device id: cpu: -1, gpu: >= 0 (default: {-1})

        Returns:
            list -- batch caching spans feature
        """

        assert len(batch_spans) == len(batch_seq_encoder_reprs), "batch spans' size is not correct."

        all_spans = []
        all_seq_encoder_reprs = []
        for spans, seq_encoder_reprs in zip(batch_spans, batch_seq_encoder_reprs):
            all_spans.extend((span[0], span[1]) for span in spans)
            all_seq_encoder_reprs.extend([seq_encoder_reprs for _ in range(len(spans))])

        batch_spans_feature = [{} for _ in range(len(batch_spans))]

        if len(all_spans) == 0:
            return batch_spans_feature

        if span_batch_size > 0:
            all_spans_feature = []
            for idx in range(0, len(all_spans), span_batch_size):
                batch_spans_tensor = torch.LongTensor(all_spans[idx:idx +
                                                                span_batch_size]).unsqueeze(1)
                if self.device > -1:
                    batch_spans_tensor = batch_spans_tensor.cuda(device=device, non_blocking=True)
                batch_seq_encoder_reprs = torch.stack(all_seq_encoder_reprs[idx:idx +
                                                                            span_batch_size])

                all_spans_feature.append(
                    span_extractor(batch_seq_encoder_reprs, batch_spans_tensor).squeeze(1))
            all_spans_feature = torch.cat(all_spans_feature, dim=0)
        else:
            all_spans_tensor = torch.LongTensor(all_spans).unsqueeze(1)
            if self.device > -1:
                all_spans_tensor = all_spans_tensor.cuda(device=device, non_blocking=True)
            all_seq_encoder_reprs = torch.stack(all_seq_encoder_reprs)
            all_spans_feature = span_extractor(all_seq_encoder_reprs, all_spans_tensor).squeeze(1)

        idx = 0
        for i, spans in enumerate(batch_spans):
            for span in spans:
                batch_spans_feature[i][span] = all_spans_feature[idx]
                idx += 1

        return batch_spans_feature

    def generate_all_spans(self, all_candi_rels, seq_lens):
        """This function generates all entity and context spans

        Arguments:
            all_candi_rels {list} -- all candidate relation list
            seq_lens {list} -- batch sequence length

        Returns:
            list -- all entity and context spans
        """

        assert len(all_candi_rels) == len(seq_lens), "candidate relations' size is not correct."

        batch_entity_spans = []
        batch_context_spans = []
        for candi_rels, seq_len in zip(all_candi_rels, seq_lens):
            entity_spans = set()
            context_spans = set()
            for e1, e2 in candi_rels:
                if e1[0] < e1[1]:
                    entity_spans.add(e1)
                if e2[0] < e2[1]:
                    entity_spans.add(e2)

                L = (0, e1[0])
                M = (e1[1], e2[0])
                R = (e2[1], seq_len)

                # L, M, R can be empty
                for span in [L, M, R]:
                    if span[0] < span[1]:
                        context_spans.add(span)

            batch_entity_spans.append(list(entity_spans))
            batch_context_spans.append(list(context_spans))

        return batch_entity_spans, batch_context_spans
