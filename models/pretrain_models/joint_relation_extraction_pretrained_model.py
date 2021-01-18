import logging

import torch
import torch.nn as nn

from modules.token_embedders.bert_encoder import BertEncoder
from modules.token_embedders.bert_encoder import BertLinear
from modules.span_extractors.cnn_span_extractor import CNNSpanExtractor
from modules.decoders.decoder import VanillaSoftmaxDecoder
from utils.nn_utils import batched_index_select
from modules.seq2vec_encoders.position_away_attention_encoder import PosAwareAttEncoder

logger = logging.getLogger(__name__)


class PretrainedSpanEncoder(nn.Module):
    """PretrainedSpanEncoder encodes span into vector.
    """
    def __init__(self, cfg, momentum=False):
        """This funciton constructs `PretrainedSpanEncoder` components

        Arguments:
            cfg {dict} -- config parameters for constructing multiple models

        Keyword Arguments:
            momentum {bool} -- whether this encoder is momentum encoder (default: {False})
        """

        super().__init__()
        self.ent_output_size = cfg.ent_output_size
        self.span_batch_size = cfg.span_batch_size
        self.position_embedding_dims = cfg.position_embedding_dims
        self.att_size = cfg.att_size
        self.momentum = momentum
        self.activation = nn.GELU()
        self.device = cfg.device

        self.bert_encoder = BertEncoder(bert_model_name=cfg.bert_model_name,
                                        trainable=cfg.fine_tune,
                                        output_size=cfg.bert_output_size,
                                        activation=self.activation)

        self.entity_span_extractor = CNNSpanExtractor(
            input_size=self.bert_encoder.get_output_dims(),
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

        self.entity_span_mlp = BertLinear(input_size=self.ent_output_size,
                                          output_size=self.ent_output_size,
                                          activation=self.activation,
                                          dropout=cfg.dropout)
        self.entity_span_decoder = VanillaSoftmaxDecoder(hidden_size=self.ent_output_size,
                                                         label_size=6)

        self.global_position_embedding = nn.Embedding(150, 200)
        self.global_position_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.masked_token_mlp = BertLinear(input_size=self.bert_encoder.get_output_dims() + 200,
                                           output_size=self.bert_encoder.get_output_dims(),
                                           activation=self.activation,
                                           dropout=cfg.dropout)
        self.masked_token_decoder = nn.Linear(self.bert_encoder.get_output_dims(),
                                              28996,
                                              bias=False)
        self.masked_token_decoder.weight.data.normal_(mean=0.0, std=0.02)
        self.masked_token_decoder_bias = nn.Parameter(torch.zeros(28996))

        self.position_embedding = nn.Embedding(7, self.position_embedding_dims)
        self.position_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.attention_encoder = PosAwareAttEncoder(self.ent_output_size,
                                                    self.bert_encoder.get_output_dims(),
                                                    2 * self.position_embedding_dims,
                                                    self.att_size,
                                                    activation=self.activation,
                                                    dropout=cfg.dropout)

        self.mlp_head1 = BertLinear(self.ent_output_size,
                                    self.bert_encoder.get_output_dims(),
                                    activation=self.activation,
                                    dropout=cfg.dropout)
        self.mlp_head2 = BertLinear(self.bert_encoder.get_output_dims(),
                                    self.bert_encoder.get_output_dims(),
                                    activation=self.activation,
                                    dropout=cfg.dropout)

        self.masked_token_loss = nn.CrossEntropyLoss()

    def forward(self, batch_inputs):
        """This function propagetes forwardly

        Arguments:
            batch_inputs {dict} -- batch inputs
        
        Returns:
            dict -- results
        """

        batch_seq_wordpiece_tokens_repr, batch_seq_cls_repr = self.bert_encoder(
            batch_inputs['wordpiece_tokens'])
        batch_seq_tokens_repr = batched_index_select(batch_seq_wordpiece_tokens_repr,
                                                     batch_inputs['wordpiece_tokens_index'])

        results = {}

        entity_feature = self.entity_span_extractor(batch_seq_tokens_repr,
                                                    batch_inputs['span_mention'])
        entity_feature = self.ent2hidden(entity_feature)

        subj_pos = torch.LongTensor([-1, 0, 1, 2, 3]) + 3
        obj_pos = torch.LongTensor([-3, -2, -1, 0, 1]) + 3

        if self.device > -1:
            subj_pos = subj_pos.cuda(device=self.device, non_blocking=True)
            obj_pos = obj_pos.cuda(device=self.device, non_blocking=True)

        subj_pos_emb = self.position_embedding(subj_pos)
        obj_pos_emb = self.position_embedding(obj_pos)
        pos_emb = torch.cat([subj_pos_emb, obj_pos_emb], dim=1).unsqueeze(0).repeat(
            batch_inputs['wordpiece_tokens_index'].size()[0], 1, 1)

        span_mention_attention_repr = self.attention_encoder(inputs=entity_feature,
                                                             query=batch_seq_cls_repr,
                                                             feature=pos_emb)
        results['span_mention_repr'] = self.mlp_head2(self.mlp_head1(span_mention_attention_repr))

        if self.momentum:
            return results

        zero_loss = torch.Tensor([0])
        zero_loss.requires_grad = True
        if self.device > -1:
            zero_loss = zero_loss.cuda(device=self.device, non_blocking=True)

        if sum([len(masked_index) for masked_index in batch_inputs['masked_index']]) == 0:
            results['masked_token_loss'] = zero_loss
        else:
            masked_wordpiece_tokens_repr = []
            all_masked_label = []
            for masked_index, masked_position, masked_label, seq_wordpiece_tokens_repr in zip(
                    batch_inputs['masked_index'], batch_inputs['masked_position'],
                    batch_inputs['masked_label'], batch_seq_wordpiece_tokens_repr):
                masked_index_tensor = torch.LongTensor(masked_index)
                masked_position_tensor = torch.LongTensor(masked_position)

                if self.device > -1:
                    masked_index_tensor = masked_index_tensor.cuda(device=self.device,
                                                                   non_blocking=True)
                    masked_position_tensor = masked_position_tensor.cuda(device=self.device,
                                                                         non_blocking=True)

                masked_wordpiece_tokens_repr.append(
                    torch.cat([
                        seq_wordpiece_tokens_repr[masked_index_tensor],
                        self.global_position_embedding(masked_position_tensor)
                    ],
                              dim=1))
                all_masked_label.extend(masked_label)

            masked_wordpiece_tokens_input = torch.cat(masked_wordpiece_tokens_repr, dim=0)
            masked_wordpiece_tokens_output = self.masked_token_decoder(
                self.masked_token_mlp(
                    masked_wordpiece_tokens_input)) + self.masked_token_decoder_bias

            all_masked_label_tensor = torch.LongTensor(all_masked_label)
            if self.device > -1:
                all_masked_label_tensor = all_masked_label_tensor.cuda(device=self.device,
                                                                       non_blocking=True)
            results['masked_token_loss'] = self.masked_token_loss(masked_wordpiece_tokens_output,
                                                                  all_masked_label_tensor)

        all_spans = []
        all_spans_label = []
        all_seq_tokens_reprs = []
        for spans, spans_label, seq_tokens_repr in zip(batch_inputs['spans'],
                                                       batch_inputs['spans_label'],
                                                       batch_seq_tokens_repr):
            all_spans.extend(spans)
            all_spans_label.extend(spans_label)
            all_seq_tokens_reprs.extend(seq_tokens_repr for _ in range(len(spans)))

        assert len(all_spans) == len(all_seq_tokens_reprs) and len(all_spans) == len(
            all_spans_label)

        if len(all_spans) == 0:
            results['span_loss'] = zero_loss
        else:
            if self.span_batch_size > 0:
                all_span_loss = []
                for idx in range(0, len(all_spans), self.span_batch_size):
                    batch_ents_tensor = torch.LongTensor(
                        all_spans[idx:idx + self.span_batch_size]).unsqueeze(1)
                    if self.device > -1:
                        batch_ents_tensor = batch_ents_tensor.cuda(device=self.device,
                                                                   non_blocking=True)

                    batch_seq_tokens_reprs = torch.stack(all_seq_tokens_reprs[idx:idx +
                                                                              self.span_batch_size])

                    batch_spans_feature = self.ent2hidden(
                        self.entity_span_extractor(batch_seq_tokens_reprs,
                                                   batch_ents_tensor).squeeze(1))

                    batch_spans_label = torch.LongTensor(all_spans_label[idx:idx +
                                                                         self.span_batch_size])
                    if self.device > -1:
                        batch_spans_label = batch_spans_label.cuda(device=self.device,
                                                                   non_blocking=True)

                    span_outputs = self.entity_span_decoder(
                        self.entity_span_mlp(batch_spans_feature), batch_spans_label)

                    all_span_loss.append(span_outputs['loss'])
                results['span_loss'] = sum(all_span_loss) / len(all_span_loss)
            else:
                all_spans_tensor = torch.LongTensor(all_spans).unsqueeze(1)
                if self.device > -1:
                    all_spans_tensor = all_spans_tensor.cuda(device=self.device, non_blocking=True)
                all_seq_tokens_reprs = torch.stack(all_seq_tokens_reprs)
                all_spans_feature = self.entity_span_extractor(all_seq_tokens_reprs,
                                                               all_spans_tensor).squeeze(1)

                all_spans_feature = self.ent2hidden(all_spans_feature)

                all_spans_label = torch.LongTensor(all_spans_label)
                if self.device > -1:
                    all_spans_label = all_spans_label.cuda(device=self.device, non_blocking=True)

                entity_typing_outputs = self.entity_span_decoder(
                    self.entity_span_mlp(all_spans_feature), all_spans_label)

                results['span_loss'] = entity_typing_outputs['loss']

        return results


class JointREPretrainedModel(nn.Module):
    """JointREPretrainedModel pretrained model based on MOCO.
    """
    def __init__(self, cfg):
        """This funciton constructs `JointREPretrainedModel` components

        Arguments:
            cfg {dict} -- config parameters for constructing multiple models
        """

        super().__init__()
        self.K = cfg.queue_size
        self.output_size = cfg.ent_mention_output_size
        self.qid = 0
        self.span_pair_queue = []
        self.context_pair_queue = []
        self.span_tmp_queue = []
        self.context_tmp_queue = []
        self.temperature = cfg.temperature
        self.gama = 0.999
        self.device = cfg.device

        self.span_pair_encoder = PretrainedSpanEncoder(cfg, momentum=False)
        self.momentum_span_pair_encoder = PretrainedSpanEncoder(cfg, momentum=True)

        self.contrastive_loss = nn.CrossEntropyLoss()

    def forward(self, batch_inputs):
        """This function propagetes forwardly

        Arguments:
            batch_inputs {dict} -- batch inputs
        
        Returns:
            dict -- results
        """

        zero_loss = torch.Tensor([0])
        zero_loss.requires_grad = True
        if self.device > -1:
            zero_loss = zero_loss.cuda(device=self.device, non_blocking=True)

        results = {}

        span_pair_inputs = {
            'wordpiece_tokens': batch_inputs['span_pair_wordpiece_tokens'],
            'wordpiece_tokens_index': batch_inputs['wordpiece_tokens_index'],
            'masked_index': batch_inputs['span_pair_masked_index'],
            'masked_position': batch_inputs['span_pair_masked_position'],
            'masked_label': batch_inputs['span_pair_masked_label'],
            'spans': batch_inputs['span_pair_spans'],
            'spans_label': batch_inputs['span_pair_spans_label'],
            'span_mention': batch_inputs['span_mention']
        }
        span_pair_results = self.span_pair_encoder(span_pair_inputs)
        momentum_span_pair_results = self.momentum_span_pair_encoder(span_pair_inputs)

        context_pair_inputs = {
            'wordpiece_tokens': batch_inputs['context_pair_wordpiece_tokens'],
            'wordpiece_tokens_index': batch_inputs['wordpiece_tokens_index'],
            'masked_index': batch_inputs['context_pair_masked_index'],
            'masked_position': batch_inputs['context_pair_masked_position'],
            'masked_label': batch_inputs['context_pair_masked_label'],
            'spans': batch_inputs['context_pair_spans'],
            'spans_label': batch_inputs['context_pair_spans_label'],
            'span_mention': batch_inputs['span_mention']
        }
        context_pair_results = self.span_pair_encoder(context_pair_inputs)
        momentum_context_pair_results = self.momentum_span_pair_encoder(context_pair_inputs)

        results['masked_token_loss'] = (span_pair_results['masked_token_loss'] +
                                        context_pair_results['masked_token_loss']) / 2
        results['span_loss'] = (span_pair_results['span_loss'] +
                                context_pair_results['span_loss']) / 2

        momentum_span_pair_results['span_mention_repr'] = momentum_span_pair_results[
            'span_mention_repr'].detach()
        momentum_context_pair_results['span_mention_repr'] = momentum_context_pair_results[
            'span_mention_repr'].detach()

        batch_size, repr_dims = span_pair_results['span_mention_repr'].size()

        if self.qid == 0:
            results['contrastive_loss'] = zero_loss
        else:
            s2c_positive_score = torch.bmm(
                span_pair_results['span_mention_repr'].view(batch_size, 1, repr_dims),
                momentum_context_pair_results['span_mention_repr'].view(batch_size, repr_dims,
                                                                        1)).view(batch_size, 1)
            s2c_negative_score = torch.mm(
                span_pair_results['span_mention_repr'].view(batch_size, repr_dims),
                torch.cat(self.context_pair_queue, dim=0).transpose(0, 1).contiguous())
            s2c_logits = torch.cat([s2c_positive_score, s2c_negative_score], dim=1)

            c2s_positive_score = torch.bmm(
                context_pair_results['span_mention_repr'].view(batch_size, 1, repr_dims),
                momentum_span_pair_results['span_mention_repr'].view(batch_size, repr_dims,
                                                                     1)).view(batch_size, 1)
            c2s_negative_score = torch.mm(
                context_pair_results['span_mention_repr'].view(batch_size, repr_dims),
                torch.cat(self.span_pair_queue, dim=0).transpose(0, 1).contiguous())
            c2s_logits = torch.cat([c2s_positive_score, c2s_negative_score], dim=1)

            labels = torch.zeros(batch_size, dtype=torch.long)
            if self.device > -1:
                labels = labels.cuda(device=self.device, non_blocking=True)

            results['contrastive_loss'] = (
                self.contrastive_loss(s2c_logits / self.temperature, labels) +
                self.contrastive_loss(c2s_logits / self.temperature, labels)) / 2

        self.span_tmp_queue.append(momentum_span_pair_results['span_mention_repr'])
        self.context_tmp_queue.append(momentum_context_pair_results['span_mention_repr'])

        return results

    @torch.no_grad()
    def momentum_update(self):
        for encoder_param, momentum_encoder_param in zip(
                self.span_pair_encoder.parameters(), self.momentum_span_pair_encoder.parameters()):
            momentum_encoder_param.data = momentum_encoder_param.data * self.gama + encoder_param.data * (
                1. - self.gama)

    def update_queue(self):
        for idx in range(len(self.span_tmp_queue)):
            if self.qid < self.K:
                self.span_pair_queue.append(self.span_tmp_queue[idx])
                self.context_pair_queue.append(self.context_tmp_queue[idx])
            else:
                self.span_pair_queue[self.qid % self.K] = self.span_tmp_queue[idx]
                self.context_pair_queue[self.qid % self.K] = self.context_tmp_queue[idx]

            self.qid += 1

        self.span_tmp_queue = []
        self.context_tmp_queue = []
