import torch
import torch.nn as nn
import numpy as np


class JointREModel(nn.Module):
    """This class combines entity model and relation
    model to solve realtion extraction task, which is a pipeline task.
    """
    def __init__(self, cfg, ent_model, rel_model, vocab):
        """This funciton decides `JointREModel` components

        Arguments:
            cfg {dict} -- config parameters for constructing multiple models
            ent_mdoel {nn.Module} -- entity model
            rel_model {nn.Module} -- relation model
            vocab {dict} -- vocabulary
        """

        super().__init__()
        self.ent_model = ent_model
        self.rel_model = rel_model
        self.vocab = vocab
        self.schedule_k = cfg.schedule_k
        self.device = cfg.device
        self.pretrain_epoches = cfg.pretrain_epoches

    def forward(self, batch_inputs):
        """This function propagetes forwardly

        Arguments:
            batch_inputs {dict} -- batch inputs

        Returns:
            dict -- results
        """

        outputs = {}

        ent_model_outputs = self.ent_model(batch_inputs)
        outputs['sequence_label_preds'] = ent_model_outputs['sequence_label_preds']
        outputs['ent_loss'] = ent_model_outputs['ent_loss']
        outputs['all_ent_span_preds'] = ent_model_outputs['all_ent_span_preds']
        outputs['all_ent_preds'] = ent_model_outputs['all_ent_preds']

        batch_inputs['ent_preds'] = ent_model_outputs['all_ent_preds']

        if self.training and self.schedule_k > 0 and 'epoch' in batch_inputs:
            if batch_inputs['epoch'] > self.pretrain_epoches:
                schedule_p = self.schedule_k / (self.schedule_k + np.exp(
                    (batch_inputs['epoch'] - self.pretrain_epoches) / self.schedule_k))
                scheduled_ent_preds = [
                    gold if np.random.random() < schedule_p else pred
                    for gold, pred in zip(batch_inputs['span2ent'], batch_inputs['ent_preds'])
                ]
            else:
                scheduled_ent_preds = batch_inputs['span2ent']
        else:
            scheduled_ent_preds = batch_inputs['ent_preds']

        all_candi_rels, all_candi_rel_labels = self.generate_all_candi_rels(
            scheduled_ent_preds, batch_inputs['span2rel'])

        batch_inputs['all_candi_rels'] = all_candi_rels
        batch_inputs['all_candi_rel_labels'] = all_candi_rel_labels

        outputs['all_candi_rels'] = all_candi_rels

        if sum(len(ent_pred) for ent_pred in ent_model_outputs['all_ent_preds']) != 0 and sum(
                len(candi_rels) for candi_rels in batch_inputs['all_candi_rels']) != 0:
            rel_model_ouputs = self.rel_model(batch_inputs)
            rel_preds = self.get_rel_preds(batch_inputs, rel_model_ouputs)

            outputs['all_rel_preds'] = rel_preds
            outputs['rel_loss'] = rel_model_ouputs['rel_loss']
        else:
            zero_loss = torch.Tensor([0])
            zero_loss.requires_grad = True
            if self.device > -1:
                zero_loss = zero_loss.cuda(device=self.device, non_blocking=True)

            outputs['all_rel_preds'] = [{} for _ in range(len(batch_inputs['tokens_lens']))]
            outputs['rel_loss'] = zero_loss[0]

        return outputs

    def generate_all_candi_rels(self, span2ent, span2rel):
        """This function genarates all candidate relations

        Arguments:
            ent_preds {list} -- span2ent list
            span2rel {list} -- span2rel list

        Returns:
            list -- all_candi_rel_list, all_candi_rel_label_list
        """

        all_candi_rels = []
        all_candi_rel_labels = []

        for s2e, s2r in zip(span2ent, span2rel):
            candi_rel_set = set()
            cur_candi_rels = []
            cur_candi_rel_labels = []

            for ent1_span in s2e:
                for ent2_span in s2e:
                    if ent1_span[0] >= ent2_span[0]:
                        continue
                    candi_rel_set.add((ent1_span, ent2_span))

            if self.training:
                for ent1_span, ent2_span in s2r:
                    candi_rel_set.add((ent1_span, ent2_span))

            for ent1_span, ent2_span in candi_rel_set:
                if (ent1_span, ent2_span) in s2r:
                    rel_label = s2r[(ent1_span, ent2_span)]
                else:
                    rel_label = self.vocab.get_token_index('None', 'span2rel')
                cur_candi_rels.append((ent1_span, ent2_span))
                cur_candi_rel_labels.append(rel_label)

            all_candi_rels.append(cur_candi_rels)
            all_candi_rel_labels.append(cur_candi_rel_labels)

        return all_candi_rels, all_candi_rel_labels

    def get_rel_preds(self, batch_inputs, rel_model_outputs):
        """This funtion gets relation predictions from relation model outputs
        
        Arguments:
            batch_inputs {dict} -- batch input data
            rel_model_outputs {dict} -- relation model outputs
        
        Returns:
            list -- relation predictions
        """

        rel_preds = []
        candi_rel_cnt = 0
        for rels in batch_inputs['all_candi_rels']:
            cur_rels_num = len(rels)
            rel_pred = {}
            for rel, pred in zip(
                    rels,
                    rel_model_outputs['rel_preds'][candi_rel_cnt:candi_rel_cnt + cur_rels_num]):
                rel_pred_label = self.vocab.get_token_from_index(pred.item(), 'span2rel')
                if rel_pred_label != 'None':
                    rel_pred[rel] = rel_pred_label
            rel_preds.append(rel_pred)
            candi_rel_cnt += cur_rels_num

        return rel_preds
