from collections import defaultdict
import os
import random
import logging
import json

import torch
import torch.nn as nn
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.argparse import ConfigurationParer
from models.pretrain_models.joint_relation_extraction_pretrained_model import JointREPretrainedModel
from utils.nn_utils import get_n_trainable_parameters

logger = logging.getLogger(__name__)


def step(model, batch_inputs, device):
    fields = [
        'span_pair_wordpiece_tokens', 'context_pair_wordpiece_tokens', 'wordpiece_tokens_index',
        'span_mention'
    ]

    for field in fields:
        if field in batch_inputs:
            batch_inputs[field] = torch.LongTensor(batch_inputs[field])
            if device > -1:
                batch_inputs[field] = batch_inputs[field].cuda(device=device, non_blocking=True)

    outputs = model(batch_inputs)
    return outputs['masked_token_loss'], outputs['span_loss'], outputs['contrastive_loss']


def train(cfg, model):
    logger.info("Training starting...")

    for name, param in model.named_parameters():
        logger.info("{!r}: size: {} requires_grad: {}.".format(name, param.size(),
                                                               param.requires_grad))

    logger.info("Trainable parameters size: {}.".format(get_n_trainable_parameters(model)))

    parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
        [param for name, param in parameters if not any(item in name for item in no_decay)],
        'weight_decay_rate':
        cfg.adam_weight_decay_rate
    }, {
        'params': [param for name, param in parameters if any(item in name for item in no_decay)],
        'weight_decay_rate':
        0.0
    }]

    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(cfg.adam_beta1, cfg.adam_beta2),
                      lr=cfg.learning_rate,
                      eps=cfg.adam_epsilon,
                      weight_decay=cfg.adam_weight_decay_rate,
                      correct_bias=False)

    total_train_steps = 400000
    num_warmup_steps = 18000
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_train_steps)
    step_cnt = 0

    model.zero_grad()
    model.train()

    data_file_index = 0
    data_file_n = len(cfg.pretrain_file_list)
    data_file = open(cfg.pretrain_file_list[data_file_index], 'r')

    pad_namespace = [
        'span_pair_wordpiece_tokens', 'context_pair_wordpiece_tokens', 'wordpiece_tokens_index'
    ]

    batch_loss = defaultdict(list)

    while True:
        batch = defaultdict(list)
        for _ in range(cfg.train_batch_size):
            try:
                line = next(data_file)
            except StopIteration:
                data_file.close()
                data_file_index = (data_file_index + 1) % data_file_n
                data_file = open(cfg.pretrain_file_list[data_file_index], 'r')
                line = next(data_file)
            sent = json.loads(line.strip())
            for field, value in sent.items():
                batch[field].append(value)
                if field in pad_namespace:
                    batch[field + '_lens'].append(len(value))
        for field in pad_namespace:
            max_len = max(batch[field + '_lens'])
            for idx in range(len(batch[field + '_lens'])):
                batch[field][idx].extend([0] * (max_len - batch[field + '_lens'][idx]))

        masked_token_loss, span_loss, contrastive_loss = step(model, batch, cfg.device)
        loss = 0.5 * masked_token_loss + 1.0 * span_loss + 0.3 * contrastive_loss

        step_cnt += 1

        # if step_cnt <= cfg.queue_size:
        #     continue

        batch_loss['loss'].append(loss.item())
        batch_loss['masked_token_loss'].append(masked_token_loss.item())
        batch_loss['span_loss'].append(span_loss.item())
        batch_loss['contrastive_loss'].append(contrastive_loss.item())

        loss /= cfg.gradient_accumulation_steps
        loss.backward()

        if step_cnt % cfg.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=cfg.gradient_clipping)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            with torch.no_grad():
                model.momentum_update()
                model.update_queue()

            logger.info(
                "Step: {} Loss: {:.8f} (Masked_token_loss: {:.8f} Span_loss: {:.8f} Contrastive_loss: {:.8f})"
                .format(step_cnt // cfg.gradient_accumulation_steps,
                        sum(batch_loss['loss']) / cfg.gradient_accumulation_steps,
                        sum(batch_loss['masked_token_loss']) / cfg.gradient_accumulation_steps,
                        sum(batch_loss['span_loss']) / cfg.gradient_accumulation_steps,
                        sum(batch_loss['contrastive_loss']) / cfg.gradient_accumulation_steps))

            batch_loss.clear()

        if step_cnt % (1000 * cfg.gradient_accumulation_steps) == 0:
            torch.save(
                model.state_dict(),
                open(
                    cfg.pretrained_model_path + '_' +
                    str(step_cnt // (1000 * cfg.gradient_accumulation_steps)) + 'k', "wb"))

        if step_cnt == total_train_steps * cfg.gradient_accumulation_steps:
            break

    torch.save(model.state_dict(), open(cfg.pretrained_model_path, "wb"))
    logger.info("Pretraining Completed!")


def main():
    # config settings
    parser = ConfigurationParer()
    parser.add_save_cfgs()
    parser.add_data_cfgs()
    parser.add_model_cfgs()
    parser.add_optimizer_cfgs()
    parser.add_run_cfgs()

    cfg = parser.parse_args()
    logger.info(parser.format_values())

    # set random seed
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.device > -1 and not torch.cuda.is_available():
        logger.error('config conflicts: no gpu available, use cpu for training.')
        cfg.device = -1
    if cfg.device > -1:
        torch.cuda.manual_seed(cfg.seed)

    # joint model
    model = JointREPretrainedModel(cfg)

    # continue training
    if cfg.continue_training and os.path.exists(cfg.pretrained_model_path):
        state_dict = torch.load(open(cfg.pretrained_model_path, 'rb'),
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loading last training model {} successfully.".format(
            cfg.pretrained_model_path))

    if cfg.device > -1:
        model.cuda(device=cfg.device)

    train(cfg, model)


if __name__ == '__main__':
    main()
