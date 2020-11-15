from collections import defaultdict
import os
import random
import logging

import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from utils.argparse import ConfigurationParer
from utils.prediction_outputs import print_predictions
from utils.eval import eval_file
from inputs.vocabulary import Vocabulary
from inputs.fields.token_field import TokenField
from inputs.fields.raw_token_field import RawTokenField
from inputs.fields.char_token_field import CharTokenField
from inputs.fields.map_token_field import MapTokenField
from inputs.instance import Instance
from inputs.datasets.dataset import Dataset
from inputs.dataset_readers.ace_reader_for_bert import ACEReaderForBert
from models.ent_models.joint_ent_model import JointEntModel
from models.ent_models.pipeline_ent_model import PipelineEntModel
from models.rel_models.context_rel_att_model import ConRelAttModel
from models.rel_models.ent_context_rel_att_model import EntConRelAttModel
from models.joint_models.joint_relation_extraction_model import JointREModel
from utils.nn_utils import get_n_trainable_parameters, load_weight_from_pretrained_model

logger = logging.getLogger(__name__)


def step(cfg, model, batch_inputs, device):
    batch_inputs["tokens"] = torch.LongTensor(batch_inputs["tokens"])
    batch_inputs["char_tokens"] = torch.LongTensor(batch_inputs["char_tokens"])
    if cfg.entity_model == 'joint':
        batch_inputs["entity_labels"] = torch.LongTensor(batch_inputs["entity_labels"])
    else:
        batch_inputs["entity_span_labels"] = torch.LongTensor(batch_inputs["entity_span_labels"])
    batch_inputs["tokens_mask"] = torch.LongTensor(batch_inputs["tokens_mask"])

    if cfg.embedding_model == 'bert':
        batch_inputs["wordpiece_tokens"] = torch.LongTensor(batch_inputs["wordpiece_tokens"])
        batch_inputs["wordpiece_tokens_index"] = torch.LongTensor(
            batch_inputs["wordpiece_tokens_index"])
        batch_inputs["wordpiece_segment_ids"] = torch.LongTensor(
            batch_inputs["wordpiece_segment_ids"])

    if device > -1:
        batch_inputs["tokens"] = batch_inputs["tokens"].cuda(device=device, non_blocking=True)
        batch_inputs["char_tokens"] = batch_inputs["char_tokens"].cuda(device=device,
                                                                       non_blocking=True)
        if cfg.entity_model == 'joint':
            batch_inputs["entity_labels"] = batch_inputs["entity_labels"].cuda(device=device,
                                                                               non_blocking=True)
        else:
            batch_inputs["entity_span_labels"] = batch_inputs["entity_span_labels"].cuda(
                device=device, non_blocking=True)
        batch_inputs["tokens_mask"] = batch_inputs["tokens_mask"].cuda(device=device,
                                                                       non_blocking=True)

        if cfg.embedding_model == 'bert':
            batch_inputs["wordpiece_tokens"] = batch_inputs["wordpiece_tokens"].cuda(
                device=device, non_blocking=True)
            batch_inputs["wordpiece_tokens_index"] = batch_inputs["wordpiece_tokens_index"].cuda(
                device=device, non_blocking=True)
            batch_inputs["wordpiece_segment_ids"] = batch_inputs["wordpiece_segment_ids"].cuda(
                device=device, non_blocking=True)

    outputs = model(batch_inputs)
    batch_outputs = []
    for sent_idx in range(len(batch_inputs['tokens_lens'])):
        sent_output = dict()
        sent_output['tokens'] = batch_inputs['tokens'][sent_idx].cpu().numpy()
        if cfg.entity_model == 'joint':
            sent_output["sequence_labels"] = batch_inputs["entity_labels"][sent_idx].cpu().numpy()
        else:
            sent_output["sequence_labels"] = batch_inputs["entity_span_labels"][sent_idx].cpu(
            ).numpy()
        sent_output['span2ent'] = batch_inputs['span2ent'][sent_idx]
        sent_output['span2rel'] = batch_inputs['span2rel'][sent_idx]
        sent_output['seq_len'] = batch_inputs['tokens_lens'][sent_idx]
        sent_output["sequence_label_preds"] = outputs['sequence_label_preds'][sent_idx].cpu().numpy(
        )
        sent_output['all_ent_span_preds'] = outputs['all_ent_span_preds'][sent_idx]
        sent_output['all_ent_preds'] = outputs['all_ent_preds'][sent_idx]
        sent_output['all_rel_preds'] = outputs['all_rel_preds'][sent_idx]
        batch_outputs.append(sent_output)

    return batch_outputs, outputs['ent_loss'], outputs['rel_loss']


def train(cfg, dataset, model):
    logger.info("Training starting...")

    for name, param in model.named_parameters():
        logger.info("{!r}: size: {} requires_grad: {}.".format(name, param.size(),
                                                               param.requires_grad))

    logger.info("Trainable parameters size: {}.".format(get_n_trainable_parameters(model)))

    parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_layer_lr = {}
    base_lr = cfg.bert_learning_rate
    for i in range(11, -1, -1):
        bert_layer_lr['.' + str(i) + '.'] = base_lr
        base_lr *= cfg.lr_decay_rate

    optimizer_grouped_parameters = []
    for name, param in parameters:
        params = {'params': [param], 'lr': cfg.learning_rate}
        if any(item in name for item in no_decay):
            params['weight_decay_rate'] = 0.0
        else:
            params['weight_decay_rate'] = cfg.adam_weight_decay_rate

        for bert_layer_name, lr in bert_layer_lr.items():
            if bert_layer_name in name:
                params['lr'] = lr
                break

        optimizer_grouped_parameters.append(params)

    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(cfg.adam_beta1, cfg.adam_beta2),
                      lr=cfg.learning_rate,
                      eps=cfg.adam_epsilon,
                      weight_decay=cfg.adam_weight_decay_rate,
                      correct_bias=False)

    total_train_steps = (dataset.get_dataset_size("train") +
                         cfg.train_batch_size * cfg.gradient_accumulation_steps -
                         1) / (cfg.train_batch_size * cfg.gradient_accumulation_steps) * cfg.epoches
    num_warmup_steps = int(cfg.warmup_rate * total_train_steps) + 1
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_train_steps)

    last_epoch = 1
    batch_id = 0
    best_f1 = 0.0
    early_stop_cnt = 0
    accumulation_steps = 0
    model.zero_grad()

    if cfg.embedding_model == 'word_char' or cfg.lstm_layers > 0:
        sort_key = "tokens"
    else:
        sort_key = None

    for epoch, batch in dataset.get_batch('train', cfg.train_batch_size, sort_key):

        if last_epoch != epoch or (batch_id != 0 and batch_id % cfg.validate_every == 0):
            if accumulation_steps != 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if epoch > cfg.pretrain_epoches:
                dev_f1 = dev(cfg, dataset, model)

                if dev_f1 > best_f1:
                    early_stop_cnt = 0
                    best_f1 = dev_f1
                    logger.info("Save model...")

                    # torch.save(
                    #     model.state_dict(),
                    #     open(
                    #         os.path.join(
                    #             cfg.train_model_dir,
                    #             "epoch_{}_batch_{}_{:04.2f}".format(last_epoch, batch_id,
                    #                                                 100 * best_f1)),
                    #         "wb",
                    #     ),
                    # )
                    torch.save(model.state_dict(), open(cfg.best_model_path, "wb"))
                elif last_epoch != epoch:
                    early_stop_cnt += 1
                    if early_stop_cnt > cfg.early_stop:
                        logger.info("Early Stop: best F1 score: {:6.2f}%".format(100 * best_f1))
                        break
        if epoch > cfg.epoches:
            torch.save(model.state_dict(), open(cfg.last_model_path, "wb"))
            logger.info("Training Stop: best F1 score: {:6.2f}%".format(100 * best_f1))
            break

        if last_epoch != epoch:
            batch_id = 0
            last_epoch = epoch

        model.train()
        batch_id += len(batch['tokens_lens'])
        batch['epoch'] = (epoch - 1)
        _, ent_loss, rel_loss = step(cfg, model, batch, cfg.device)
        loss = ent_loss + rel_loss
        if batch_id % cfg.logging_steps == 0:
            logger.info("Epoch: {} Batch: {} Loss: {} (Ent_loss: {} Rel_loss: {})".format(
                epoch, batch_id, loss.item(), ent_loss.item(), rel_loss.item()))

        if cfg.gradient_accumulation_steps > 1:
            loss /= cfg.gradient_accumulation_steps

        loss.backward()

        accumulation_steps = (accumulation_steps + 1) % cfg.gradient_accumulation_steps
        if accumulation_steps == 0:
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=cfg.gradient_clipping)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

    state_dict = torch.load(open(cfg.best_model_path, "rb"),
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    test(cfg, dataset, model)


def dev(cfg, dataset, model):
    logger.info("Validate starting...")
    model.zero_grad()

    all_outputs = []
    all_ent_loss = []
    all_rel_loss = []

    if cfg.embedding_model == 'word_char' or cfg.lstm_layers > 0:
        sort_key = "tokens"
    else:
        sort_key = None

    for _, batch in dataset.get_batch('dev', cfg.test_batch_size, sort_key):
        model.eval()
        with torch.no_grad():
            batch_outpus, ent_loss, rel_loss = step(cfg, model, batch, cfg.device)
        all_outputs.extend(batch_outpus)
        all_ent_loss.append(ent_loss.item())
        all_rel_loss.append(rel_loss.item())
    mean_ent_loss = np.mean(all_ent_loss)
    mean_rel_loss = np.mean(all_rel_loss)
    mean_loss = mean_ent_loss + mean_rel_loss

    logger.info("Validate Avgloss: {} (Ent_loss: {} Rel_loss: {})".format(
        mean_loss, mean_ent_loss, mean_rel_loss))

    dev_output_file = os.path.join(cfg.save_dir, "dev.output")
    print_predictions(all_outputs, dev_output_file, dataset.vocab,
                      'entity_labels' if cfg.entity_model == 'joint' else 'entity_span_labels')
    eval_metrics = ['token', 'span', 'ent', 'rel', 'exact-rel']
    token_score, span_score, ent_score, rel_score, exact_rel_score = eval_file(
        dev_output_file, eval_metrics)
    return ent_score + exact_rel_score


def test(cfg, dataset, model):
    logger.info("Testing starting...")
    model.zero_grad()

    all_outputs = []

    if cfg.embedding_model == 'word_char' or cfg.lstm_layers > 0:
        sort_key = "tokens"
    else:
        sort_key = None

    for _, batch in dataset.get_batch('test', cfg.test_batch_size, sort_key):
        model.eval()
        with torch.no_grad():
            batch_outpus, ent_loss, rel_loss = step(cfg, model, batch, cfg.device)
        all_outputs.extend(batch_outpus)
    test_output_file = os.path.join(cfg.save_dir, "test.output")
    print_predictions(all_outputs, test_output_file, dataset.vocab,
                      'entity_labels' if cfg.entity_model == 'joint' else 'entity_span_labels')
    eval_metrics = ['token', 'span', 'ent', 'rel', 'exact-rel']
    eval_file(test_output_file, eval_metrics)


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

    # define fields
    tokens = TokenField("tokens", "tokens", "tokens", True)
    raw_tokens = RawTokenField("raw_tokens", "tokens")
    char_tokens = CharTokenField("char_tokens", "char_tokens", "tokens", True)
    entity_span_labels = TokenField("entity_span_labels", "entity_span_labels",
                                    "entity_span_labels", True)
    entity_labels = TokenField("entity_labels", "entity_labels", "entity_labels", True)
    span2ent = MapTokenField("span2ent", "span2ent", "span2ent", True)
    span2rel = MapTokenField("span2rel", "span2rel", "span2rel", True)
    wordpiece_tokens = TokenField("wordpiece_tokens", "wordpiece", "wordpiece_tokens", False)
    wordpiece_tokens_index = RawTokenField("wordpiece_tokens_index", "wordpiece_tokens_index")
    wordpiece_segment_ids = RawTokenField("wordpiece_segment_ids", "wordpiece_segment_ids")

    fields = [
        tokens, raw_tokens, char_tokens, entity_span_labels, entity_labels, span2ent, span2rel
    ]

    if cfg.embedding_model == 'bert':
        fields.extend([wordpiece_tokens, wordpiece_tokens_index, wordpiece_segment_ids])

    # define counter and vocabulary
    counter = defaultdict(lambda: defaultdict(int))
    vocab = Vocabulary()

    # define instance
    train_instance = Instance(fields)
    dev_instance = Instance(fields)
    test_instance = Instance(fields)

    # define dataset reader
    max_len = {'tokens': cfg.max_sent_len, 'wordpiece_tokens': cfg.max_wordpiece_len}
    tokenizers = {}
    pretrained_vocab = {}
    if cfg.embedding_model == 'bert':
        bert_tokenizer = BertTokenizer.from_pretrained(cfg.bert_model_name,
                                                       do_lower_case=cfg.low_case)
        logger.info("Load bert tokenizer successfully.")
        tokenizers['wordpiece'] = bert_tokenizer.tokenize
        pretrained_vocab['wordpiece'] = bert_tokenizer.vocab
    ace_train_reader = ACEReaderForBert(cfg.train_file, False, cfg.low_case, max_len, tokenizers,
                                        cfg.entity_schema)
    ace_dev_reader = ACEReaderForBert(cfg.dev_file, False, cfg.low_case, max_len, tokenizers,
                                      cfg.entity_schema)
    ace_test_reader = ACEReaderForBert(cfg.test_file, False, cfg.low_case, max_len, tokenizers,
                                       cfg.entity_schema)

    # define dataset
    ace_dataset = Dataset("ACE2005")
    ace_dataset.add_instance("train",
                             train_instance,
                             ace_train_reader,
                             is_count=True,
                             is_train=True)
    ace_dataset.add_instance("dev", dev_instance, ace_dev_reader, is_count=True, is_train=False)
    ace_dataset.add_instance("test", test_instance, ace_test_reader, is_count=True, is_train=False)

    min_count = {
        "tokens": 1,
        "char_tokens": 1,
        "entity_span_labels": 1,
        "entity_labels": 1,
        "span2ent": 1,
        "span2rel": 1
    }
    no_pad_namespace = ["raw_tokens", "span2ent", "span2rel"]
    no_unk_namespace = [
        "raw_tokens", "entity_span_labels", "entity_labels", "span2ent", "span2rel",
        "wordpiece_tokens_index", "wordpiece_segment_ids"
    ]
    tokens_to_add = {"span2ent": ["None"], "span2rel": ["None"]}
    contain_pad_namespace = {"wordpiece": "[PAD]"}
    contain_unk_namespace = {"wordpiece": "[UNK]"}
    ace_dataset.build_dataset(vocab=vocab,
                              counter=counter,
                              min_count=min_count,
                              pretrained_vocab=pretrained_vocab,
                              no_pad_namespace=no_pad_namespace,
                              no_unk_namespace=no_unk_namespace,
                              contain_pad_namespace=contain_pad_namespace,
                              contain_unk_namespace=contain_unk_namespace,
                              tokens_to_add=tokens_to_add)

    if cfg.test:
        vocab = Vocabulary.load(cfg.vocabulary_file)
    else:
        vocab.save(cfg.vocabulary_file)

    # entity model
    if cfg.entity_model == 'joint':
        ent_model = JointEntModel(cfg, vocab)
        rel_model = EntConRelAttModel(cfg, vocab, ent_model.get_hidden_size())
    else:
        ent_model = PipelineEntModel(cfg, vocab)
        rel_model = ConRelAttModel(cfg, vocab, ent_model.get_hidden_size(),
                                   ent_model.get_ent_span_feature_size())

    # joint model
    model = JointREModel(cfg=cfg, ent_model=ent_model, rel_model=rel_model, vocab=vocab)

    # continue training
    if cfg.continue_training and os.path.exists(cfg.last_model_path):
        state_dict = torch.load(open(cfg.last_model_path, 'rb'),
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        logger.info("Loading last training model {} successfully.".format(cfg.last_model_path))

    if cfg.test and os.path.exists(cfg.best_model_path):
        state_dict = torch.load(open(cfg.best_model_path, 'rb'),
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        logger.info("Loading best training model {} successfully for testing.".format(
            cfg.best_model_path))

    if cfg.fine_tune and os.path.exists(cfg.pretrained_model_path):
        state_dict = torch.load(open(cfg.pretrained_model_path, 'rb'),
                                map_location=lambda storage, loc: storage)
        load_weight_from_pretrained_model(model, state_dict, prefix="span_pair_encoder.")
        # load_weight_from_pretrained_model(model, state_dict, prefix="momentum_span_pair_encoder.")
        logger.info("Loading pretrained model {} successfully for fine-tuning.".format(
            cfg.pretrained_model_path))

        model.rel_model.context_span_extractor.load_state_dict(
            model.ent_model.cnn_ent_model.entity_span_extractor.state_dict())
        model.rel_model.context2hidden.load_state_dict(
            model.ent_model.cnn_ent_model.ent2hidden.state_dict())

    if cfg.device > -1:
        model.cuda(device=cfg.device)

    if cfg.test:
        test(cfg, ace_dataset, model)
    else:
        train(cfg, ace_dataset, model)


if __name__ == '__main__':
    main()
