import json
import json
import random
import math
import itertools

import numpy as np
import fire
from transformers import BertTokenizer


def contrastive_loss_preprocess(source_file, target_file, vocab_file):
    """contrastive_loss_preprocess preprocess data for contrastive loss pre-training

    Args:
        source_file (str): source file path
        target_file (str): target file path
        vocab_file (str): vocabulary file of BERT
    """

    p = 0.2
    len_distrib = [p * (1 - p)**(i - 1) for i in range(1, 11)]
    len_distrib = [x / (sum(len_distrib)) for x in len_distrib]

    lens = list(range(1, 11))

    # intra-span permutation
    permutations = list(itertools.permutations([0, 1, 2], 3))
    parts = [[0, 1], [1, -1], [-1, 1000]]

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    print("Load bert tokenizer successfully.")

    cnt = 0
    all_words = 0
    all_masked_cnt = 0
    normal_permuted_cnt = 0
    other_permuted_cnt = 0

    with open(source_file, 'r', encoding='utf-8') as fin, open(target_file,
                                                               'w') as fout, open(vocab_file,
                                                                                  'r') as vocab_fin:
        vocab = json.load(vocab_fin)
        span_ids = list(range(5))
        for line in fin:
            tokens = line.strip().split(' ')
            length = len(tokens)
            if length > 128 or length < 2:
                continue
            sampled_spans = []
            sampled_index = [0] * length
            n_sample_tokens = 0
            tmp = 0
            while tmp < 1000 and len(sampled_spans) < 2 and n_sample_tokens < length:
                tmp += 1
                span_len = int(np.random.choice(lens, p=len_distrib))
                start = int(np.random.choice(range(length)))
                if start + span_len > length:
                    continue
                valid = True
                for i in range(start, start + span_len):
                    if sampled_index[i] == 1:
                        valid = False
                        break
                if not valid:
                    continue
                sampled_spans.append([start, start + span_len])
                n_sample_tokens += span_len
                for i in range(start, start + span_len):
                    sampled_index[i] = 1
            if len(sampled_spans) < 2:
                continue
            sent = {}
            span_pair = [sampled_spans[0], sampled_spans[1]
                         ] if sampled_spans[0][1] <= sampled_spans[1][0] else [
                             sampled_spans[1], sampled_spans[0]
                         ]
            context_pair = [[0, span_pair[0][0]], [span_pair[0][1], span_pair[1][0]],
                            [span_pair[1][1], length]]

            spans = span_pair + context_pair
            np.random.shuffle(span_ids)
            sent['span_pair_spans'] = []
            sent['context_pair_spans'] = []
            sent['span_pair_spans_label'] = []
            sent['context_pair_spans_label'] = []
            for i in span_ids:
                span = spans[i]
                if span[1] - span[0] < 3:
                    continue
                permutation_label = int(np.random.choice(6))
                if span in span_pair:
                    sent['span_pair_spans'].append(span)
                    sent['span_pair_spans_label'].append(permutation_label)
                else:
                    sent['context_pair_spans'].append(span)
                    sent['context_pair_spans_label'].append(permutation_label)

                permutated_tokens = []
                for idx in permutations[permutation_label]:
                    for token in tokens[span[0]:span[1]][parts[idx][0]:parts[idx][1]]:
                        permutated_tokens.append(token)

                for idx in range(span[0], span[1]):
                    tokens[idx] = permutated_tokens[idx - span[0]]

                if len(sent['span_pair_spans']) + len(sent['context_pair_spans']) >= 2:
                    break

            maksed_tokens = []
            wordpiece_tokens = ['[CLS]']
            wordpiece_tokens_index = []
            cur_pos = 1
            for idx, token in enumerate(tokens):
                tokenized_token = list(bert_tokenizer.tokenize(token))
                wordpiece_tokens.extend(tokenized_token)
                tokenized_token_length = len(tokenized_token)
                wordpiece_tokens_index.append([cur_pos, cur_pos + tokenized_token_length])
                cur_pos += tokenized_token_length
                if tokenized_token_length > 1:
                    maksed_tokens.append(idx)
            wordpiece_tokens.append('[SEP]')

            if len(wordpiece_tokens) > 128:
                continue

            max_predictions_per_seq = 20
            max_n_masked = min(max_predictions_per_seq,
                               math.ceil(0.15 * (len(wordpiece_tokens) - 2)))
            random.shuffle(maksed_tokens)
            n_masked = 0

            sent['span_pair_masked_index'] = []
            sent['span_pair_masked_label'] = []
            sent['span_pair_masked_position'] = []

            sent['context_pair_masked_index'] = []
            sent['context_pair_masked_label'] = []
            sent['context_pair_masked_position'] = []

            for idx in maksed_tokens:
                if (span_pair[0][0] <= idx < span_pair[0][1]) or (span_pair[1][0] <= idx <
                                                                  span_pair[1][1]):
                    for i in range(wordpiece_tokens_index[idx][0] + 1,
                                   wordpiece_tokens_index[idx][1]):
                        sent['span_pair_masked_index'].append(wordpiece_tokens_index[idx][0])
                        sent['span_pair_masked_label'].append(vocab[wordpiece_tokens[i]])
                        sent['span_pair_masked_position'].append(i - wordpiece_tokens_index[idx][0])
                        wordpiece_tokens[i] = '[MASK]'
                else:
                    for i in range(wordpiece_tokens_index[idx][0] + 1,
                                   wordpiece_tokens_index[idx][1]):
                        sent['context_pair_masked_index'].append(wordpiece_tokens_index[idx][0])
                        sent['context_pair_masked_label'].append(vocab[wordpiece_tokens[i]])
                        sent['context_pair_masked_position'].append(i -
                                                                    wordpiece_tokens_index[idx][0])
                        wordpiece_tokens[i] = '[MASK]'

                n_masked += (wordpiece_tokens_index[idx][1] - wordpiece_tokens_index[idx][0] - 1)
                if n_masked >= max_n_masked:
                    break

            sent['span_pair_wordpiece_tokens'] = [vocab['[CLS]']]
            sent['context_pair_wordpiece_tokens'] = [vocab['[CLS]']]

            for idx in range(1, len(wordpiece_tokens) - 1):
                if (span_pair[0][0] <= idx < span_pair[0][1]) or (span_pair[1][0] <= idx <
                                                                  span_pair[1][1]):
                    sent['span_pair_wordpiece_tokens'].append(vocab[wordpiece_tokens[idx]])
                    sent['context_pair_wordpiece_tokens'].append(vocab['[MASK]'])
                else:
                    sent['context_pair_wordpiece_tokens'].append(vocab[wordpiece_tokens[idx]])
                    sent['span_pair_wordpiece_tokens'].append(vocab['[MASK]'])

            sent['span_pair_wordpiece_tokens'].append(vocab['[SEP]'])
            sent['context_pair_wordpiece_tokens'].append(vocab['[SEP]'])
            span_idx = np.random.choice(2)
            sent['span_wordpiece_tokens'] = [
                vocab['[CLS]'],
                sent['span_pair_wordpiece_tokens'][span_pair[span_idx][0]:span_pair[span_idx][1]],
                vocab['[SEP]']
            ]
            sent['wordpiece_tokens_index'] = [item[0] for item in wordpiece_tokens_index]
            sent['span_mention'] = [
                context_pair[0], span_pair[0], context_pair[1], span_pair[1], context_pair[2]
            ]

            print(json.dumps(sent), file=fout)

            cnt += 1
            all_words += len(tokens)
            all_masked_cnt += n_masked
            for label in sent['span_pair_spans_label'] + sent['context_pair_spans_label']:
                if label == 0:
                    normal_permuted_cnt += 1
                else:
                    other_permuted_cnt += 1

            if cnt % 10000 == 0:
                print("Processed {} sentences: words {} masked_tokens {} span0 {} other_spans {}.".
                      format(cnt, all_words, all_masked_cnt, normal_permuted_cnt,
                             other_permuted_cnt))


if __name__ == '__main__':
    fire.Fire({'contrastive_loss_preprocess': contrastive_loss_preprocess})
