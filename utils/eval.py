from collections import defaultdict
import logging
import sys

logger = logging.getLogger(__name__)


class EvalCounter():
    """EvalCounter evaluating counter class
    """
    def __init__(self):
        self.pred_correct_cnt = 0
        self.correct_cnt = 0
        self.pred_cnt = 0

        self.pred_correct_types_cnt = defaultdict(int)
        self.correct_types_cnt = defaultdict(int)
        self.pred_types_cnt = defaultdict(int)


def eval_file(file_path, eval_metrics):
    """eval_file evaluates results file

    Args:
        file_path (str): file path
        eval_metrics (list): eval metrics

    Returns:
        tuple: results
    """

    with open(file_path, 'r') as fin:
        sents = []
        metric2labels = {
            'token': ['Sequence-Label-True', 'Sequence-Label-Pred'],
            'span': ['Ent-Span-Pred'],
            'ent': ['Ent-True', 'Ent-Pred'],
            'rel': ['Rel-True', 'Rel-Pred'],
            'exact-rel': ['Rel-True', 'Rel-Pred']
        }
        labels = set()
        for metric in eval_metrics:
            labels.update(metric2labels[metric])
        label2idx = {label: idx for idx, label in enumerate(labels)}
        sent = [[] for _ in range(len(labels))]
        for line in fin:
            line = line.strip('\r\n')
            if line == "":
                sents.append(sent)
                sent = [[] for _ in range(len(labels))]
            else:
                words = line.split('\t')
                if words[0] in ['Sequence-Label-True', 'Sequence-Label-Pred']:
                    sent[label2idx[words[0]]].extend(words[1].split(' '))
                elif words[0] in ['Ent-Span-Pred']:
                    sent[label2idx[words[0]]].append(eval(words[1]))
                elif words[0] in ['Ent-True', 'Ent-Pred']:
                    sent[label2idx[words[0]]].append([words[1], eval(words[2])])
                elif words[0] in ['Rel-True', 'Rel-Pred']:
                    sent[label2idx[words[0]]].append([words[1], eval(words[2]), eval(words[3])])
        sents.append(sent)

    counters = {metric: EvalCounter() for metric in eval_metrics}

    for sent in sents:
        evaluate(sent, counters, label2idx)

    results = []

    logger.info("-------------------------------START-----------------------------------")

    for metric, counter in counters.items():
        logger.info("------------------------------{}------------------------------".format(metric))
        score = report(counter)
        results += [score]

    logger.info("-------------------------------END-----------------------------------")

    return results


def evaluate(sent, counters, label2idx):
    """evaluate calculates counters
    
    Arguments:
        sent {list} -- line

    Args:
        sent (list): line
        counters (dict): counters
        label2idx (dict): label -> idx dict
    """

    # evaluate token
    if 'token' in counters:
        for token1, token2 in zip(sent[label2idx['Sequence-Label-True']],
                                  sent[label2idx['Sequence-Label-Pred']]):
            if token1 != 'O':
                counters['token'].correct_cnt += 1
                counters['token'].correct_types_cnt[token1] += 1
                counters['token'].pred_correct_types_cnt[token1] += 0
            if token2 != 'O':
                counters['token'].pred_cnt += 1
                counters['token'].pred_types_cnt[token2] += 1
                counters['token'].pred_correct_types_cnt[token2] += 0
            if token1 == token2 and token1 != 'O':
                counters['token'].pred_correct_cnt += 1
                counters['token'].pred_correct_types_cnt[token1] += 1

    # evaluate span & entity
    correct_ent2idx = defaultdict(set)
    correct_span2ent = dict()
    correct_span = set()
    for ent, span in sent[label2idx['Ent-True']]:
        correct_span.add(span)
        correct_span2ent[span] = ent
        correct_ent2idx[ent].add(span)

    pred_ent2idx = defaultdict(set)
    pred_span2ent = dict()
    for ent, span in sent[label2idx['Ent-Pred']]:
        pred_span2ent[span] = ent
        pred_ent2idx[ent].add(span)

    if 'span' in counters:
        pred_span = set(sent[label2idx['Ent-Span-Pred']])
        counters['span'].correct_cnt += len(correct_span)
        counters['span'].pred_cnt += len(pred_span)
        counters['span'].pred_correct_cnt += len(correct_span & pred_span)

    if 'ent' in counters:
        all_ents = set(correct_ent2idx) | set(pred_ent2idx)
        for ent in all_ents:
            counters['ent'].correct_cnt += len(correct_ent2idx[ent])
            counters['ent'].correct_types_cnt[ent] += len(correct_ent2idx[ent])
            counters['ent'].pred_cnt += len(pred_ent2idx[ent])
            counters['ent'].pred_types_cnt[ent] += len(pred_ent2idx[ent])
            pred_correct_cnt = len(correct_ent2idx[ent] & pred_ent2idx[ent])
            counters['ent'].pred_correct_cnt += pred_correct_cnt
            counters['ent'].pred_correct_types_cnt[ent] += pred_correct_cnt

    # evaluate relation
    if 'rel' in counters:
        correct_rel2idx = defaultdict(set)
        for rel, span1, span2 in sent[label2idx['Rel-True']]:
            if span1 not in correct_span2ent or span2 not in correct_span2ent:
                continue
            correct_rel2idx[rel].add((span1, span2))

        pred_rel2idx = defaultdict(set)
        for rel, span1, span2 in sent[label2idx['Rel-Pred']]:
            if span1 not in pred_span2ent or span2 not in pred_span2ent:
                continue
            pred_rel2idx[rel].add((span1, span2))

        all_rels = set(correct_rel2idx) | set(pred_rel2idx)
        for rel in all_rels:
            counters['rel'].correct_cnt += len(correct_rel2idx[rel])
            counters['rel'].correct_types_cnt[rel] += len(correct_rel2idx[rel])
            counters['rel'].pred_cnt += len(pred_rel2idx[rel])
            counters['rel'].pred_types_cnt[rel] += len(pred_rel2idx[rel])
            pred_correct_rel_cnt = len(correct_rel2idx[rel] & pred_rel2idx[rel])
            counters['rel'].pred_correct_cnt += pred_correct_rel_cnt
            counters['rel'].pred_correct_types_cnt[rel] += pred_correct_rel_cnt

    # exact relation evaluation
    if 'exact-rel' in counters:
        exact_correct_rel2idx = defaultdict(set)
        for rel, span1, span2 in sent[label2idx['Rel-True']]:
            if span1 not in correct_span2ent or span2 not in correct_span2ent:
                continue
            exact_correct_rel2idx[rel].add(
                (span1, correct_span2ent[span1], span2, correct_span2ent[span2]))

        exact_pred_rel2idx = defaultdict(set)
        for rel, span1, span2 in sent[label2idx['Rel-Pred']]:
            if span1 not in pred_span2ent or span2 not in pred_span2ent:
                continue
            exact_pred_rel2idx[rel].add((span1, pred_span2ent[span1], span2, pred_span2ent[span2]))

        all_exact_rels = set(exact_correct_rel2idx) | set(exact_pred_rel2idx)
        for rel in all_exact_rels:
            counters['exact-rel'].correct_cnt += len(exact_correct_rel2idx[rel])
            counters['exact-rel'].correct_types_cnt[rel] += len(exact_correct_rel2idx[rel])
            counters['exact-rel'].pred_cnt += len(exact_pred_rel2idx[rel])
            counters['exact-rel'].pred_types_cnt[rel] += len(exact_pred_rel2idx[rel])
            exact_pred_correct_rel_cnt = len(exact_correct_rel2idx[rel] & exact_pred_rel2idx[rel])
            counters['exact-rel'].pred_correct_cnt += exact_pred_correct_rel_cnt
            counters['exact-rel'].pred_correct_types_cnt[rel] += exact_pred_correct_rel_cnt


def report(counter):
    """report prints evaluation results

    Args:
        counter (EvalCounter): counter

    Returns:
        float: f1 score
    """

    p, r, f = calculate_metrics(counter.pred_correct_cnt, counter.pred_cnt, counter.correct_cnt)
    logger.info("truth cnt: {} pred cnt: {} correct cnt: {}".format(counter.correct_cnt,
                                                                    counter.pred_cnt,
                                                                    counter.pred_correct_cnt))
    logger.info("precision: {:6.2f}%".format(100 * p))
    logger.info("recall: {:6.2f}%".format(100 * r))
    logger.info("f1: {:6.2f}%".format(100 * f))

    score = f

    for type in counter.pred_correct_types_cnt:
        p, r, f = calculate_metrics(counter.pred_correct_types_cnt[type],
                                    counter.pred_types_cnt[type], counter.correct_types_cnt[type])
        logger.info("--------------------------------------------------")
        logger.info("type: {:17}".format(type))
        logger.info("truth cnt: {} pred cnt: {} correct cnt: {}".format(
            counter.correct_types_cnt[type], counter.pred_types_cnt[type],
            counter.pred_correct_types_cnt[type]))
        logger.info("precision: {:6.2f}%".format(100 * p))
        logger.info("recall: {:6.2f}%".format(100 * r))
        logger.info("f1: {:6.2f}%".format(100 * f))

    return score


def calculate_metrics(pred_correct_cnt, pred_cnt, correct_cnt):
    """calculate_metrics calculates metrics: precision, recall, f1-score.

    Args:
        pred_correct_cnt (int): the number of corrected prediction
        pred_cnt (int): the number of prediction
        correct_cnt (int): the numbert of truth

    Returns:
        tuple: precision, recall, f1-score
    """

    tp, fp, fn = pred_correct_cnt, pred_cnt - pred_correct_cnt, correct_cnt - pred_correct_cnt
    p = 0 if tp + fp == 0 else (tp / (tp + fp))
    r = 0 if tp + fn == 0 else (tp / (tp + fn))
    f = 0 if p + r == 0 else (2 * p * r / (p + r))
    return p, r, f


if __name__ == '__main__':
    eval_file(sys.argv[1])
