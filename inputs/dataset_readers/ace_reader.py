import json
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ACEReader():
    """ACEReader text data reader that preprocesses data on ACE dataset.
    """
    def __init__(self,
                 file_path,
                 is_test=False,
                 low_case=True,
                 max_len=dict(),
                 entity_schema='BIEOU'):
        """This function defines file path and indicates training or testing
        
        Arguments:
            file_path {str} -- file path

        Keyword Arguments:
            is_test {bool} -- indicate training or testing (default: {False})
            low_case {bool} -- transform word to low case (default: {True})
            max_len {dict} -- max length for some namespace (default: {dict()})
            entity_schema {str} -- entity tagging method (default: {'BIEOU'})
        """

        self.file_path = file_path
        self.is_test = is_test
        self.low_case = low_case
        self.max_len = dict(max_len)
        self.entity_schema = entity_schema
        self.seq_lens = defaultdict(list)

    def __iter__(self):
        """Generator function
        """

        with open(self.file_path, 'r') as fin:
            for line in fin:
                line = json.loads(line)
                sentence = {}

                state, results = self.get_tokens(line)
                self.seq_lens['tokens'].append(len(results['tokens']))
                if not state or ('tokens' in self.max_len
                                 and len(results['tokens']) > self.max_len['tokens']
                                 and not self.is_test):
                    if not self.is_test:
                        continue
                sentence.update(results)

                # test data only contains sentence text
                # to predict entities and relations
                # if self.is_test:
                #     yield sentence

                state, results = self.get_entity_label(line, len(sentence['tokens']))
                for key, result in results.items():
                    self.seq_lens[key].append(len(result))
                    if key in self.max_len and len(result) > self.max_len[key]:
                        state = False
                if not state:
                    continue
                sentence.update(results)

                state, results = self.get_relation_label(line, sentence['idx2ent'])
                self.seq_lens['span2rel'].append(len(results['span2rel']))
                if not state or ('span2rel' in self.max_len
                                 and len(results['span2rel']) > self.max_len['span2rel']):
                    continue
                sentence.update(results)

                yield sentence

    def get_tokens(self, line):
        """This function splits text into tokens

        Arguments:
            line {dict} -- text

        Returns:
            bool -- execute state
            dict -- results: tokens
        """

        results = {}

        if 'sentText' not in line:
            logger.error("article id: {} sentence id: {} doesn't contain 'sentText'.".format(
                line['articleId'], line['sentId']))
            return False, results

        tokens = line['sentText'].strip().split(' ')
        if self.low_case:
            tokens = list(map(str.lower, tokens))
        results['tokens'] = tokens
        return True, results

    def get_entity_label(self, line, sent_length):
        """This function constructs entity label sequence and entity span sequence
        using entity_schema, in addition to construct mapping relation
        from offset to entity label

        Arguments:
            line {dict} -- text
            sent_length {int} -- sent length

        Returns:
            bool -- execute state
            dict -- results: entity span label sequence,
            entity label sequence, entity id mapping to entity,
            entity span mapping to entity lable,
            tokenized tokens, tokenized index, tokenized label
        """

        results = {}

        if 'entityMentions' not in line:
            logger.error("article id: {} sentence id: {} doesn't contain 'entityMentions'.".format(
                line['articleId'], line['sentId']))
            return False, results

        entity_label = ['O'] * sent_length
        idx2ent = {}
        span2ent = {}

        for entity in line['entityMentions']:
            st, ed = entity['offset']
            idx2ent[entity['emId']] = ((st, ed), entity['text'])
            if st >= ed or st < 0 or st > sent_length or ed < 0 or ed > sent_length:
                logger.error("article id: {} sentence id: {} offset error'.".format(
                    line['articleId'], line['sentId']))
                return False, results

            span2ent[(st, ed)] = entity['label']
            if any(map(lambda x: x != 'O', entity_label[st:ed])):
                logger.error("article id: {} sentence id: {} entity span overlap.".format(
                    line['articleId'], line['sentId']))
                return False, results
            if self.entity_schema == 'BIO':
                entity_label[st] = 'B-' + entity['label']
                for idx in range(st + 1, ed):
                    entity_label[idx] = 'I-' + entity['label']
            elif self.entity_schema == 'BIEOU':
                if ed - st == 1:
                    entity_label[st] = 'U-' + entity['label']
                else:
                    entity_label[st] = 'B-' + entity['label']
                    for idx in range(st + 1, ed - 1):
                        entity_label[idx] = 'I-' + entity['label']
                    entity_label[ed - 1] = 'E-' + entity['label']

        results['entity_labels'] = entity_label
        results['span2ent'] = span2ent
        results['idx2ent'] = idx2ent

        entity_span_label = list(map(lambda x: x[0], entity_label))
        results['entity_span_labels'] = entity_span_label

        return True, results

    def get_relation_label(self, line, idx2ent):
        """This function constructs mapping relation from offset to relation label

        Arguments:
            line {dict} -- text
            idx2ent {dict} -- entity id mapping to entity

        Returns:
            bool -- execute state
            dict -- span2rel: two entity span mapping to relation lable
        """

        results = {}

        if 'relationMentions' not in line:
            logger.error(
                "article id: {} sentence id: {} doesn't contain 'relationMentions'.".format(
                    line['articleId'], line['sentId']))
            return False, results

        span2rel = {}
        for relation in line['relationMentions']:
            entity1_span, entity1_text = idx2ent[relation['em1Id']]
            entity2_span, entity2_text = idx2ent[relation['em2Id']]

            if entity1_text != relation['em1Text'] or entity2_text != relation['em2Text']:
                logger.error(
                    "article id: {} sentence id: {} entity text doesn't match realtiaon text.".
                    format(line['articleId'], line['sentId']))
                return False, None

            direction = '>'
            if entity1_span[0] > entity2_span[0]:
                direction = '<'
                entity1_span, entity2_span = entity2_span, entity1_span

            # two entity overlap
            if entity1_span[1] > entity2_span[0]:
                logger.error(
                    "article id: {} sentence id: {} two entity ({}, {}) are overlap.".format(
                        line['articleId'], line['sentId'], relation['em1Id'], relation['em2Id']))
                continue

            span2rel[(entity1_span, entity2_span)] = relation['label'] + '@' + direction

        results['span2rel'] = span2rel

        return True, results

    def get_seq_lens(self):
        return self.seq_lens
