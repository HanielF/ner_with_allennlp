# -*- coding: utf-8 -*-
import numpy as np
import math

from collections import Counter
from collections import defaultdict
#  from allennlp.data.dataset_readers.dataset_utils.span_utils import bioul_tags_to_spans

class HMM:
    def __init__(self, sentences, entities, poses=None, chunks=None):
        """
        Inıtializer of the HMM model. Takes 4 parameters, two of them is necessary which are sentences and entities.
        :param sentences: List of sentences which is list of words.
        :param entities: List of entity tags of each sentence.
        :param poses: List of pos tags of each sentence.
        :param chunks: List of chunk tags of each sentence.
        """
        super().__init__()
        self.sentences, self.poses, self.chunks, self.entities = sentences, poses, chunks, entities
        self.all_words, self.all_poses, self.all_chunks, self.all_entities = self.merge_lists()
        self.unique_entity_count = len(set(self.all_entities))
        self.unique_word_count = len(set(self.all_words))

        self.n_init_entity = self.init_entity_count()
        self.init_prob, self.s_init_prob = self.cal_init_prob()

        self.n_entity_word = self.entity_word_count()
        self.emis_prob, self.s_emis_prob = self.cal_emission_prob()

        self.n_bi_en = self.bi_en_count()
        self.trans_prob, self.s_trans_prob = self.cal_trans_prob()
        print("self.n_bi_en")
        print(self.n_bi_en)
        print("self.trans_prob, self.s_trans_prob ")
        print(self.trans_prob, self.s_trans_prob )

        self._true_positives = dict()  #: Dict[str, int]
        self._false_positives = dict()  #: Dict[str, int]
        self._false_negatives = dict()  #: Dict[str, int]

    def entity_word_count(self):
        """
        This function traveses list of word-entity pairs and counts them.
        :return: Word counts for each entity tag as dict, for example: {"O":{"car":2,"fire":3,...},...}
        """
        n_entity_word_dict = dict()

        for e, w in zip(self.all_entities, self.all_words):
            if e not in n_entity_word_dict.keys():
                n_entity_word_dict[e] = dict()
            n_entity_word_dict[e][w] = 1 if w not in n_entity_word_dict[e].keys() else n_entity_word_dict[e][w] + 1

        return n_entity_word_dict

    def cal_emission_prob(self):
        """
        This function calculates the normal and smoothed probabilities of the selection of the word in the word set of each entity tag.
        :return: Normal and Smoothed Probabilities dicts as tuple in order.

        Example return values:
        Normal: {"O":{"car":0.2,"fire":0.03,...},...}
        Smoothed: {"O":{"car":0.18,"fire":0.025,...},...}
        """
        res = dict()
        s_res = dict()

        for e in self.n_entity_word.keys():
            # Initialize the keys and dicts
            if e not in res.keys():
                res[e] = dict()
                s_res[e] = dict()

            total = sum(self.n_entity_word[e].values())
            for w in self.n_entity_word[e].keys():
                res[e][w] = self.n_entity_word[e][w] / total
                s_res[e][w] = (self.n_entity_word[e][w] + 1) / (total + self.unique_word_count)
            s_res[e]['NaN'] = 1 / (total + self.unique_word_count)
        return res, s_res

    def cal_trans_prob(self):
        """
        This function calculates the normal and smoothed probabilities of which entity tag comes after another entity tag.
        :return: Normal and Smoothed Probabilities dicts as tuple in order.

        Example return values:
        Normal: {"O":{"B-PER":0.2,"O":0.03,...},...}
        Smoothed: {"O":{"B-PER":0.18,"O":0.025,...},...}
        """
        res = dict()
        s_res = dict()

        for e1 in self.n_bi_en.keys():
            # Initialize the keys and dicts
            if e1 not in res.keys():
                res[e1] = dict()
                s_res[e1] = dict()
            total = sum(self.n_bi_en[e1].values())
            for e2 in self.n_bi_en[e1].keys():
                res[e1][e2] = self.n_bi_en[e1][e2] / total
                s_res[e1][e2] = (self.n_bi_en[e1][e2] + 1) / (total + self.unique_entity_count)

        return res, s_res

    def bi_en_count(self):
        """
        This function traverses entities list and creates biagrams of entities for each sentence, then counts biagrams.
        :return: Biagram counts as dict, for example: {"O":{"O":10,...},"B-PER":{"O":8,...}}
        """
        bi_entity = dict()

        unique_ent = list(set(self.all_entities))
        # Initialize the keys and dicts and initial values of leaves
        for ent in unique_ent:
            bi_entity[ent] = dict()
            for ent2 in unique_ent:
                bi_entity[ent][ent2] = 0

        for entity_list in self.entities:
            bi_list = zip(*[entity_list[i:] for i in range(2)])     # Creates list of biagrams
            for bi in bi_list:
                bi_entity[bi[0]][bi[1]] += 1

        return bi_entity

    def cal_init_prob(self):
        """
        Thic function calculates the normal and smoothed probabilities of which tag the sentences have in the first index.
        :return: Normal and Smoothed Probabilities dicts as tuple in order.

        Example return values:
        Normal: {"B-PER":0.2,"O":0.03,...}
        Smoothed: {"B-PER":0.18,"O":0.025,...}
        """
        total = sum(self.n_init_entity.values())
        res = dict()
        s_res = dict()
        for key in self.n_init_entity.keys():
            res[key] = self.n_init_entity[key] / total
            s_res[key] = (self.n_init_entity[key] + 1) / (total + self.unique_entity_count)
        return res, s_res

    def init_entity_count(self):
        """
        This function traverses entities list which contains lists of entities of the training sentences.
        Counts the first entities of each sentence.
        :return: Counts of entities of the first words of sentences as dict -> for example: {"B-PER":3,"O":8,...}
        """
        res = dict()
        unique_ent = list(set(self.all_entities))
        # Set initial values
        for ent in unique_ent:
            res[ent] = 0

        for entity_list in self.entities:
            res[entity_list[0]] += 1

        return res

    def merge_lists(self):
        """
        This function flattens nested lists for use when necessary.
        Model does not need POS tags and Chunk tags, so this function checks are there is initialized or not.
        :return: Flattened sentences, POS tags, Chunk tags and entity tags lists as tuple in order.

        Example return values
        Parsed Sentences : ["a","b","c","x",...]
        POS Values: ["NNP","VP","NNP","NNP",...]
        Chunks: ["B-NP","B-PP","I-NP","B-NP",...]
        Entities: ["O","B-PER","O","O",...]
        """
        all_words = []
        all_pos = []
        all_chunk = []
        all_entities = []

        for se in self.sentences:
            for w in se:
                all_words.append(w)
        if self.poses is not None:
            for po in self.poses:
                for p in po:
                    all_pos.append(p)
        if self.chunks is not None:
            for ch in self.chunks:
                for c in ch:
                    all_chunk.append(c)
        for en in self.entities:
            for e in en:
                all_entities.append(e)

        return all_words, all_pos, all_chunk, all_entities


    def get_metric(self, label, predict, reset: bool = False):
        """

        paras:
            label: 2dList(str) e.g. [["B-PER", "L-PER", "O"]]
            predict: 2dList(str) e.g. [["B-PER", "L-PER", "O"]]
            shape as (samples, length)

        returns

        `Dict[str, float]`
            A Dict per label containing following the span based metrics:
            - precision : `float`
            - recall : `float`
            - f1-measure : `float`

            Additionally, an `overall` key is included, which provides the precision,
            recall and f1-measure for all spans.
        """

        for i in range(len(label)):
            i_lab = label[i]
            i_pre = predict[i]

            #  predicted_spans = self.tags_to_spans_function(i_pre)
            #  gold_spans = self.tags_to_spans_function(i_lab)
            #  print("label sequence:{}".format(i_lab))
            #  print("predict sequence:{}".format(i_pre))
            predicted_spans = self.iob1_tags_to_spans(i_pre)
            #  print("predicted_span:{}".format(predicted_spans))
            gold_spans = self.iob1_tags_to_spans(i_lab)
            #  print("gold_spans:{}".format(gold_spans))

            for span in predicted_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] = self._true_positives.get(span[0], 0) + 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] = self._false_positives.get(span[0], 0) + 1
            # These spans weren't predicted.
            for span in gold_spans:
                self._false_negatives[span[0]] = self._false_negatives.get(span[0], 0) + 1

        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        print(all_tags)
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag], self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()),
        )
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset_metrics()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = true_positives / (true_positives + false_positives + 1e-13)
        recall = true_positives / (true_positives + false_negatives + 1e-13)
        f1_measure = 2.0 * (precision * recall) / (precision + recall + 1e-13)
        return precision, recall, f1_measure

    def reset_metrics(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

    def _iob1_start_of_chunk(
        self, 
        prev_bio_tag,
        prev_conll_tag,
        curr_bio_tag,
        curr_conll_tag,
    ):
        if curr_bio_tag == "B":
            return True
        if curr_bio_tag == "I" and prev_bio_tag == "O":
            return True
        if curr_bio_tag != "O" and prev_conll_tag != curr_conll_tag:
            return True
        return False

    def iob1_tags_to_spans(self, tag_sequence, classes_to_ignore = None):
        """
        Given a sequence corresponding to IOB1 tags, extracts spans.
        Spans are inclusive and can be of zero length, representing a single word span.
        Ill-formed spans are also included (i.e., those where "B-LABEL" is not preceded
        by "I-LABEL" or "B-LABEL").

        # Parameters

        tag_sequence : `List[str]`, required.
            The integer class labels for a sequence.
        classes_to_ignore : `List[str]`, optional (default = `None`).
            A list of string class labels `excluding` the bio tag
            which should be ignored when extracting spans.

        # Returns

        spans : `List[TypedStringSpan]`
            The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
            Note that the label `does not` contain any BIO tag prefixes.
        """
        classes_to_ignore = classes_to_ignore or []
        spans: Set[Tuple[str, Tuple[int, int]]] = set()
        span_start = 0
        span_end = 0
        active_conll_tag = None
        prev_bio_tag = None
        prev_conll_tag = None
        #  print(tag_sequence)
        for index, string_tag in enumerate(tag_sequence):
            curr_bio_tag = string_tag[0]
            curr_conll_tag = string_tag[2:]

            if curr_bio_tag not in ["B", "I", "O"]:
                raise InvalidTagSequence(tag_sequence)
            if curr_bio_tag == "O" or curr_conll_tag in classes_to_ignore:
                # The span has ended.
                if active_conll_tag is not None:
                    spans.add((active_conll_tag, (span_start, span_end)))
                active_conll_tag = None
            elif self._iob1_start_of_chunk(prev_bio_tag, prev_conll_tag, curr_bio_tag, curr_conll_tag):
                # We are entering a new span; reset indices
                # and active tag to new span.
                if active_conll_tag is not None:
                    spans.add((active_conll_tag, (span_start, span_end)))
                active_conll_tag = curr_conll_tag
                span_start = index
                span_end = index
            else:
                # bio_tag == "I" and curr_conll_tag == active_conll_tag
                # We're continuing a span.
                span_end += 1

            prev_bio_tag = string_tag[0]
            prev_conll_tag = string_tag[2:]
        # Last token might have been a part of a valid span.
        if active_conll_tag is not None:
            spans.add((active_conll_tag, (span_start, span_end)))
        return list(spans)



    def bioul_tags_to_spans(self, tag_sequence, classes_to_ignore=None):
        """
        Given a sequence corresponding to BIOUL tags, extracts spans.
        Spans are inclusive and can be of zero length, representing a single word span.
        Ill-formed spans are not allowed and will raise `InvalidTagSequence`.
        This function works properly when the spans are unlabeled (i.e., your labels are
        simply "B", "I", "O", "U", and "L").

        # Parameters

        tag_sequence : `List[str]`, required.
            The tag sequence encoded in BIOUL, e.g. ["B-PER", "L-PER", "O"].
        classes_to_ignore : `List[str]`, optional (default = `None`).
            A list of string class labels `excluding` the bio tag
            which should be ignored when extracting spans.

        # Returns

        spans : `List[TypedStringSpan]`
            The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        """
        spans = []
        classes_to_ignore = classes_to_ignore or []
        index = 0
        while index < len(tag_sequence):
            label = tag_sequence[index]
            if label[0] == "U":
                spans.append((label.partition("-")[2], (index, index)))
            elif label[0] == "B":
                start = index
                span_flag = True
                while label[0] != "L":
                    index += 1
                    if index >= len(tag_sequence):
                        #  raise InvalidTagSequence(tag_sequence)
                        span_flag = False
                        break
                    label = tag_sequence[index]
                    if not (label[0] == "I" or label[0] == "L"):
                        #  raise InvalidTagSequence(tag_sequence)
                        span_flag = False
                        break
                if span_flag:
                    spans.append((label.partition("-")[2], (start, index)))
            #  else:
            #  if label != "O":
            #  print(label)
            #  raise InvalidTagSequence(tag_sequence)
            #  continue
            index += 1
        return [span for span in spans if span[0] not in classes_to_ignore]




def viterbi(hmm, t_sentences):
    """
    This function dynamically applies the viterbi algorithm.
    :param hmm: Trained HMM model
    :param t_sentences: Test sentences
    :return: Predicted entity tags for each sentence in test sentences
    """
    res = list()
    states = list(hmm.n_init_entity.keys())

    for sent in t_sentences:
        D = np.zeros([len(states), len(sent)])      # Viterbi matrix
        E = np.zeros([len(states), len(sent) - 1])      # Matrix to store where highest calculation come to that cell.

        # Fill the first column of the Viterbi matrix
        for s in range(len(states)):
            init = hmm.init_prob[states[s]] if hmm.init_prob[states[s]] != 0 else hmm.s_init_prob[states[s]]
            emis = hmm.s_emis_prob[states[s]][sent[0]] if sent[0] in hmm.emis_prob[states[s]].keys() else \
                hmm.s_emis_prob[states[s]]["NaN"]
            D[s, 0] = math.log2(init) + math.log2(emis)

        # For each column (except first)
        for w in range(1, len(sent)):
            for i in range(len(states)):
                temp = np.zeros(len(states))
                past = D[:, w - 1]  # Retrieves the column with calculated values ​​for the previous word.

                # It calculates the probability of coming to the cell from the previous column and adds it to the list.
                for j in range(len(states)):
                    trans = hmm.trans_prob[states[j]][states[i]] if hmm.trans_prob[states[j]][states[i]] != 0 else \
                        hmm.s_trans_prob[states[j]][states[i]]
                    temp[j] = math.log2(trans) + past[j]

                emis = hmm.s_emis_prob[states[i]][sent[w]] if sent[w] in hmm.emis_prob[states[i]].keys() else \
                    hmm.s_emis_prob[states[i]]["NaN"]

                # Calculates the value of the current cell using the value of the cell that is most likely to arrive.
                D[i, w] = np.amax(temp) + math.log2(emis)

                # Set the cell to index of the cell that is most likely to arrive.
                E[i, w - 1] = np.argmax(temp)

        max_ind = np.zeros(len(sent))
        max_ind[-1] = np.argmax(D[:, -1])   # Get the index of maximum value at the last column

        # By using backtracking, the indices of the most suitable tags are recorded.
        for n in range(len(sent) - 2, -1, -1):
            max_ind[n] = E[int(max_ind[n + 1]), n]

        # Finds the names of the tags from the indices, creates a list and adds them to the answer list.
        res.append([states[n] for n in max_ind.astype(int)])
    return res


def dataset(conll_file):
    """
    Function to read CoNNL file
    :param conll_file: Path of the CoNNL file
    :return: Parsed sentences, POS tags, Chunk tags and Entity tags in given order as tuple

    Example return values
    Parsed Sentences : [["a","b",...],["c","x"],...]
    POS Values: [["NNP","VP",...],["NNP","NNP"],...]
    Chunks: [["B-NP","B-PP",...],["I-NP","B-NP"],...]
    Entities: [["O","B-PER",...],["O","O"],...]
    """
    sent = []
    pos = []
    chunk = []
    entity = []
    temp_sent = []
    temp_pos = []
    temp_chunk = []
    temp_entity = []

    with open(conll_file) as f:
        conll_raw_data = f.readlines()
    conll_raw_data = [x.strip() for x in conll_raw_data]  # Clean the left/right spaces if there are

    for line in conll_raw_data:
        if line != '':
            split_line = line.split()
            if len(split_line) == 4:
                if split_line[0] != '-DOCSTART-':  # Do not get the lines which start with -DOCSTART-
                    temp_sent.append(split_line[0].lower())  # Get words in lowercase
                    temp_pos.append(split_line[1])
                    temp_chunk.append(split_line[2])
                    temp_entity.append(split_line[3])
            else:
                raise IndexError(
                    'Line split length does not equal 4.')  # A line must contain a word and 3 tags (POS, Chunk, Entity)
        else:
            if len(temp_sent) > 0:
                assert (len(sent) == len(pos))
                assert (len(sent) == len(chunk))
                assert (len(sent) == len(entity))
                sent.append(temp_sent)
                pos.append(temp_pos)
                chunk.append(temp_chunk)
                entity.append(temp_entity)
                temp_sent = []
                temp_pos = []
                temp_chunk = []
                temp_entity = []

    return sent, pos, chunk, entity


def accuracy(org_ent, pred_ent):
    """
    Counts the true predicted entities and returns the calculated accuracy.
    :param org_ent: Original entities of the test sentences as [["O","B-PER",...],["O","O"],...]
    :param pred_ent: Predicted entities of the test sentences as [["O","B-PER",...],["O","O"],...]
    :return: accuracy as float
    """
    true = 0

    org = [item for sublist in org_ent for item in sublist]
    pred = [item for sublist in pred_ent for item in sublist]

    for t, r in zip(org, pred):
        if t == r:
            true += 1
    print(true)
    print(len(org))
    return true / len(org)




def main():
    """
    Main function of the program
    """
    sentences, _, _, entities = dataset('./data/eng.train')
    #  dev_sentences, _, _, dev_entities = dataset('./data/eng.testa')
    test_sentences, _, _, test_entities = dataset('./data/eng.testb')

    hmm = HMM(sentences=sentences, entities=entities)

    res = viterbi(hmm=hmm, t_sentences=test_sentences)

    #  print(res)
    print("test_entities  and result shape: {} {}".format(len(test_entities), len(res)))
    metrics = hmm.get_metric(test_entities, res)
    print(metrics)
    print("Acc: {}".format(accuracy(org_ent=test_entities, pred_ent=res)))


main()
