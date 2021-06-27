#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Function: ner with hmm
# Label: [O ,U-LOC ,B-PER ,L-PER ,U-ORG ,U-MISC ,B-ORG ,L-ORG ,U-PER ,I-ORG ,B-LOC ,L-LOC ,B-MISC ,L-MISC ,I-MISC ,I-PER ,I-LOC]
# A: state transition probability matrix
# B: observation Probability Matrix

import numpy as np
import os
from collections import Counter

import logging
def get_logger(log_path=None, streamhandler=True, filehandler=False):
    if streamhandler is False and filehandler is False:
        return None

    logger = logging.getLogger(__name__)
    fmt_str = ('%(asctime)s.%(msecs)03d %(levelname)7s '
               '[%(thread)d][%(process)d] %(message)s')
    fmt = logging.Formatter(fmt_str, datefmt='%H:%M:%S')

    fh = ch = None
    if filehandler is True:
        fh = logging.FileHandler(log_path, 'w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if streamhandler is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    logger.setLevel(logging.DEBUG)
    return logger


class hmm_tagger:
    ''' ner tagger applied with hmm algorithm
    
    info:
        dataset: CoNLL2003 eng dataset

    train:
        create a hmm tagger and init with conll2003 eng.train dataset

    predict:
        accept only one sentense and hmm tagger will return predicted entities.
    '''
    def __init__(self, conll_path, max_word_num=0, logger=None):
        if logger is None:
            self.logger = get_logger()

        self.logger.info("Init HMM tagger with {}".format(conll_path))

        assert os.path.exists(conll_path), "Dataset doesn't exists!" 

        self.dataset_path = conll_path

        self.logger.info("Loading dataset ...")
        self.data, self.label = self.read_data(conll_path)
        self.logger.info("First sample: {}".format(self.data[0]))
        self.logger.info("First sample label: {}".format(self.label[0]))

        data_set = set([i for x in self.data for i in x])
        if max_word_num is 0:
            self.max_word_num = len(data_set)
        self.max_word_num = max_word_num
        self.logger.info("words number of dataset: {}, max_word_num: {}".format(len(data_set), max_word_num))

        self.logger.info("Encoding dataset ...")
        self.data, self.label, self.word_set, self.word2idx, self.lab2idx = self.encode_dataset(self.data, self.label, max_word_num)
        self.idx2lab = {v:k for k,v in self.lab2idx.items()}

        assert len(self.data)==len(self.label)
        self.logger.info("Sample cnt: {}, vocab size: {}, label cnt: {}".format(len(self.data), len(self.word2idx), len(self.lab2idx)))
        self.logger.info("Label to index map: {}".format(self.lab2idx))

        self.logger.info("Calculating initial hidden matrix ...")
        self.init_mat = self.get_init_matrix(self.label, self.lab2idx)
        self.logger.info("Initial hidden prob: {}".format({self.idx2lab[k]:v for k,v in self.init_mat.items()}))

        self.logger.info("Calculating transition matrix ...")
        self.transition_mat = self.get_transition_matrix(self.label, self.lab2idx)
        self.logger.info("Transition matrix shape: {}".format(self.transition_mat.shape))
        self.logger.info("Transition matrix from hidden 0 to others: {}".format(self.transition_mat[0]))

        self.logger.info("Calculating observation matrix ...")
        self.observation_mat = self.get_observation_matrix(self.data, self.label, self.word2idx, self.lab2idx)
        self.logger.info("Observation matrix shape: {}".format(self.observation_mat.shape))

        self.logger.info("Completed the training of HMM tagger")

    def read_data(self, path):
        '''extract sentense and label from conll2003 dataset

        return:
            unencoded data and label, 2d list
            [['eu', 'rejects', 'german', 'call', 'to', 'boycott', 'british', 'lamb', '.'], ['peter', 'blackburn']]
            [['I-ORG', 'O', 'I-MISC', 'O', 'O', 'O', 'I-MISC', 'O', 'O'], ['I-PER', 'I-PER']]
        '''
        data = []
        label = []

        raw_data = None
        with open(path) as fp:
            raw_data = fp.readlines()
            raw_data = [x.strip().split() for x in raw_data]

        l_data = []
        l_lab = []
        for line in raw_data:
            if len(line)==0:
                if len(l_data)>0:
                    assert len(l_data)==len(l_lab), "Length of data and label must be same"
                    data.append(l_data)
                    label.append(l_lab)
                    l_data = []
                    l_lab = []
                    continue
            else:
                if line[0]=='-DOCSTART-' or len(line)!=4:
                    continue
                l_data.append(line[0].lower())
                l_lab.append(line[-1])
        
        return data, label

    def encode_dataset(self, data, label, max_word_num=15000):
        '''encode the dataset and return index format matrix

        return:
            data: index format dataset
            label: index format label
            word_set: vocab
            word2idx: map of word to index
            lab2idx: map of label to index
        '''
        assert len(data)==len(label), "Length of data and label must be same"

        all_words = [x for l in data for x in l]
        all_lab = [x for l in label for x in l]

        lab2idx = {x:idx for idx, x in enumerate(set(all_lab))}

        # word_freq: [('the', 1143), ('and', 966), ('to', 762), ('of', 669), ('i', 631)]
        word_freq = Counter(all_words).most_common(max_word_num)
        word_set = [x[0] for x in word_freq]

        word2idx = {x:idx+1 for idx, x in enumerate(word_set)}
        word2idx['unk']=0

        for i in range(len(data)):
            for j in range(len(data[i])):
                w = data[i][j]
                lab = label[i][j]

                if w not in word_set:
                    w = 'unk'

                data[i][j]=word2idx[w]
                label[i][j]=lab2idx[lab]

        return data, label, word_set, word2idx, lab2idx

    def get_transition_matrix(self, label_data, lab2idx):
        ''' Get matrix A, where A[i, j] means the probability of state i to j

        The matrix will be normalized in each line.

        para:
            label_data: sentense label index 2d list

        return:
            A: transition matrix, ndarray of shape (lab_len, lab_len)
        '''
        lab_len = len(lab2idx)
        A = np.zeros((lab_len, lab_len))

        for sample in label_data:
            for i in range(len(sample)-1):
                A[sample[i]][sample[i+1]]+=1

        max_a = np.max(A, axis=-1, keepdims=True)
        exp_a = np.exp(A-max_a)

        sum_a = np.sum(exp_a, axis=-1, keepdims=True)
        A = exp_a/sum_a

        return A

    def get_observation_matrix(self, data, label, word2idx, lab2idx):
        ''' Get observation probability matrix B, where B[i, j] means the the probability of hidden state i to observation j

        para:
            data: index of dataset 2d list, notice that it contain `unk`, [[word idx]]
            label: index of label 2d list [[label idx]]
            word2idx: map of word to index
            lab2idx: map of label to index
        
        info:
            default index of unk is 0

        return:
            B: observation matrix, 2d ndarray where B[i, j] means the probability of hidden i is observed as j
        '''
        n = len(data)

        num_of_lab = len(lab2idx)
        num_of_words = len(word2idx)
        B = np.zeros((num_of_lab, num_of_words))

        for i in range(n):
            sentense = data[i]
            hidden = label[i]

            for j in range(len(sentense)):
                B[hidden[j]][sentense[j]] += 1

        max_b = np.max(B, axis=-1, keepdims=True)
        exp_b = np.exp(B-max_b)

        sum_b = np.sum(exp_b, axis=-1, keepdims=True)
        B = exp_b/sum_b
        
        return B

    def get_init_matrix(self, label, lab2idx):
        ''' Get the initation matrix \pi, where \pi[i] means the probability of hidden state init with i

        para:
            label: index format of ner label, 2d ndarray
            lab2idx: label to index dict

        return:
            lab_freq: dict of label to frequency, {lab_idx: freq}
        '''
        lab_freq = {idx:0 for k, idx in lab2idx.items()}
        all_cnt = 0

        for line in label:
            for lab in line:
                lab_freq[lab] = lab_freq.get(lab, 0)+1
                all_cnt += 1

        for k,v in lab_freq.items():
            lab_freq[k] = lab_freq.get(k)/all_cnt

        return lab_freq

    def evaluate(self, val_path, logger=None):
        '''evaluate validation dataset.

        para:
            val_data: raw data path for validation
            word2idx: map of word to index
            lab2idx: map of label to index
            transition_mat: matrix of prob from one state trans to another, 2d list
            observation_mat: matrix of the prob from hidden state to observation state, 2d list
            init_mat: probability of hidden state 
            logger: to log info
        '''
        assert os.path.exists(val_path),"Validation dataset does not exists!" 

        self.val_data, self.val_label = self.read_data(val_path)
        self.val_res = []

        for line in self.val_data:
            predicted_tag = self.predict(line, self.word2idx, self.idx2lab, self.transition_mat, self.observation_mat, self.init_mat)
            self.val_res.append(predicted_tag)

        return self.val_label, self.val_res

    def predict(self, data, word2idx, idx2lab, transition_mat, observation_mat, init_mat, logger=None):
        '''apply viterbi algorithm to predict the ner tag of input data sentense.

        para:
            data: input sentense, raw words, 1d list
            word2idx: map of word to index
            lab2idx: map of label to index
            transition_mat: matrix of prob from one state trans to another, 2d list
            observation_mat: matrix of the prob from hidden state to observation state, 2d list
            init_mat: probability of hidden state 
            logger: to log info
        '''
        assert len(data)>0, "Data to be predicted must be non-empty!" 
        
        num_hidden = len(idx2lab)
        num_word = len(data)

        viterbi_mat = np.zeros((num_hidden, num_word))

        # 0: index of unk
        sentense_idx = [word2idx.get(x, 0) for x in data]

        # init with first word
        for i in range(num_hidden):
            viterbi_mat[i][0] = init_mat[i]*observation_mat[i][sentense_idx[0]]

        # i indicates cur position
        for i in range(1,num_word):
            # j indicates cur possible hidden state loc
            for j in range(num_hidden):
                # x indicates last possible hidden state loc
                trans_observe_prob = [ viterbi_mat[x][i-1]*transition_mat[x][j]*observation_mat[j][sentense_idx[i]] for x in range(num_hidden)]
                viterbi_mat[j][i] = max(trans_observe_prob)

        res_lab = [np.argmax(viterbi_mat[:,i]) for i in range(num_word)]
        res = [idx2lab[x] for x in res_lab]
        return res


if __name__=='__main__':
    train_path = './data/eng.train'
    val_path = './data/eng.testa'

    max_word_num = 21009

    # train phase
    tagger = hmm_tagger(train_path, max_word_num)
    
    # predict phase
    val_lab, val_res = tagger.evaluate(val_path)

    val_base = os.path.dirname(val_path)
    val_out = os.path.join(val_base, 'val_output.txt')

    with open(val_out, 'w') as fout:
        fout.write("label\tpredict\n")
        for i in range(len(val_lab)):
            for j in range(len(val_lab[i])):
                fout.write("{}\t{}\n".format(val_lab[i][j], val_res[i][j]))
            fout.write("\n")

