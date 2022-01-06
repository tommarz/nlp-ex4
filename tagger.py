"""
nlp, assignment 4, 2021

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""
from typing import List, Any

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import torch.nn as nn
from torchtext import data
import torch.optim as optim
from math import log, isfinite, inf
from collections import Counter, defaultdict
import numpy as np

import sys, os, time, platform, nltk, random

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.
def use_seed(seed=2512021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)
    # torch.backends.cudnn.deterministic = True


# utility functions to read the corpus
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    # TODO edit the dictionary to have your own details
    return {'name': 'Tom Marzea', 'id': '318443595', 'email': 'tommarz@post.bgu.ac.il'}


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append((word, tag))
        line = f.readline()
    return sentence


def load_annotated_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

EOS = "<EOS>"

allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = defaultdict(Counter)
transitionCounts = defaultdict(Counter)
emissionCounts = defaultdict(Counter)
# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {}  # transisions probabilities
B = {}  # emmissions probabilities


def insert_to_counters_dict(counter_dict, dict_key, counter_key, value=1):
    if dict_key not in counter_dict:
        counter_dict[dict_key] = Counter()
    counter_dict[dict_key][counter_key] += value


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
    and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transitionCounts and emissionCounts
    should be computed with pseudo tags and shoud be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transitionCounts and  emissionCounts

    Args:
    tagged_sentences: a list of tagged sentences, each tagged sentence is a
     list of pairs (w,t), as returned by load_annotated_corpus().

    Return:
    [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
    """
    for tagged_sentence in tagged_sentences:
        prev_t = START
        for word, t in tagged_sentence:
            w = word.lower()
            allTagCounts[t] += 1
            perWordTagCounts[w][t] += 1
            emissionCounts[t][w] += 1
            transitionCounts[prev_t][t] += 1
            prev_t = t
        transitionCounts[prev_t][END] += 1

    for t in list(emissionCounts):
        tags = emissionCounts[t]
        for w in list(perWordTagCounts) + [UNK]:
            tags[w] += 1

    for prev_t in list(transitionCounts):
        tags = transitionCounts[prev_t]
        all_tags = list(allTagCounts) + [END]
        if prev_t == START:
            all_tags.remove(END)
        for t in all_tags:
            tags[t] += 1

    for prev_t in list(transitionCounts):
        prev_t_count = sum(transitionCounts[prev_t].values())
        A[prev_t] = {t: log(transition_count / prev_t_count) for t, transition_count in
                     transitionCounts[prev_t].items()}

    for t in list(emissionCounts):
        t_count = sum(emissionCounts[t].values())
        B[t] = {w: log(emission_count / t_count) for w, emission_count in emissionCounts[t].items()}

    return [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        list: list of pairs
    """

    tagged_sentence = []
    for w in sentence:
        word = w.lower()
        tag = perWordTagCounts[word].most_common(1)[0][0] if word in perWordTagCounts else \
            random.choices(list(allTagCounts.keys()), k=1, weights=list(allTagCounts.values()))[0]
        tagged_sentence.append((w, tag))
    return tagged_sentence


# ===========================================
#       POS tagging with HMM
# ===========================================


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        list: list of pairs
    """

    end_item = viterbi(sentence, A, B)
    tags = retrace(end_item)
    tagged_sentence = list(zip(sentence, tags))

    return tagged_sentence


def viterbi(sentence, A, B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tupple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probability of the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtraking.

        """
    # Hint 1: For efficiency reasons - for words seen in training there is no
    #      need to consider all tags in the tagset, but only tags seen with that
    #      word. For OOV you have to consider all tags.
    # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
    #         current list = [ the dummy item ]
    # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END

    dummy_item = (START, None, 0)
    prev_col = [dummy_item]

    for w in sentence:
        col = []
        word = w.lower()
        for tag in list(A):
            if tag in [START, END]:
                continue
            if word not in B[tag].keys():
                word = UNK
            elif tag not in perWordTagCounts[word]:
                continue
            emission_prob = B[tag][word]
            best_v = predict_next_best(tag, emission_prob, prev_col, A)
            col.append(best_v)
        prev_col = col
    v_last = predict_next_best(tag=END, emission_prob=0, predecessor_list=prev_col, A=A)
    return v_last


# a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """
    :param end_item: tuple of (t, r, p) as defined above
    :return: Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    lst = []
    curr_item = end_item[1]
    while curr_item[1] is not None:
        lst.append(curr_item[0])
        curr_item = curr_item[1]
    return reversed(lst)


# a suggestion for a helper function. Not an API requirement
def predict_next_best(tag, emission_prob, predecessor_list, A):
    """
    :param tag: The tag of the current state in the current column
    :param emission_prob: The emission probability of the current tag and word
    :param predecessor_list: The list of states in the previous column of the Viterbi matrix
    :param A: The transition matrix as defined above
    :return: Returns a new item (tupple) of (t,r,p) as defined above
    """
    max_prob = float(-inf)
    best_cell = None
    for t, r, p in predecessor_list:
        cell_prob = p + A[t][tag] + emission_prob
        if cell_prob > max_prob:
            max_prob = cell_prob
            best_cell = (tag, (t, r, p), cell_prob)
    return best_cell


# # a suggestion for a helper function. Not an API requirement
# def predict_next_best(word, tag, predecessor_list):
#     """
#     :param word: The word of the current state
#     :param tag: The tag of the current state
#     :param predecessor_list: List of (t, p) where t is the previous tag and p is it's probability
#     :return: Returns a new item (tupple)
#     """
#     for t,r,p in predecessor_list:


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): the HMM emmission probabilities.
     """
    p = 0  # joint log prob. of words and tags

    prev_t = START
    for w, t in sentence:
        p += (B[t][w.lower()] + A[prev_t][t])
        prev_t = t
    p += A[prev_t][END]

    assert isfinite(p) and p < 0  # Should be negative. Think why!
    return p


# ===========================================
#       POS tagging with BiLSTM
# ===========================================

""" You are required to support two types of bi-LSTM:
    1. a vanilla biLSTM in which the input layer is based on simple word embeddings
    2. a case-based BiLSTM in which input vectors combine a 3-dim binary vector
        encoding case information, see
        https://arxiv.org/pdf/1510.06168.pdf
"""


# Suggestions and tips, not part of the required API
#
#  1. You can use PyTorch torch.nn module to define your LSTM, see:
#     https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
#  2. You can have the BLSTM tagger model(s) implemented in a dedicated class
#     (this could be a subclass of torch.nn.Module)
#  3. Think about padding.
#  4. Consider using dropout layers
#  5. Think about the way you implement the input representation
#  6. Consider using different unit types (LSTM, GRU,LeRU)


def get_sentence_indices(seq, to_ix):
    # idxs = [to_ix.get(w.lower(), 0) for w in seq]
    # return torch.tensor(idxs, dtype=torch.long)
    return [to_ix.get(w, 0) for w in seq]


def get_case_features(seq):
    # return torch.tensor([[word.islower(), word.isupper(), word[0].isupper()] for word in seq], dtype=torch.bool)
    return [[word.islower(), word.isupper(), word[0].isupper()] for word in seq]


def prepare_data(data, word_to_idx, tag_to_idx, max_seq_len, input_rep=0, train=True):
    sentences_word_features = []
    sentences_tag_indices = []
    for sentence in data:
        sentence += ([(EOS, END)] if train else [EOS]) * max(0, max_seq_len - len(sentence))
        sentence = sentence[:max_seq_len]
        sentence_features = []
        sentence_tags = []
        for token in sentence:
            word = token[0] if train else token
            sentence_features.append([word_to_idx.get(word, 0)] + (
                [word.islower(), word.isupper(), word[0].isupper()] if input_rep else []))
            if train:
                tag = token[1]
                sentence_tags.append(tag_to_idx.get(tag, 0))
        sentences_word_features.append(sentence_features)
        if train:
            sentences_tag_indices.append(sentence_tags)

    return sentences_word_features, sentences_tag_indices


def loss_batch(model, loss_func, xb, yb, opt=None):
    """
    Calculates the loss of the model for a given batch (xb, yb) and loss function
    :param model: The model to calculate the loss with
    :param loss_func: The criterion for calculating the loss
    :param xb: The x of the batch
    :param yb: The y of the batch
    :param opt: The optimizer to perform the backprop with, defaults to None which means no backprop will be performed
    :return: The loss for the given batch
    """
    y_pred = model(xb)
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    yb = yb.reshape(-1)
    loss = loss_func(y_pred, yb.long())

    # If not none, update the weights using the back-propagation algorithm
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


class BiLSTMTagger(nn.Module):
    def __init__(self, max_vocab_size, embedding_dimension, num_of_layers, output_dimension, min_frequency, input_rep,
                 pretrained_embeddings_fn, data_fn, max_seq_len=None, hidden_dim=64, proj_size=None, dropout=0.1):
        super().__init__()
        # self.max_vocab_size = max_vocab_size
        # self.min_frequency = min_frequency
        self.input_rep = input_rep
        # proj_size = hidden_dim if proj_size is None else proj_size
        self.lstm = nn.LSTM(input_size=embedding_dimension + 3 * input_rep, hidden_size=hidden_dim, num_layers=num_of_layers,
                            # proj_size=proj_size,
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.softmax = nn.Softmax(2)

        tagged_sentences = load_annotated_corpus(data_fn)

        self.tag_to_idx = {}

        words_counter = Counter()

        for sentence in tagged_sentences:
            for word, tag in sentence:
                words_counter[word.lower()] += 1
                if tag not in self.tag_to_idx:
                    self.tag_to_idx[tag] = len(self.tag_to_idx)

        self.max_seq_len = max_seq_len if max_seq_len else max(
            [len(sentence) for sentence in tagged_sentences]) + 1  # +1 for the EOS token

        vocab = [word for word, count in words_counter.items() if
                 count >= min_frequency] if max_vocab_size == -1 else [w[0] for w in
                                                                       words_counter.most_common(max_vocab_size)]

        self.idx_to_tag = {v: k for k, v in self.tag_to_idx.items()}

        vectors = load_pretrained_embeddings(pretrained_embeddings_fn, vocab)

        pretrained_embeddings = vectors['weights']
        self.word_to_idx = vectors['word_to_idx']

        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

        if pretrained_embeddings_fn is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        else:
            self.embedding = nn.Embedding(max_vocab_size, embedding_dim)

        self.hidden2tag = nn.Linear(2 * hidden_dim, output_dimension + 1)

    def forward(self, inp):
        indices = inp[:, :, 0]
        embeds = self.embedding(indices)
        x = torch.cat((embeds, inp[:, :, 1:]), dim=2) if self.input_rep else embeds
        lstm_out, _ = self.lstm(x)
        tag_space = self.hidden2tag(lstm_out)
        return self.softmax(tag_space)


def initialize_rnn_model(params_d):
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
       the lstm model. The LSTM is initialized based on the specified parameters.
       thr returned dict is may have other or additional fields.

    Args:
        params_d (dict): a dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'max_vocab_size': max vocabulary size (int),
                        'min_frequency': the occurence threshold to consider (int),
                        'input_rep': 0 for the vanilla and 1 for the case-base (int),
                        'embedding_dimension': embedding vectors size (int),
                        'num_of_layers': number of layers (int),
                        'output_dimension': number of tags in tagset (int),
                        'pretrained_embeddings_fn': str,
                        'data_fn': str
                        }
                        max_vocab_size sets a constraints on the vocab dimention.
                            If the its value is smaller than the number of unique
                            tokens in data_fn, the words to consider are the most
                            frequent words. If max_vocab_size = -1, all words
                            occuring more that min_frequency are considered.
                        min_frequency privides a threshold under which words are
                            not considered at all. (If min_frequency=1 all words
                            up to max_vocab_size are considered;
                            If min_frequency=3, we only consider words that appear
                            at least three times.)
                        input_rep (int): sets the input representation. Values:
                            0 (vanilla), 1 (case-base);
                            <other int>: other models, if you are playful
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        a dictionary with the at least the following key-value pairs:
                                       {'lstm': torch.nn.Module object,
                                       input_rep: [0|1]}
        #Hint: you may consider adding the embeddings and the vocabulary
        #to the returned dict
    """

    # TODO complete the code

    model = {'lstm': BiLSTMTagger(**params_d), 'input_rep': params_d['input_rep']}

    return model


def load_pretrained_embeddings(path, vocab=None):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        You can also access the vectors at:
         https://www.dropbox.com/s/qxak38ybjom696y/glove.6B.100d.txt?dl=0
         (for efficiency (time and memory) - load only the vectors you need)
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.

    Args:
        path (str): full path to the embeddings file
        vocab (list): a list of words to have embeddings for. Defaults to None.

    """
    word_to_idx = {UNK: 0}
    weights = defaultdict(lambda: [0.0] * 100)
    weights[0] = [0.0] * 100

    with open(path, 'rb') as f:
        idx = 1  # starting index
        for l in f.readlines():
            line = l.decode().strip().split()
            word = line[0]
            if vocab is None or word in vocab:
                weights[idx] = list(map(float, line[1:]))
                word_to_idx[word] = idx
                idx += 1
    vectors = {'word_to_idx': word_to_idx, 'weights': torch.Tensor(list(weights.values()))}
    return vectors


def train_rnn(model, train_data, val_data=None):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (dict): the model dict as returned by initialize_rnn_model()
        train_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus()
        val_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus() to be used for validation.
                            Defaults to None
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful
    """
    # Tips:
    # 1. you have to specify an optimizer
    # 2. you have to specify the loss function and the stopping criteria
    # 3. consider using batching
    # 4. some of the above could be implemented in helper functions (not part of
    #    the required API)

    # TODO complete the code

    bilstm_model = model['lstm']
    input_rep = model['input_rep']

    criterion = nn.CrossEntropyLoss()  # you can set the parameters as you like
    optimizer = optim.Adam(bilstm_model.parameters())

    bilstm_model: BiLSTMTagger = bilstm_model.to(device)
    criterion = criterion.to(device)

    epochs = 5
    bs = 32

    if not val_data:
        # np.random.shuffle(train_data)
        words_features, tags = prepare_data(train_data, bilstm_model.word_to_idx, bilstm_model.tag_to_idx,
                                            max_seq_len=bilstm_model.max_seq_len,
                                            input_rep=input_rep)
        train_bound = round(len(train_data) * 0.8)
        # torch.utils.data.random_split()
        # X_train, y_train = torch.Tensor(words[:train_bound]), torch.Tensor(tags[:train_bound])
        # X_val, y_val = torch.Tensor(words[train_bound:]), torch.Tensor(tags[train_bound:])
        # X, y = torch.LongTensor(words_features), torch.LongTensor(tags)
        dataset = TensorDataset(torch.LongTensor(words_features), torch.LongTensor(tags))
        train_ds, val_ds = torch.utils.data.random_split(dataset,
                                                         [train_bound, len(train_data) - train_bound])
    else:
        train_features, train_tags = prepare_data(train_data, bilstm_model.word_to_idx, bilstm_model.tag_to_idx,
                                                  max_seq_len=bilstm_model.max_seq_len,
                                                  input_rep=input_rep)
        val_features, val_tags = prepare_data(val_data, bilstm_model.word_to_idx, bilstm_model.tag_to_idx,
                                              max_seq_len=bilstm_model.max_seq_len,
                                              input_rep=input_rep)

        train_ds = TensorDataset(torch.LongTensor(train_features), torch.LongTensor(train_tags))
        val_ds = TensorDataset(torch.LongTensor(val_features), torch.LongTensor(val_tags))

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=True)

    verbose = 1

    best_loss = float('inf')

    PATH = "model2.pt"

    for epoch in range(epochs):
        bilstm_model.train()
        for xb, yb in train_dl:
            loss_batch(bilstm_model, criterion, xb, yb, optimizer)

        bilstm_model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(bilstm_model, criterion, xb, yb) for xb, yb in val_dl]
            )
            if verbose:
                val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
                if val_loss < best_loss:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': bilstm_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                    }, PATH)
                    best_loss = val_loss
                print(epoch, val_loss)


def rnn_tag_sentence(sentence, model):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence and t is the predicted tag.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict):  a dictionary with the trained BiLSTM model and all that is needed
                        to tag a sentence.

    Return:
        list: list of pairs
    """

    # TODO complete the code

    bilstm_model = model['lstm']
    input_rep = model['input_rep']

    # pad the sentence:
    X = get_sentence_indices(sentence + [EOS], to_ix=bilstm_model.word_to_idx)

    predictions = bilstm_model(X.reshape(1, -1).to(device))
    tags_indices = torch.argmax(predictions, dim=-1).reshape(-1)
    tags = [bilstm_model.idx_to_tag.get(int(idx), UNK) for idx in tags_indices]
    tags = tags[:len(sentence)]

    tagged_sentence = list(zip(sentence, tags))

    return tagged_sentence


def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """

    model_params = {'max_vocab_size': -1,
                    'min_frequency': 2,
                    'input_rep': 1,
                    'embedding_dimension': 100,
                    'num_of_layers': 3,
                    'output_dimension': 17,
                    'pretrained_embeddings_fn': 'glove.6B.100d.txt',
                    'data_fn': 'en-ud-train.upos.tsv'
                    }

    return model_params


# ===========================================================
#       Wrapper function (tagging with a specified model)
# ===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM)
           or the model isteld and the input_rep flag (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[model_dict]}
        4. BiLSTM+case: {'cblstm': [model_dict]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

        Parameters for an LSTM: the model dictionary (allows tagging the given sentence)


    Return:
        list: list of pairs
    """
    if list(model.keys())[0] == 'baseline':
        return baseline_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])
    if list(model.keys())[0] == 'hmm':
        return hmm_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])
    if list(model.keys())[0] == 'blstm':
        return rnn_tag_sentence(sentence, list(model.values())[0][0])
    if list(model.keys())[0] == 'cblstm':
        return rnn_tag_sentence(sentence, list(model.values())[0][0])


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correctly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence) == len(pred_sentence)

    correct = 0
    correctOOV = 0
    OOV = 0

    for gold, pred in zip(gold_sentence, pred_sentence):
        w_gold, t_gold = gold
        w_pred, t_pred = pred
        if w_pred == UNK:
            OOV += 1
            correctOOV += t_gold == t_pred
        correct += t_gold == t_pred

    return correct, correctOOV, OOV
