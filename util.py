# -*- coding: utf-8 -*-

import sys
import os
import re
import random
from codecs import open as codecs_open
from collections import Counter, defaultdict
import cPickle as pickle
import numpy as np


UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
RANDOM_SEED = 1234

UNLABELED_SUFFIX = 'unl'


class TextReader(object):

    def __init__(self, data_dir, class_names):
        self.data_dir = data_dir
        self.class_names = list(set(class_names) - set([UNLABELED_SUFFIX]))
        self.num_classes = len(set(class_names) - set([UNLABELED_SUFFIX]))
        self.data_files = None
        self.init()

    def init(self):
        if not os.path.exists(self.data_dir):
            sys.exit('Data directory does not exist.')
        self.set_filenames()
        class_file = os.path.join(self.data_dir, 'classes')
        dump_to_file(class_file, self.class_names)

    def set_filenames(self):
        data_files = {}
        for f in os.listdir(self.data_dir):
            f = os.path.join(self.data_dir, f)
            if os.path.isfile(f):
                chunks = f.split('.')
                class_name = chunks[-1]
                if class_name in self.class_names + [UNLABELED_SUFFIX]:
                    data_files[class_name] = f
        assert data_files
        self.data_files = data_files

    def prepare_dict(self, vocab_size=10000, tokenize_level='word'):
        max_sent_len = 0
        c = Counter()
        # store the preprocessed raw text to avoid cleaning it again
        self.tok_text = defaultdict(list)
        for label, f in self.data_files.iteritems():
            #strings = []
            with codecs_open(f, 'r', encoding='utf-8') as infile:
                for line in infile:
                    if tokenize_level == 'char':
                        clean_string = tokenize_char(line)
                    else:
                        clean_string = tokenize_word(line)
                    toks = clean_string.split()
                    if len(toks) > max_sent_len:
                        max_sent_len = len(toks)
                    for t in toks:
                        c[t] += 1
                    self.tok_text[label].append(clean_string)
        total_words = len(c)
        assert total_words >= vocab_size
        word_list = [p[0] for p in c.most_common(vocab_size - 2)]
        word_list.insert(0, PAD_TOKEN)
        word_list.insert(0, UNK_TOKEN)
        self.word2freq = c
        self.word2id = dict()
        vocab_file = os.path.join(self.data_dir, 'vocab')
        #with open(vocab_file, 'w') as outfile:
        #    for idx, w in enumerate(word_list):
        #        self.word2id[w] = idx
        #        outfile.write(w + '\t' + str(idx) + '\n')
        for idx, w in enumerate(word_list):
            self.word2id[w] = idx
        self.word2id['max_sent_len'] = max_sent_len
        self.word2id['tokenize_level'] = tokenize_level
        dump_to_file(vocab_file, self.word2id)
        del self.word2id['max_sent_len']
        del self.word2id['tokenize_level']
        print '%d words found in training set. Truncate to vocabulary size %d.' % (total_words, vocab_size)
        print 'Dictionary saved to file %s. Max sentence length in data is %d.' % (vocab_file, max_sent_len)
        return max_sent_len

    def generate_id_data(self, max_sent_len=100):
        self.max_sent_len = max_sent_len
        #sentence_and_label_pairs = []
        self.id_text = defaultdict(list)
        for label, sequences in self.tok_text.iteritems():
            #label_id = [0] * self.num_classes
            #label_id[self.class_names.index(label)] = 1
            for seq in sequences:
                toks = seq.split()
                toks_len = len(toks)
                if toks_len <= max_sent_len:
                    pad_left = (max_sent_len - toks_len) / 2
                    pad_right = int(np.ceil((max_sent_len - toks_len) / 2.0))
                else:
                    continue
                toks_ids = [1 for _ in xrange(pad_left)] \
                           + [self.word2id[t] if t in self.word2id else 0 for t in toks] \
                           + [1 for _ in xrange(pad_right)]
                self.id_text[label].append(toks_ids)
                #sentence_and_label_pairs.append((toks_ids, label_id))
        #return sentence_and_label_pairs
        return

    def shuffle_and_split(self, test_fraction=0.1):
        #random.seed(RANDOM_SEED)
        #random.shuffle(sentence_and_label_pairs)
        #sentences, labels = zip(*sentence_and_label_pairs)
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        raw_x = []
        for label, sequences in self.id_text.iteritems():
            length = len(sequences)
            np.random.seed(RANDOM_SEED)
            permutation = np.random.permutation(length)
            raw_seq = self.tok_text[label]
            #random.shuffle(sequences)
            shuffled_id = [sequences[i] for i in permutation]
            shuffled_raw = [raw_seq[i] for i in permutation]
            test_num = int(length * test_fraction)
            train_num = length - test_num
            if label == UNLABELED_SUFFIX:
                test_num = 0
                train_num = length

            label_id = [0] * self.num_classes
            if label != UNLABELED_SUFFIX:
                label_id[self.class_names.index(label)] = 1

            #for i, seq in enumerate(shuffled_id):
            #    if i < test_num:
            #        test_x.append(seq)
            #        test_y.append(label_id)
            #    else:
            #        train_x.append(seq)
            #        train_y.append(label_id)

            test_x.extend(shuffled_id[:test_num])
            test_y.extend([label_id for _ in xrange(test_num)])
            train_x.extend(shuffled_id[test_num:])
            train_y.extend([label_id for _ in xrange(train_num)])
            raw_x.extend(shuffled_raw[test_num:])

        assert len(train_x) == len(train_y)
        assert len(train_x) == len(raw_x)
        assert len(test_x) == len(test_y)
        self.num_examples = len(test_y) + len(train_y)
        self.test_data = (test_x, test_y)
        self.train_data = (train_x, train_y)
        dump_to_file(os.path.join(self.data_dir, 'train.cPickle'), self.train_data)
        dump_to_file(os.path.join(self.data_dir, 'raw.cPickle'), raw_x)
        dump_to_file(os.path.join(self.data_dir, 'test.cPickle'), self.test_data)
        print 'Split dataset into train/test set: %d for training, %d for evaluation.' % \
              (len(train_y), len(test_y))
        return

    def prepare_data(self, vocab_size=10000, test_fraction=0.1, tokenize_level='word'):
        max_sent_len = self.prepare_dict(vocab_size, tokenize_level=tokenize_level)
        self.generate_id_data(max_sent_len)
        self.shuffle_and_split(test_fraction)
        return

class DataLoader(object):

    def __init__(self, data_dir, filename, batch_size=50, shuffle=True, load_raw=False):
        self._x = None
        self._y = None
        self.load_and_shuffle(data_dir, filename, shuffle=shuffle, load_raw=load_raw)

        self._pointer = 0

        self._num_examples = len(self._x)
        self.batch_size = batch_size
        self.num_batch = int(np.ceil(self._num_examples / float(self.batch_size)))
        self.sent_len = len(self._x[0])

        self.num_classes = len(self._y[0])
        self.class_names = load_from_dump(os.path.join(data_dir, 'classes'))
        assert len(self.class_names) == self.num_classes

        self.agreement = defaultdict(list)
        self.pool_flag = np.zeros(self._num_examples)

        print 'Loaded target classes (length %d).' % len(self.class_names)
        print 'Loaded data with %d examples. %d examples per batch will be used.' % \
              (self._num_examples, self.batch_size)

    def load_and_shuffle(self, data_dir, filename, shuffle=True, load_raw=False):
        _x, _y = load_from_dump(os.path.join(data_dir, filename))
        assert len(_x) == len(_y)
        length = len(_y)
        if load_raw:
            _raw_x = load_from_dump(os.path.join(data_dir, 'raw.cPickle'))
        if shuffle:
            np.random.seed(RANDOM_SEED)
            perm = np.random.permutation(length)
            _x = [_x[i] for i in perm]
            _y = [_y[i] for i in perm]
            if load_raw:
                _raw_x = [_raw_x[i] for i in perm]
            #random.seed(RANDOM_SEED)
            #data = zip(_x, _y)
            #random.shuffle(data)
            #_x, _y = zip(*data)
        self._x = np.array(_x)
        self._y = np.array(_y)
        if load_raw:
            self._raw_x = np.array(_raw_x)
        return

    def next_batch(self):
        if self.batch_size + self._pointer >= self._num_examples:
            batch_x, batch_y = self._x[self._pointer:], self._y[self._pointer:]
            #self._pointer = (self._pointer + self.batch_size) % self._num_examples
            #return (batch_x + self._x[:self._pointer], batch_y + self._y[:self._pointer])
            return (batch_x, batch_y)
        self._pointer += self.batch_size
        return (self._x[self._pointer-self.batch_size:self._pointer],
                self._y[self._pointer-self.batch_size:self._pointer])

    def reset_pointer(self):
        self._pointer = 0
        self.pool_flag = np.zeros(self._num_examples)

    def next_batch_idx_active(self, config):
        remain_examples = np.where(self.pool_flag == 0)[0]
        #print '%d / %d' % (len(remain_examples), self._num_examples)

        # first batch
        if self._num_examples == len(remain_examples):
            idx = range(self.batch_size)
        # last batch
        elif self.batch_size > len(remain_examples):
            idx = remain_examples
        else:
            pool_idx = remain_examples[:config['pool_size']]
            x_pool = np.vstack(tuple([np.array(self._x[i]) for i in pool_idx]))

            # get idx of k-most informative examples
            active_idx = most_informative(x_pool, config)
            idx = [pool_idx[i] for i in active_idx]

        # update flag
        for i in idx:
            self.pool_flag[i] = 1

        return idx

    def get_oracle(self, idx):
        y_batch = [self._y[i] for i in idx] # true label
        _raw_x = [self._x[i] for i in idx]

        _y = []
        for k in xrange(self.batch_size):
            input_class = {l: c for l, c in enumerate(self.class_names)}
            _raw_y = raw_input('\t%s\n\t%s\n\t' % (_raw_x[k], str(input_class)))
            try:
                _raw_y = int(_raw_y)
                if int(_raw_y) >= self.num_classes or int(_raw_y) < 0:
                    raise TypeError
            except TypeError:
                '\tKeyError:, "%s" not found. Use class 0: %s' % (_raw_y, input_class[0])
                _raw_y = 0

            _true_y = int(np.argmax(y_batch[k]))
            self.agreement[(self._pointer + k, _true_y)].append(int(_raw_y))

            # check agreement with gold standard
            if np.all(y_batch[k] == np.zeros(self.num_classes)):    #nonlabeled data
                pass
            elif _raw_y != _true_y:              #labeled data (mismatch)
                print '\t>>> Nope, "%s" is not correct. Use gold standard: "%s"\n' % \
                      (input_class[_raw_y], input_class[_true_y])
                _raw_y = _true_y
            else:                                                   #labeled data (match)
                print '\t>>> Nice, "%s" is correct.\n' % \
                      (input_class[_raw_y], input_class[_true_y])

            # vectorize label
            _id_y = [0] * self.num_classes
            _id_y[_raw_y] = 1
            _y.append(np.array(_id_y))

        return _y


def most_informative(x_pool, config):
    import predict
    import scipy.stats

    #print 'x_pool', x_pool.shape
    result = predict.predict(x_pool, config, raw_text=False)

    strategy = config['strategy']
    if strategy == 'max_entropy':
        prob = [scipy.stats.entropy(res) for res in result['scores']]
        idx = np.argsort(prob)
        idx = list(idx)[-1 * config['batch_size']:] # take last-k examples

    elif strategy == 'least_confident':
        prob = np.argmax(result['scores'], axis=1)
        idx = np.argsort(prob)
        idx = list(idx)[:config['batch_size']] # take first-k examples

    elif strategy == 'min_margin':
        prob = [s[-1] - s[-2] for s in np.sort(result['scores'], axis=1)]
        idx = np.argsort(prob)
        idx = list(idx)[:config['batch_size']] # take first-k examples

    return idx

class VocabLoader(object):

    def __init__(self, data_dir):
        self.word2id = None
        self.max_sent_len = None
        self.class_names = None
        self.tokenize_level = 'word'
        self.restore(data_dir)

    def restore(self, data_dir):
        class_file = os.path.join(data_dir, 'classes')
        self.class_names = load_from_dump(class_file)
        print 'Loaded target classes (length %d).' % len(self.class_names)

        vocab_file = os.path.join(data_dir, 'vocab')
        tmp = load_from_dump(vocab_file)
        self.max_sent_len = tmp['max_sent_len']
        self.tokenize_level = tmp['tokenize_level']
        del tmp['max_sent_len']
        del tmp['tokenize_level']
        self.word2id = tmp
        print 'Loaded vocabulary (size %d).' % len(self.word2id)

    def text2id(self, raw_text):
        """
        Generate id data from one raw sentence
        """
        if not self.max_sent_len:
            raise Exception('max_sent_len is not set.')
        if not self.word2id:
            raise Exception('word2id is not set.')
        max_sent_len = self.max_sent_len

        if self.tokenize_level == 'char':
            toks = tokenize_char(raw_text).split()
        else:
            toks = tokenize_word(raw_text).split()
        toks_len = len(toks)
        if toks_len <= max_sent_len:
            pad_left = (max_sent_len - toks_len) / 2
            pad_right = int(np.ceil((max_sent_len - toks_len) / 2.0))
        else:
            return None
        toks_ids = [1 for _ in xrange(pad_left)] + \
                   [self.word2id[t] if t in self.word2id else 0 for t in toks] + \
                   [1 for _ in xrange(pad_right)]
        return toks_ids

def tokenize_word(string):
    """
    Tokenization/string cleaning.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def sanitize_char(text):
    text = re.sub(ur'<.*?>', '', text)              # html tags in subtitles
    text = re.sub(ur'[0-9]', '', text)              # numbers
    text = re.sub(ur'[.,:|/"_\[\]()]', '', text)    # punctuations
    text = re.sub(ur'[♫♪%–]', '', text)             # cross-lingual symbols
    text = re.sub(ur' {2,}', ' ', text)             # more than two consective spaces
    return text.strip().lower()

def tokenize_char(text, bos=True, eos=True, ws_symbol=u'<ws>', bos_symbol=u'<s>', eos_symbol=u'</s>'):
    text = list(sanitize_char(text))
    text = ' '.join([x.replace(' ', ws_symbol) for x in text])
    if bos:
        text = bos_symbol + ' ' + text
    if eos:
        text = text + ' ' + eos_symbol
    return text

def dump_to_file(filename, obj):
    with open(filename, 'wb') as outfile:
        pickle.dump(obj, file=outfile)
    return

def load_from_dump(filename):
    with open(filename, 'rb') as infile:
        obj = pickle.load(infile)
    return obj

def _load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return (word_vecs, layer1_size)

def _add_random_vec(word_vecs, vocab, emb_size=300):
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,emb_size)
    return word_vecs

def prepare_pretrained_embedding(fname, word2id):
    print 'Reading pretrained word vectors from file ...'
    word_vecs, emb_size = _load_bin_vec(fname, word2id)
    word_vecs = _add_random_vec(word_vecs, word2id, emb_size)
    embedding = np.zeros([len(word2id), emb_size])
    for w,idx in word2id.iteritems():
        embedding[idx,:] = word_vecs[w]
    print 'Generated embeddings with shape ' + str(embedding.shape)
    return embedding

def load_language_codes():
    ret = {}
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'language_codes.csv')
    with codecs_open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if not line.startswith('#') and len(line.strip()) > 0:
                c = line.strip().split('\t')
                if len(c) > 1:
                    ret[c[0]] = c[1]
    return ret

def main():
    # language detection
    #data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'ted500')
    #class_names = ["ar", "az", "bg", "bn", "bo", "cs", "da", "de", "el", "en", "es",
    #               "fa", "fi", "fil", "fr", "gu", "he", "hi", "ht", "hu", "hy", "id",
    #               "is", "it", "ja", "ka", "km", "kn", "ko", "ku", "lt", "mg", "ml",
    #               "mn", "ms", "my", "nb", "ne", "nl", "nn", "pl", "ps", "pt", "ro",
    #               "ru", "si", "sk", "sl", "so", "sq", "sv", "sw", "ta", "te", "tg",
    #               "th", "tl", "tr", "ug", "uk", "ur", "uz", "vi", "zh-cn", "zh-tw"]
    #reader = TextReader(data_dir, class_names=class_names)
    #reader.prepare_data(vocab_size=4090, test_fraction=0.1, tokenize_level='char')

    # sentiment analysis
    data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'mr')
    class_names = ["neg", "pos"]
    reader = TextReader(data_dir, class_names=class_names)
    reader.prepare_data(vocab_size=15000, test_fraction=0.1, tokenize_level='word')
    embedding = prepare_pretrained_embedding('./data/word2vec/GoogleNews-vectors-negative300.bin', reader.word2id)
    np.save(os.path.join(data_dir, 'emb.npy'), embedding)


if __name__ == '__main__':
    main()