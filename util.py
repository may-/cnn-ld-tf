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

# TODO: I need to clean up this preprocessing script a bit
class TextReader(object):

    def __init__(self, data_dir, class_names):
        self.data_dir = data_dir
        self.class_names = class_names
        self.num_classes = len(class_names)
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
                if class_name in self.class_names:
                    data_files[class_name] = f
        assert data_files
        self.data_files = data_files

    def prepare_dict(self, vocab_size=10000):
        max_sent_len = 0
        c = Counter()
        # store the preprocessed raw text to avoid cleaning it again
        self.tok_text = defaultdict(list)
        for label, f in self.data_files.iteritems():
            #strings = []
            with codecs_open(f, 'r', encoding='utf-8') as infile:
                for line in infile:
                    clean_string = tokenize(line)
                    #clean_string = clean_str(line)
                    #strings.append(clean_string)
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
        dump_to_file(vocab_file, self.word2id)
        del self.word2id['max_sent_len']
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
        random.seed(RANDOM_SEED)
        #random.shuffle(sentence_and_label_pairs)
        #sentences, labels = zip(*sentence_and_label_pairs)
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for label, sequences in self.id_text.iteritems():
            random.shuffle(sequences)
            test_num = int(len(sequences) * test_fraction)
            #train_num = len(sequences) - test_num

            label_id = [0] * self.num_classes
            label_id[self.class_names.index(label)] = 1

            for i, seq in enumerate(sequences):
                if i < test_num:
                    test_x.append(seq)
                    test_y.append(label_id)
                else:
                    train_x.append(seq)
                    train_y.append(label_id)
                    #test_x.expand(sequences[:test_num])
                    #test_y.expand([label_id for _ in xrange(test_num)])
                    #train_x.expand(sequences[test_num:])
                    #train_y.expand([label_id for _ in xrange(train_num)])
        assert len(train_x) == len(train_y)
        assert len(test_x) == len(test_y)
        self.num_examples = len(test_y) + len(train_y)
        self.test_data = (test_x, test_y)
        self.train_data = (train_x, train_y)
        dump_to_file(os.path.join(self.data_dir, 'train.cPickle'), self.train_data)
        dump_to_file(os.path.join(self.data_dir, 'test.cPickle'), self.test_data)
        print 'Split dataset into train/test set: %d for training, %d for evaluation.' % \
              (len(train_y), len(test_y))
        return

    def prepare_data(self, vocab_size=10000, test_fraction=0.1):
        max_sent_len = self.prepare_dict(vocab_size)
        self.generate_id_data(max_sent_len)
        self.shuffle_and_split(test_fraction)
        return

class DataLoader(object):

    def __init__(self, filename, batch_size=50, shuffle=True):
        self._x, self._y = self.load_and_shuffle(filename, shuffle)
        assert len(self._x) == len(self._y)
        self._pointer = 0
        self._num_examples = len(self._x)
        self.batch_size = batch_size
        self.num_batch = int(np.ceil(self._num_examples / self.batch_size))
        self.num_classes = len(self._y[0])
        self.sent_len = len(self._x[0])
        print 'Loaded data with %d examples. %d examples per batch will be used.' % \
              (self._num_examples, self.batch_size)

    def load_and_shuffle(self, filename, shuffle=True):
        _x, _y = load_from_dump(filename)
        if shuffle:
            random.seed(RANDOM_SEED)
            data = zip(_x, _y)
            random.shuffle(data)
            _x, _y = zip(*data)
        return np.array(_x), np.array(_y)

    def next_batch(self, loop=True):
        # reset pointer
        if self.batch_size + self._pointer >= self._num_examples:
            batch_x, batch_y = self._x[self._pointer:], self._y[self._pointer:]
            if loop:
                self._pointer = (self._pointer + self.batch_size) % self._num_examples
                return (batch_x + self._x[:self._pointer], batch_y + self._y[:self._pointer])
            else:
                return (batch_x, batch_y)
        self._pointer += self.batch_size
        return (self._x[self._pointer-self.batch_size:self._pointer],
                self._y[self._pointer-self.batch_size:self._pointer])

    def reset_pointer(self):
        self._pointer = 0

class VocabLoader(object):

    def __init__(self, data_dir):
        self.word2id = None
        self.max_sent_len = None
        self.class_names = None
        self.restore(data_dir)

    def restore(self, data_dir):
        class_file = os.path.join(data_dir, 'classes')
        self.class_names = load_from_dump(class_file)
        print 'Loaded target classes (length %d).' % len(self.class_names)

        vocab_file = os.path.join(data_dir, 'vocab')
        tmp = load_from_dump(vocab_file)
        self.max_sent_len = tmp['max_sent_len']
        del tmp['max_sent_len']
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

        toks = tokenize(raw_text).split()
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

def clean_str(string):
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

def sanitize(text):
    text = re.sub(ur'<.*?>', '', text)              # html tags in subtitles
    text = re.sub(ur'[0-9]', '', text)              # numbers
    text = re.sub(ur'[.,:|/"_\[\]()]', '', text)    # punctuations
    text = re.sub(ur'[♫♪%–]', '', text)             # cross-lingual symbols
    text = re.sub(ur' {2,}', ' ', text)             # more than two consective spaces
    return text.strip().lower()

def tokenize(text, bos=True, eos=True, ws_symbol=u'<ws>', bos_symbol=u'<s>', eos_symbol=u'</s>'):
    text = list(sanitize(text))
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

def main():
    data_dir = os.path.join(os.getcwd(), 'data', 'ted500')
    #language_codes = pd.read_csv(os.path.join(os.getcwd(), 'language_codes.csv'), sep='\t',
    #                          skip_blank_lines=True, comment='#', header=None, names=['code', 'language'])
    #languages = language_codes.keys()
    class_names = ["ar", "az", "bg", "bn", "bo", "cs", "da", "de", "el", "en", "es",
                   "fa", "fi", "fil", "fr", "gu", "he", "hi", "ht", "hu", "hy", "id",
                   "is", "it", "ja", "ka", "km", "kn", "ko", "ku", "lt", "mg", "ml",
                   "mn", "ms", "my", "nb", "ne", "nl", "nn", "pl", "ps", "pt", "ro",
                   "ru", "si", "sk", "sl", "so", "sq", "sv", "sw", "ta", "te", "tg",
                   "th", "tl", "tr", "ug", "uk", "ur", "uz", "vi", "zh-cn", "zh-tw"]
    reader = TextReader(data_dir, class_names=class_names)
    reader.prepare_data(vocab_size=4090, test_fraction=0.1)
    #embedding = prepare_pretrained_embedding('./data/word2vec/GoogleNews-vectors-negative300.bin', reader.word2id)
    #np.save(os.path.join(data_dir, 'emb.npy'), embedding)


if __name__ == '__main__':
    main()