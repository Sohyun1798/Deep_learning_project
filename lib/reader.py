import os
import pickle

import numpy as np
import spacy
from abc import abstractmethod

import torch
from torchtext.data import Field, Example, Dataset, Iterator


class BaseReader(object):
    def __init__(self, datafile, PAD_TOKEN='<pad>', pad_size=60):
        self.pad_size = pad_size
        self.PAD_TOKEN = PAD_TOKEN
        self.nlp = spacy.load('en')

        self.datafile = datafile
        self.dataset = self.load_dataset()
        self.folds = None

    def load_dataset(self):
        fields = self._get_fields()
        data_dict_generator = self._data_dict_generator()

        examples = [Example.fromdict(data_dict, fields) for data_dict in data_dict_generator]
        fields = {k: v for k, v in fields.values()}
        return Dataset(examples, fields)

    def build_vocabs(self, vocab_dir):
        fields = self.dataset.fields
        for name, field in fields.items():
            if not field.use_vocab:
                continue

            if not os.path.isdir(vocab_dir):
                os.makedirs(vocab_dir)

            path = os.path.join(vocab_dir, name)
            if os.path.isfile(path):
                with open(path, 'rb') as fread:
                    field.vocab = pickle.load(fread)
            else:
                field.build_vocab(self.dataset)
                with open(path, 'wb') as fwrite:
                    pickle.dump(field.vocab, fwrite)

    def set_vocabs(self, vocabs):
        for name, vocab in vocabs.items():
            if name in self.dataset.fields:
                self.dataset.fields[name].vocab = vocab

    def get_vocabs(self):
        vocabs = {}

        for name, field in self.dataset.fields.items():
            if field.use_vocab:
                vocabs[name] = field.vocab

        return vocabs

    def get_dataset_iterator(self, batch_size, **kwargs):
        return Iterator(self.dataset, batch_size, **kwargs)

    def get_cross_val_dataset_iterator(self, batch_size, k_fold=5, **kwargs):
        if self.folds is None:
            total_len = len(self.dataset)
            fold_idx = [total_len // k_fold * i_fold for i_fold in range(k_fold)]
            fold_idx.append(total_len)

            folds = []
            for i in range(k_fold):
                folds.append(Dataset(self.dataset.examples[fold_idx[i]:fold_idx[i+1]], self.dataset.fields))
            self.folds = folds

        return [Iterator(fold, batch_size, **kwargs) for fold in self.folds]

    def get_vocab(self, key):
        return self.dataset.fields[key].vocab

    @abstractmethod
    def _data_dict_generator(self):
        raise NotImplementedError('You should use child class')

    @abstractmethod
    def _get_fields(self):
        raise NotImplementedError('You should use child class')

    # @abstractmethod
    # def __iter__(self):
    #     raise NotImplementedError('You should use child class')
    #
    #
    # @classmethod
    # def build_vocab(cls, token_set, vocab_path):
    #     vocab_dir = os.path.dirname(vocab_path)
    #     if not os.path.isdir(vocab_dir):
    #         os.makedirs(vocab_dir)
    #
    #     if os.path.isdir(vocab_path):
    #         vocab = spacy.vocab.Vocab().from_disk(vocab_path)
    #     else:
    #         vocab = spacy.vocab.Vocab(strings=list(token_set))
    #         vocab.to_disk(vocab_path)
    #
    #     itos = ['<pad>'] + [k.text for k in vocab]
    #     stoi = defaultdict(lambda: 0)
    #     for i, k in enumerate(itos):
    #         stoi[k] = i
    #
    #     return vocab, itos, stoi


class UIUCReader(BaseReader):
    def __init__(self, datafile, PAD_TOKEN='<pad>', pad_size=60):
        super(UIUCReader, self).__init__(datafile, PAD_TOKEN=PAD_TOKEN, pad_size=pad_size)
        self.nlp = spacy.load('en')

    def preprocess(self, sentence):
        doc = self.nlp(sentence)
        return [token.lower_ for token in doc]

    def _data_dict_generator(self):
        with open(self.datafile, 'r') as fread:
            for line in fread:
                category, sentence = line.strip().split(maxsplit=1)
                category = category.split(':')[0]
                words = self.preprocess(sentence)
                yield {'category': category, 'words': words}

    def _get_fields(self):
        fields = {
            'words': ('words', Field(pad_token=self.PAD_TOKEN, batch_first=True, sequential=True, use_vocab=True, fix_length=self.pad_size)),
            'category': ('category', Field(batch_first=True, sequential=False, use_vocab=True, unk_token=None)),
        }

        return fields


class QFocusReader(BaseReader):
    def __init__(self, datafile, PAD_TOKEN='<pad>', pad_size=60):
        super(QFocusReader, self).__init__(datafile, PAD_TOKEN=PAD_TOKEN, pad_size=pad_size)
        self.nlp = spacy.load('en')

    def preprocess(self, sentence):
        doc = self.nlp(sentence)
        return [token.lower_ for token in doc]

    def _data_dict_generator(self):
        with open(self.datafile, 'r') as fread:
            for line in fread:
                raw_words = self.preprocess(line.strip())

                if raw_words[0] == 'IMPL'.lower() or raw_words.count('#') != 1:
                    continue

                focus_index = -1
                words = []
                for i, raw_word in enumerate(raw_words):
                    if raw_word == '#':
                        focus_index = i
                    else:
                        words.append(raw_word)

                yield {'focus': focus_index, 'words': words}

    def _get_fields(self):
        fields = {
            'words': ('words', Field(pad_token=self.PAD_TOKEN, batch_first=True, sequential=True, use_vocab=True, fix_length=self.pad_size)),
            'focus': ('focus', Field(batch_first=True, sequential=False, use_vocab=False))
        }

        return fields


class WikiqaPairReader(BaseReader):
    def __init__(self, datafile, category_clf, focus_clf, word_stoi, category_itos,
                 PAD_TOKEN='<pad>', pad_size=60):
        # trick for use vocab in data load
        self.word_stoi = word_stoi
        self.category_itos = category_itos

        self.category_ne_map = {
            'HUM': { 'PERSON' },
            'LOC': { 'LOC', 'GPE' },
            'NUM': { 'DATE', 'TIME', 'PERCENT', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'MONEY' },
            'ENTY': { 'NORP', 'ORG', 'FAC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE' },
        }
        self.category_ne_map['DESC'] = self.category_ne_map['NUM'] | self.category_ne_map['ENTY']
        self.category_ne_map['ABBR'] = self.category_ne_map['ENTY'].copy()

        self.category_clf = category_clf
        self.focus_clf = focus_clf

        super(WikiqaPairReader, self).__init__(datafile, PAD_TOKEN=PAD_TOKEN, pad_size=pad_size)

    def preprocess(self, sentence, answer=False):
        doc = self.nlp(sentence)

        if answer:
            return ([token.lower_ for token in doc], [token.ent_type_ for token in doc])
        else:
            return [token.lower_ for token in doc]

    def _data_dict_generator(self):
        with open(self.datafile, 'r') as fread:
            next(fread)
            for i, line in enumerate(fread):
                if i > 6:
                    break # Debugging

                cells = line.split('\t')
                question = cells[1]
                answer = cells[5]
                label = int(cells[6])

                q_words = self.preprocess(question)
                a_words, a_nes = self.preprocess(answer, answer=True)

                over_words = set(q_words) & set(a_words)
                q_word_over = [int(word in over_words) for word in q_words]
                a_word_over = [int(word in over_words) for word in a_words]

                input_q_words = np.zeros((1, self.pad_size), dtype=np.int)
                input_q_words[0, :len(q_words)] = [self.word_stoi[word] for word in q_words]
                category = self.category_clf.predict(torch.from_numpy(input_q_words))[0] + 1
                focus = self.focus_clf.predict(torch.from_numpy(input_q_words))[0]

                q_sem_over = [0] * len(q_words)
                if focus < len(q_words):
                    q_sem_over[focus] = category

                category_str = self.category_itos[category]
                a_sem_over = [0] * len(a_words)
                for i, a_ne in enumerate(a_nes):
                    if a_ne in self.category_ne_map[category_str]:
                        a_sem_over[i] = category

                yield {
                    'q_words': q_words, 'a_words': a_words, 'q_word_over': q_word_over, 'a_word_over': a_word_over,
                    'q_sem_over': q_sem_over, 'a_sem_over': a_sem_over, 'label': label
                }


    def _get_fields(self):
        fields = {
            'q_words': ('q_words', Field(pad_token=self.PAD_TOKEN, batch_first=True, sequential=True, use_vocab=True, fix_length=self.pad_size)),
            'a_words': ('a_words', Field(pad_token=self.PAD_TOKEN, batch_first=True, sequential=True, use_vocab=True, fix_length=self.pad_size)),
            'q_word_over': ('q_word_over', Field(pad_token=0, batch_first=True, sequential=True, use_vocab=False, fix_length=self.pad_size)),
            'a_word_over': ('a_word_over', Field(pad_token=0, batch_first=True, sequential=True, use_vocab=False, fix_length=self.pad_size)),
            'q_sem_over': ('q_sem_over', Field(pad_token=0, batch_first=True, use_vocab=False, sequential=True, fix_length=self.pad_size)),
            'a_sem_over': ('a_sem_over', Field(pad_token=0, batch_first=True, use_vocab=False, sequential=True, fix_length=self.pad_size)),
            'label': ('label', Field(batch_first=True, sequential=False, use_vocab=False))
        }

        return fields


def test_dataset_iterator(dataset_reader, dataset_iterator, keys):
    for i, batch in enumerate(iter(dataset_iterator)):
        if i > 3:
            break

        print('===== %d th batch =====' % i)
        print()
        print('batch info:')
        print(batch)
        print()

        for i in range(batch.batch_size):
            for key in keys:
                field = dataset_reader.dataset.fields[key]
                sequential = field.sequential
                use_vocab = field.use_vocab

                value = getattr(batch, key)[i]

                if sequential:
                    print('%s:' % key)
                    print(value)

                    if use_vocab:
                        print('%s (decode):' % key)
                        print('\t'.join([field.vocab.itos[v] for v in value]))

                else:
                    if use_vocab:
                        print('%s: %d %s' % (key, value, field.vocab.itos[value]))
                    else:
                        print('%s: %d' % (key, value))

                print()
            print()

        return