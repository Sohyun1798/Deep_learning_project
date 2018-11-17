import os
import pickle
import spacy
from abc import abstractmethod

from torchtext.data import Field, Example, Dataset, Iterator


class BaseReader(object):
    def __init__(self, datafile, PAD_TOKEN='<pad>', pad_size=60):
        self.pad_size = pad_size
        self.PAD_TOKEN = PAD_TOKEN
        self.nlp = spacy.load('en')

        self.dataset = self.load_dataset(datafile)

    def load_dataset(self, datafile):
        fields = self._get_fields()
        data_dict_generator = self._data_dict_generator(datafile)

        examples = [Example.fromdict(data_dict, fields) for data_dict in data_dict_generator]
        fields = {k: v for k, v in fields.values()}
        return Dataset(examples, fields)

    def build_vocabs(self, vocab_dir):
        fields = self.dataset.fields
        for name, field in fields.items():
            if not field.use_vocab:
                continue

            path = os.path.join(vocab_dir, name)
            if os.path.isfile(path):
                with open(path, 'rb') as fread:
                    field.vocab = pickle.load(fread)
            else:
                field.build_vocab(self.dataset)
                with open(path, 'wb') as fwrite:
                    pickle.dump(field.vocab, fwrite)

    def set_vocabs(self, vocabs):
        fields = self.dataset.fields
        for name, field in fields.items():
            field.vocab = vocabs[name]

    def get_vocabs(self):
        return {name: field.vocab for name, field in self.dataset.fields.items()}

    def get_dataset_iterator(self, batch_size, **kwargs):
        return iter(Iterator(self.dataset, batch_size, **kwargs))

    def get_vocab(self, key):
        return self.dataset.fields[key].vocab

    @abstractmethod
    def _data_dict_generator(self, datafile):
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
    def __init__(self, datafile, PAD_TOKEN='<pad>', batch_size=32, pad_size=60):
        super(UIUCReader, self).__init__(datafile, PAD_TOKEN=PAD_TOKEN, pad_size=pad_size)
        self.nlp = spacy.load('en')

    def preprocess(self, sentence):
        doc = self.nlp(sentence)
        return [token.lower_ for token in doc]

    def _data_dict_generator(self, datafile):
        with open(datafile, 'r') as fread:
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