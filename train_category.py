import argparse
import json
from lib.reader import UIUCReader


def test_dataset_iterator(dataset_loader, dataset_iterator):
    for i, batch in enumerate(dataset_iterator):
        if i > 3:
            break

        print('===== %d th batch =====' % i)
        print()
        print('batch info:')
        print(batch)
        print()
        for i in range(batch.batch_size):
            print('category: %d %s' % (batch.category[i], dataset_loader.dataset.fields['category'].vocab.itos[batch.category[i]]))
            for key in ['words']:
                print('%s:' % key)
                print(getattr(batch, key)[i])
                print('%s (decode):' % key)
                row = getattr(batch, key)[i]
                print('\t'.join([dataset_loader.dataset.fields[key].vocab.itos[w] for w in row]))
                print()
            print()


def main(config_path):
    with open(config_path, 'r') as fread:
        config_dict = json.load(fread)

    path_config = config_dict['Path']
    model_dir = path_config['model_dir']
    vocab_dir = path_config['vocab_dir']
    train = path_config['train']
    test = path_config['test']

    dataset_config = config_dict['Dataset']
    train_reader = UIUCReader(train, PAD_TOKEN='<pad>', **dataset_config)
    train_reader.build_vocabs(vocab_dir)
    train_iterator = train_reader.get_dataset_iterator()

    test_dataset_iterator(train_reader, train_iterator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str, help='Path of config file')

    args = parser.parse_args()
    main(args.config_path)