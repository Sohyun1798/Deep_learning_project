import json
from collections import OrderedDict
import os

model_dir = 'models/baseline/category'

config_dict = OrderedDict()

config_dict['Path'] = OrderedDict({
    'model_dir': model_dir,
    'train': 'dataset/semlink_category/train_5500.label',
    'test': 'dataset/semlink_category/TREC_10.label.txt',
    'vocab_dir': os.path.join(model_dir, 'vocab'),
    'test_result': os.path.join(model_dir, 'test_result.txt')
})

config_dict['Dataset'] = OrderedDict({
    'batch_size': 50,
    'pad_size': 60,
})

config_dict['Model'] = OrderedDict({
    'embed': 'embedding/aquaint+wiki.txt.gz.ndim=50.bin',
    'conv_widths': [1,2,3],
    'hidden_size': 100,
})

config_dict['Train'] = OrderedDict({
    'epoch': 500,
    'weight_decay': 1e-4,
    'lr': 5e-4,
    'num_filters': 300,
})

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

with open(os.path.join(model_dir, 'config.json'), 'w') as fwrite:
    json.dump(config_dict, fwrite, indent=True)