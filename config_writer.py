import json
from collections import OrderedDict
import os

model_dir = 'models/category/1116baseline'

config_dict = OrderedDict()

config_dict['Path'] = OrderedDict({
    'model_dir': model_dir,
    'train': 'dataset/semlink_category/train_5500.label',
    'test': 'dataset/semlink_category/TREC_10.label.txt',
    'vocab_dir': os.path.join(model_dir, 'vocab'),
})

config_dict['Dataset'] = OrderedDict({
    'batch_size': 32,
    'pad_size': 60,
})

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

with open(os.path.join(model_dir, 'config.json'), 'w') as fwrite:
    json.dump(config_dict, fwrite, indent=True)