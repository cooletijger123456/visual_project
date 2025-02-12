import json
import os

# file = 'lvis_val_1k_category.json'
file = 'paco_val_1k_category.json'

path = f'data/eval/{file}'

with open(path, 'r') as f:
    data = json.load(f)
    
for d in data:
    is_train = os.path.exists('data/coco/train2017/' + d['file_name'])
    is_val = os.path.exists('data/coco/val2017/' + d['file_name'])

    assert(is_train or is_val)
    assert(not (is_train and is_val))
    
    if is_train:
        d['file_name'] = 'train2017/' + d['file_name']
    else:
        d['file_name'] = 'val2017/' + d['file_name']

with open(f'{file}', 'w') as f:
    json.dump(data, f)