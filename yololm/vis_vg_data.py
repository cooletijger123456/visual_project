import json
from tqdm import tqdm

with open('vg_test_coco_format_preds.json', 'r') as f:
    data = json.load(f)

# print(len(data['images']))

for anno in data[:10]:
    # if '/63.jpg' in anno['image']:
    #     print(anno)
    print(anno)