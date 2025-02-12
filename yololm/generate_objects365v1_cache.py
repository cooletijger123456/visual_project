import numpy as np
from ultralytics.utils import yaml_load

labels = np.load('../../datasets/Objects365v1/labels/train.cache', allow_pickle=True).item()["labels"]
names = yaml_load('../ultralytics/cfg/datasets/Objects365v1.yaml')['names']
names = {k : v.split('/')[0] for k, v in names.items()}

new_labels = []
for label in labels:
    cls = label.pop("cls")
    texts = [names[c.item()] for c in cls]
    label["texts"] = texts
    if len(cls) != 0:
        new_labels.append(label)

np.save('data/Objects365v1/train.cache', new_labels)