import numpy as np
import os

cache_path = "../../datasets/flickr/annotations/final_flickr_separateGT_train.cache"
# cache_path = "../../datasets/mixed_grounding/annotations/final_mixed_train_no_coco.cache"

labels = np.load(cache_path, allow_pickle=True)

new_labels = []
for label in labels:
    cls = label.pop("cls").reshape(-1).tolist()
    texts = label.pop("texts")
    texts = [t[0] for t in texts]
    label["texts"] = [texts[int(c)] for c in cls]
    if len(cls) != 0:
        label['im_file'] = '../' + str(label['im_file'])
        new_labels.append(label)

np.save('data/flickr/train.cache', new_labels)
# np.save('data/mixed_grounding/train.cache', new_labels)