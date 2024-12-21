import torch
import json
import numpy as np
import random

if __name__ == '__main__':
    with open('tools/global_grounding_neg_cat.json', 'r') as f:
        global_grounding_neg_cats = np.array(json.load(f))
    
    global_grounding_neg_embeddings = torch.load('tools/global_grounding_neg_embeddings.pt')
    
    cats = ["a dog", "a person"]
    train_label_embeddings = torch.load('tools/train_label_embeddings.pt')
    txt_feats = []
    for cat in cats:
        txt_feats.append(train_label_embeddings[cat])
    txt_feats = torch.stack(txt_feats, dim=0)
    
    pair_scores = txt_feats @ global_grounding_neg_embeddings.T
    
    pair_scores = pair_scores.amax(dim=0)
    pad_net_cat_indexs = np.array(torch.where(pair_scores < 0.8)[0])
    pad_net_cat_indexs = np.random.choice(pad_net_cat_indexs, size=80, replace=False)
    
    pad_net_cats = global_grounding_neg_cats[pad_net_cat_indexs]
    pad_net_cat_embeddings = global_grounding_neg_embeddings[pad_net_cat_indexs]
    
    print(pad_net_cat_embeddings.shape)
    
    