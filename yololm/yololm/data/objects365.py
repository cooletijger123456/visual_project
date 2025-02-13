import copy

import torch

from yololm.utils import preprocess
from torch.utils.data import Dataset
import numpy as np
from ultralytics.data.augment import LetterBox
from ..utils import make_mask, process_image_box

class Objects365V1Det(Dataset):

    def __init__(self,
                 tokenizer,
                 cache_path,
                 data_args=None,
                 max_gt_per_img=100,
                 scale_factor=1/8
                 ):
        self.task_str = "<roc>"
        
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.max_gt_per_img = max_gt_per_img
        
        self.scale_factor = scale_factor
        self.labels = np.load(cache_path, allow_pickle=True)
        self.transform = LetterBox(new_shape=(data_args.image_scale, data_args.image_scale), scaleup=False)
    
    def process(self, label):
        image, bboxes = process_image_box(label['im_file'], label['bboxes'], label['normalized'], self.data_args.image_scale, self.transform, label['bbox_format'])
    
        texts = np.array(label['texts'])
        
        masksz = self.data_args.image_scale * self.scale_factor
        masks = make_mask(torch.from_numpy(bboxes) * self.scale_factor, masksz, masksz)
        
        valid_index = masks.sum(dim=(-1, -2)) > 0
        masks = masks[valid_index]
        texts = texts[valid_index.numpy()]
        
        if len(texts) == 0:
            assert(False)
            return None
        
        texts = ["<bbox>" + self.task_str +  t + self.tokenizer.eos_token for t in texts]

        data = preprocess(
            texts,
            self.task_str,
            self.tokenizer)

        data['image'] = image
        data['visuals'] = masks
        
        return data

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        assert(len(label["texts"]) != 0)
        label = self.process(copy.deepcopy(label))
        return label
