from dataclasses import dataclass

import torch
import transformers
from torch.utils.data import ConcatDataset

from yololm.utils import IGNORE_INDEX

from .objects365 import Objects365V1Det
from .goldg import GoldGDet
from ..configs.tasks import spi_datasets

@dataclass
class DataCollatorForDetDataset(object):

    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances):
        
        input_ids, labels, visuals = tuple([instance.get(key,None) for instance in instances]
                                  for key in ('input_ids',
                                              'labels',
                                              'visuals'))

        # dynamic batch size (each box as a batch)
        batch_input_ids = []
        batch_labels = []
        for input_id in input_ids:
            batch_input_ids.extend(x for x in input_id)
        for label in labels:
            batch_labels.extend(x for x in label)

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=batch_input_ids,
            labels=batch_labels,
            attention_mask=batch_input_ids.ne(self.tokenizer.pad_token_id),
            visuals=visuals
        )

        images = [instance['image'] for instance in instances]
        batch['images'] = torch.stack(images)

        return batch


def make_multitask_data_module(tokenizer,
                                data_args) :

    train_dataset = build_spi_dataset(spi_datasets,
                            tokenizer=tokenizer,
                            data_args=data_args)

    data_collator = DataCollatorForDetDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def build_spi_dataset(dataset_config,
                  tokenizer=None,
                  data_args=None,
                  **kwargs):
    if isinstance(dataset_config, list):
        datasets = []
        for cfg in dataset_config:
            temp_dataset = build_spi_dataset(cfg,
                                    tokenizer=tokenizer,
                                    data_args=data_args,
                                    **kwargs)
            datasets.append(temp_dataset)

        for dataset in datasets:
            print(type(dataset), f'len = {len(dataset)}')

        return ConcatDataset(datasets)

    dataset_type = dataset_config.pop('type')
    if dataset_type == 'objects365':
        dataset = Objects365V1Det(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    elif dataset_type in ["flickr", "mixed_grounding"]:
        dataset = GoldGDet(
            **dataset_config,
            tokenizer=tokenizer,
            data_args=data_args,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return dataset
