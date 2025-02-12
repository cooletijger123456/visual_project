import copy
from typing import Dict, Optional, Sequence

import numpy as np
import cv2
import torch
import transformers

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = '<pad>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = '<unk>'
DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'

### Task Tokens
DEFAULT_OD_TASK_TOKEN = '<roc>'
DEFAULT_CAP_TASK_TOKEN = '<cap>'
DEFAULT_OCR_TASK_TOKEN = '<ocr>'
DEFAULT_VQA_TASK_TOKEN = '<vqa>'

REGION_TOKEN_NUM = 1


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def add_spatial_token(tokenizer):
    spi_tokens = ['<bbox>', '<point>', '<roc>', '<cap>', '<dcap>', '<ocr>', '<vqa>']
    tokenizer.add_tokens(spi_tokens, special_tokens=True)
    return tokenizer


def add_pad_token(tokenizer, model):
    smart_tokenizer_and_embedding_resize({'pad_token': DEFAULT_PAD_TOKEN}, tokenizer, model)
    return tokenizer


def _tokenize_task_fn(texts: Sequence[str], 
                      tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized = tokenizer(
        texts,
        return_tensors='pt',
        padding='longest',
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = labels = tokenized.input_ids
    input_ids_lens = labels_lens = tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        texts: Sequence[str],
        task: str,
        tokenizer: transformers.PreTrainedTokenizer,
        questions_mask: Optional[Sequence[str]] = None,
) -> Dict:
    """Given a list of sources, each is a conversation list.

    This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """

    # NOTE: update <bbox> to 16 tokens
    texts = [t.replace('<bbox>', '<bbox>' * REGION_TOKEN_NUM) for t in texts]
    # tokenize conversations
    conversations_tokenized = _tokenize_task_fn(texts, tokenizer)
    input_ids = conversations_tokenized['input_ids']
    targets = copy.deepcopy(input_ids)

    # ignore tokens in label
    targets[targets == tokenizer.convert_tokens_to_ids(task)] = IGNORE_INDEX
    targets[targets == tokenizer.convert_tokens_to_ids('<bbox>')] = IGNORE_INDEX
    targets[targets == tokenizer.pad_token_id] = IGNORE_INDEX
    if questions_mask is not None:
        assert(False)
        assert len(questions_mask) == len(texts)
        for i, q in enumerate(questions_mask):
            if len(q) > 0:
                targets[i, REGION_TOKEN_NUM+1:REGION_TOKEN_NUM+1+len(tokenizer.encode(q))] = IGNORE_INDEX # ignore question (one bbox only)
    return dict(input_ids=input_ids, labels=targets)

# def make_mask(boxes, h, w):
#     x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
#     r = torch.arange(w)[None, None, :]  # rows shape(1,1,w)
#     c = torch.arange(h)[None, :, None]  # cols shape(1,h,1)

#     return ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def make_mask(boxes, h, w):
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h)[None, :, None]  # cols shape(1,h,1)

    mask = ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
    
    empty_index = (mask.sum(dim=(-1, -2)) == 0)
    mask[empty_index] = ((r >= x1[empty_index].floor()) * (r < x2[empty_index].ceil()) * (c >= y1[empty_index].floor()) * (c < y2[empty_index].ceil()))
    
    return mask

import math
from ultralytics.utils.instance import Instances

def process_image_box(img_path, bboxes, normalize, imgsz, transform, bbox_format):
    image = cv2.imread(img_path)
    h0, w0 = image.shape[:2]  # orig hw
    
    if not normalize:
        bboxes[:, 0::2] /= w0
        bboxes[:, 1::2] /= h0
    
    r = imgsz / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        w, h = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    label = dict(
        img=image,
        instances=Instances(bboxes, np.zeros((0, 1000, 2), dtype=np.float32), None, bbox_format=bbox_format, normalized=True)
    )
    
    label = transform(label)
    image = label['img']
    instances = label.pop("instances")
    instances.convert_bbox(format='xyxy')
    h, w = image.shape[:2]
    instances.denormalize(w, h)
    bboxes = instances.bboxes
    
    image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)[::-1])).float() / 255
    
    return image, bboxes

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
