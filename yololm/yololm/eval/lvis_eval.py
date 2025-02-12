import argparse
import torch
import os
import random
import json
from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer
from yololm.models.yololm import YOLOForCausalLM
from yololm.utils import preprocess, add_pad_token, add_spatial_token
from yololm.models.utils import KeywordsStoppingCriteria
import numpy as np
from sentence_transformers import SentenceTransformer, util
import argparse
from transformers import logging
import torch.multiprocessing as mp
from ultralytics.data.augment import LetterBox
from yololm.utils import make_mask, process_image_box

logging.set_verbosity_error()

class LVIS_PACO_EVAL():
    def __init__(self, model_path, bert_model):
        model_path = os.path.expanduser(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="right",
            use_fast=True
        )

        self.model = YOLOForCausalLM.from_pretrained(
                                                model_path,
                                                torch_dtype=torch.float16,
                                                ).cuda()
        self.model.eval()

        self.tokenizer = add_pad_token(self.tokenizer, self.model)
        self.tokenizer = add_spatial_token(self.tokenizer)
        
        for m in self.model.modules():
            m.tokenizer = self.tokenizer

        vision_tower = self.model.get_model().vision_tower
        vision_tower.to(device='cuda', dtype=torch.float16)
        
        self.bert_model = SentenceTransformer(bert_model)
        
        self.scale_factor = 1/8
        self.imgsz = 800
        self.transform = LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)

    def eval(self, root_path, ann_file, total, rank, verbose=False):
        all_sim = 0
        all_num = 0
        all_iou = 0

        data_all = json.load(open(ann_file))
        data_all = np.array_split(data_all, total)[rank].tolist()

        for data in tqdm(data_all):
            img_path = os.path.join(root_path, data['file_name'])
            categories = []
            for category in data['categories']:
                category = category.replace('_', ' ')
                category = category.replace(':', ' ')
                categories.append(category)

            bboxes = []
            for ann in data['annotations']:
                x, y, w, h = ann['bbox']
                bbox = [x, y, x+w, y+h]
                bboxes.append(bbox)
            bboxes = torch.tensor(bboxes)

            init_inputs = self.get_init_inputs(img_path,
                                               bboxes=bboxes,
                                               question="<roc>")
            input_ids = init_inputs['input_ids'].cuda()
            image = init_inputs['image'].unsqueeze(0).half().cuda()
            visuals = [init_inputs['visuals'].half().cuda()]

            stop_str = self.tokenizer.eos_token
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            self.model.model.tokenizer = self.tokenizer

            with torch.inference_mode():

                self.model.orig_forward = self.model.forward
                self.model.forward = partial(self.model.orig_forward,
                                            visuals=visuals)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    output_ids = self.model.generate(
                        input_ids,
                        images=image,
                        do_sample=True,
                        temperature=0.2,
                        max_new_tokens=1024,
                        use_cache=True,
                        num_beams=1,
                        stopping_criteria=[stopping_criteria]
                    )
                self.model.forward = self.model.orig_forward

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(
                    f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                                skip_special_tokens=True)

            final_outputs = []
            for output in outputs:
                output = output.strip()
                if output.endswith(stop_str):
                    output = output[:-len(stop_str)]
                output = output.strip()
                if ':' in output:
                    output = output.split(':')[1]

                output = output.replace('.', ' ')
                output = output.replace(':', ' ')
                output = output.replace(',', ' ')
                final_outputs.append(output.strip())

            outputs_embeddings = self.bert_model.encode(final_outputs, convert_to_tensor=True)
            class_sentence_embeddings = self.bert_model.encode(categories, convert_to_tensor=True)
            cosine_scores = util.cos_sim(outputs_embeddings, class_sentence_embeddings)

            semantic_iou = 0
            for output, category in zip(final_outputs, categories):
                semantic_iou += SemanticIOU(output.lower(), category.lower())

            all_sim += cosine_scores.diagonal().sum().item()
            all_iou += semantic_iou
            all_num += len(final_outputs)
                
            if verbose:
                print("[pred | gt]: ", final_outputs, "|", categories)
                print("sim:{}, iou:{}".format(all_sim/all_num, all_iou/all_num))

        # print("final sim:{}, semantic iou:{}".format(all_sim/all_num, all_iou/all_num))
        return all_sim, all_iou, all_num
    
    def get_init_inputs(self,
                        img_path,
                        bboxes,
                        question):
        image, bboxes = process_image_box(img_path, bboxes, False, self.imgsz, self.transform, "xyxy")

        masksz = self.imgsz * self.scale_factor
        masks = make_mask(bboxes * self.scale_factor, masksz, masksz)
        if not (torch.all(masks.sum(dim=(-1, -2)) > 0)):
            print((bboxes * self.scale_factor)[masks.sum(dim=(-1, -2)) == 0])
            exit()

        texts = ["<bbox>" + question for _ in range(len(bboxes))]
        data = preprocess(texts, "", self.tokenizer)
        data['image'] = image
        data['visuals'] = masks
        
        return data
         

def SemanticIOU(value: list[str], target: list[str]) -> None:

    intersection = len(set(value.split()) & set(target.split()))
    union = len(set(value.split()) | set(target.split()))

    return intersection / union


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker(args):
    model, bert, img, json, total, rank, device, seed, verbose = args
    setup_seed(seed)
    torch.cuda.set_device(int(device))
    
    lvis_paco_eval = LVIS_PACO_EVAL(model, bert)
    result = lvis_paco_eval.eval(img, json, total, rank, verbose=(rank == 0) and verbose)
    print(f"rank[{rank}] results: {result}")
    return result


def main(args):
    devices = args.devices.split(",")
    ranks = list(range(len(devices)))
    total = len(devices)
    
    mp.set_start_method('spawn')
    with mp.Pool(total) as pool:
        results = pool.map(worker,
                           [(args.model, args.bert, args.img, args.json,
                             total, rank, device, args.seed, args.verbose)
                            for rank, device in zip(ranks, devices)])

    all_sim, all_iou, all_num = map(sum, zip(*results))
    print("final sim:{}, semantic iou:{}".format(all_sim/all_num, all_iou/all_num))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='osprey demo', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to osprey model', default='./exp4/stage1')
    parser.add_argument('--bert', help='path to bert model', default='all-MiniLM-L6-v2')
    parser.add_argument('--img', help='path to coco imgs', default='./data/coco')
    parser.add_argument('--json', help='path to lvis/paco val json file', default='./data/eval/lvis_val_1k_category.json')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--devices", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument('--verbose', help='verbose', action='store_true')
    args = parser.parse_args()

    main(args)