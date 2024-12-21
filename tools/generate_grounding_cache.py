import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from ultralytics.utils.ops import xyxy2xywhn
from argparse import ArgumentParser
import os

def count_instances(labels):
    instance_count = 0
    for label in labels:
        instance_count += label["bboxes"].shape[0]
    print("Instance count: ", instance_count)    

def generate_cache(json_path, img_path):
    labels = []
    with open(json_path) as f:
        annotations = json.load(f)
    images = {f'{x["id"]:d}': x for x in annotations["images"]}

    # Note: retain images without annotations
    img_to_anns = {int(k): [] for k in images.keys()}

    for ann in annotations["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)
        
    for img_id, anns in tqdm(img_to_anns.items()):
        img = images[f"{img_id:d}"]
        h, w, f = img["height"], img["width"], img["file_name"]
        im_file = Path(img_path) / f
        if not im_file.exists():
            continue
        bboxes = []
        cat2id = {}
        texts = []
        for ann in anns:
            if ann["iscrowd"]:
                continue
            box = np.array(ann["bbox"], dtype=np.float64)
            
            box[2] += box[0]
            box[3] += box[1]
            
            box = xyxy2xywhn(box, w=float(w), h=float(h), clip=True)
            
            if box[2] <= 0 or box[3] <= 0:
                continue

            cat_name = " ".join([img["caption"][t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
            if not cat_name:
                continue
            
            if cat_name not in cat2id:
                cat2id[cat_name] = len(cat2id)
                texts.append([cat_name])
            cls = cat2id[cat_name]  # class
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)
        lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)
        labels.append(
            {
                "im_file": im_file,
                "shape": (h, w),
                "cls": lb[:, 0:1],  # n, 1
                "bboxes": lb[:, 1:],  # n, 4
                "normalized": True,
                "bbox_format": "xywh",
                "texts": texts,
            }
        )
    count_instances(labels)
    
    cache_path = Path(json_path).with_suffix('.cache')
    np.save(str(cache_path), labels)
    cache_path.with_suffix(".cache.npy").rename(cache_path)
    print(f"Save {json_path} cache file {cache_path}") 
    return labels

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--json-path', required=True, type=str)
    parser.add_argument('--img-path', required=True, type=str)
    
    os.environ["PYTHONHASHSEED"] = "0"
    args = parser.parse_args()
    generate_cache(args.json_path, args.img_path)