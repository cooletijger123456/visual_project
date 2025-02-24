from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_pe import WorldPETrainer, WorldPESegTrainer
import os
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import yaml_load, LOGGER
import torch

os.environ["PYTHONHASHSEED"] = "0"

data = "ultralytics/cfg/datasets/coco.yaml"

model_path = "yolov8s-worldv2-vl-seg.yaml"

scale = guess_model_scale(model_path)
cfg_dir = "ultralytics/cfg"
default_cfg_path = f"{cfg_dir}/default.yaml"
extend_cfg_path = f"{cfg_dir}/coco_{scale}_train.yaml"
defaults = yaml_load(default_cfg_path)
extends = yaml_load(extend_cfg_path)
assert(all(k in defaults for k in extends))
LOGGER.info(f"Extends: {extends}")

model = YOLOWorld("yolov8s-vl-seg.pt")

# Ensure pe is set for classes
names = list(yaml_load(data)['names'].values())
tpe = model.get_text_pe(names)
pe_path = "coco-pe.pt"
torch.save({"names": names, "pe": tpe}, pe_path)

# freeze = [str(f) for f in range(0, 22)]
# for name, child in model.model.model[-1].named_children():
#     if 'cv3' not in name:
#         freeze.append(f"22.{name}")

# freeze.extend(["22.cv3.0.0", "22.cv3.0.1", "22.cv3.1.0", "22.cv3.1.1", "22.cv3.2.0", "22.cv3.2.1"])
        
# model.train(data=data, batch=128, epochs=5, **extends, close_mosaic=1, \
#     optimizer='AdamW', lr0=2e-3, warmup_bias_lr=0.0, \
#         weight_decay=0.025, momentum=0.9, workers=4, \
#         trainer=WorldPETrainer, device='0,1,2,3,4,5,6,7', train_pe_path=pe_path)

model.train(data=data, epochs=160, close_mosaic=10, batch=128, 
            optimizer='AdamW', lr0=1e-3, warmup_bias_lr=0.0, \
            weight_decay=0.025, momentum=0.9, workers=4, \
            device="0,1,2,3,4,5,6,7", **extends, \
            trainer=WorldPESegTrainer, train_pe_path=pe_path)