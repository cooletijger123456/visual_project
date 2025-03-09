from ultralytics import YOLOE
from copy import deepcopy

model_name = "11l"
model = YOLOE(f"yoloe-{model_name}-seg.yaml")
model.load(f"runs/pretrain/yolo{model_name}-vl-seg/weights/best.pt")

vp_model = YOLOE(f"runs/pretrain/yolo{model_name}-vl-seg-vp/weights/best.pt")
model.model.model[-1].vpe = deepcopy(vp_model.model.model[-1].vpe)
model.eval()

model.save(f"pretrain/yoloe-{model_name}-seg.pt")
