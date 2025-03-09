from ultralytics import YOLOE
from copy import deepcopy

model_name = "11l"
model = YOLOE(f"runs/pretrain/yolo{model_name}-vl-seg-pf/weights/best.pt")
model.model.model[-1].is_fused = True

model.load(f"runs/pretrain/yolo{model_name}-vl-seg/weights/best.pt")
del model.model.model[-1].vpe

model.eval()

model.save(f"pretrain/yoloe-{model_name}-seg-pf.pt")
