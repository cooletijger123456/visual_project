from ultralytics import YOLOE
import torch

model_name = "11l"
model = YOLOE(f"pretrain/yoloe-{model_name}-seg-pf.pt")

state = torch.load(f"pretrain/yoloe-{model_name}-seg-pf.pt")
state["train_args"]["task"] = "detect"
torch.save(state, f"pretrain/yoloe-{model_name}-seg-pf.pt")