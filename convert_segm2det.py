from ultralytics import YOLOWorld
import torch

det_model = YOLOWorld("yolov8l-worldv2-vl.yaml")

state = torch.load("yolov8l-worldv2-vlhead-mobileclip-ladapterglu-imgsz800-alpha1-segm.pt")

det_model.load(state["model"])
det_model.save("yolov8l-worldv2-vlhead-mobileclip-ladapterglu-imgsz800-alpha1-segm-det.pt")

model = YOLOWorld("yolov8l-worldv2-vlhead-mobileclip-ladapterglu-imgsz800-alpha1-segm-det.pt")
print(model.args)
