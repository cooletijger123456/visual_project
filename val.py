from ultralytics import YOLOWorld
from ultralytics.utils import yaml_load

model = YOLOWorld("yolov8l-worldv2-vl.yaml")
model.load("runs/detect/train4/weights/best.pt")
model.eval()

filename = "ultralytics/cfg/datasets/lvis.yaml"
data = yaml_load(filename, append_filename=True)

names = [name.split("/")[0] for name in data["names"].values()]
model.set_classes(names)
model.val(data=data['yaml_file'], batch=1, split='minival', rect=False, imgsz=800)
