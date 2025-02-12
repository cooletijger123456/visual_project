from ultralytics import YOLOWorld
from ultralytics.utils import yaml_load

model = YOLOWorld("last.pt")

filename = "ultralytics/cfg/datasets/lvis.yaml"
data = yaml_load(filename, append_filename=True)

names = [name.split("/")[0] for name in data["names"].values()]

# del model.model.pe
model.val(data=data['yaml_file'], batch=1, split='minival', rect=False, imgsz=800)
