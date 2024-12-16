from ultralytics import YOLOWorld
from ultralytics.utils import yaml_load

model = YOLOWorld("yolov8m-worldv2.pt")

filename = "ultralytics/cfg/datasets/lvis.yaml"
data = yaml_load(filename, append_filename=True)

names = [name.split("/")[0] for name in data["names"].values()]
model.set_classes(names)

model.val(data=data['yaml_file'], batch=1, split='minival', rect=False)
