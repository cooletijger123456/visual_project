from ultralytics import YOLOWorld
from ultralytics.models.yolo.detect import DetectionValidator

model = YOLOWorld("runs/final/yolov8l-vl-seg-pf/weights/best.pt")

filename = "ultralytics/cfg/datasets/lvis.yaml"

model.val(data=filename, batch=1, split='minival', rect=False, validator=DetectionValidator, single_cls=True)