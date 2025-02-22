from ultralytics import YOLOWorld

model = YOLOWorld("yolov8l-worldv2-vl-seg.yaml")
model.load("runs/final/yolov8l-vl-seg-omf/weights/best.pt")
model.eval()

filename = "ultralytics/cfg/datasets/coco.yaml"

model.val(data=filename, batch=1)