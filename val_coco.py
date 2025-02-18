from ultralytics import YOLOWorld

model = YOLOWorld("yolov8l-worldv2-vl.yaml")
model.load("runs/final/yolov8l-vl-seg/weights/best.pt")
model.eval()

filename = "ultralytics/cfg/datasets/coco.yaml"

model.val(data=filename, batch=1, rect=False)