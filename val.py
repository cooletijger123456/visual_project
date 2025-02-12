from ultralytics import YOLOWorld

model = YOLOWorld("yolov8l-worldv2-vlhead-mobileclip-ladapterglu-imgsz800-alpha1.pt")

filename = "ultralytics/cfg/datasets/lvis.yaml"

model.val(data=filename, batch=1, split='minival', rect=False, imgsz=800)
