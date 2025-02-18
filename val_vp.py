from ultralytics import YOLOWorld

model = YOLOWorld("runs/detect/train7/weights/best.pt")

filename = "ultralytics/cfg/datasets/lvis.yaml"

model.val(data=filename, batch=1, split='minival', rect=False, load_vp=True)

# Fixed AP
# model.val(data=data['yaml_file'], batch=1, split='minival', rect=False, imgsz=800, max_det=1000)
# python tools/eval_fixed_ap.py ../datasets/lvis/annotations/lvis_v1_minival.json runs/detect/val2/predictions.json 
