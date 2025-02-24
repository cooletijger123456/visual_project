from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.val_pe_free import WorldPEFreeDetectValidator

unfused_model = YOLOWorld("yolov8l-worldv2-vl.yaml")
unfused_model.load("runs/final/yolov8l-vl-seg/weights/best.pt")
unfused_model.eval()
unfused_model.cuda()

with open('tools/ram_tag_list.txt', 'r') as f:
    names = [x.strip() for x in f.readlines()]
vocab = unfused_model.get_vocab(names)

model = YOLOWorld("runs/final/yolov8l-vl-seg-pf/weights/best.pt").cuda()
model.set_vocab(vocab, names=names)
model.model.model[-1].is_fused = True
model.model.model[-1].conf = 0.001
model.model.model[-1].max_det = 1000

filename = "ultralytics/cfg/datasets/lvis.yaml"

model.val(data=filename, batch=1, split='minival', rect=False, max_det=1000, single_cls=False, validator=WorldPEFreeDetectValidator)

# python tools/eval_open_ended.py --json ../datasets/lvis/annotations/lvis_v1_minival.json --pred runs/detect/val5/predictions.json --fixed
