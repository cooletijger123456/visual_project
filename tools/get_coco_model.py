from ultralytics import YOLOE

model_name = "11l"
model = YOLOE(f"runs/pretrain/yolo{model_name}-vl-seg-coco/weights/best.pt")
model.eval()

model.model.model[-1].is_fused = True
del model.model.model[-1].vpe

model.save(f"pretrain/yoloe-{model_name}-seg-coco.pt")
