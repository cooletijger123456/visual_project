from ultralytics import YOLOE

model = YOLOE("runs/final/yolov8l-vl-seg/weights/best.pt")

names = ["person"]
model.set_classes(names, model.get_text_pe(names))

model.predict('ultralytics/assets/bus.jpg', save=True)
