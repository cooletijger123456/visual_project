from ultralytics import YOLOWorld

model = YOLOWorld("yolov8m-worldv2.pt")

model.set_classes(["person"])

model.predict('ultralytics/assets/zidane.jpg', save=True)
