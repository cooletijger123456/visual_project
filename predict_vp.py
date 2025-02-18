from ultralytics import YOLOWorld
import numpy as np
from ultralytics.models.yolo.world.predict_vp import WorldVPPredictor

model = YOLOWorld("yolov8l-worldv2-vp.pt")

visuals = dict(
    bboxes=np.array(
        [
            [221.52, 405.8, 344.98, 857.54]
        ]
    )
)

model.predict('ultralytics/assets/bus.jpg', save=True, prompts=visuals, predictor=WorldVPPredictor)