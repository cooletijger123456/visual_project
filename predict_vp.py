from ultralytics import YOLOE
import numpy as np
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPPredictor

model = YOLOE("yoloe-v8l-vp.pt")

visuals = dict(
    bboxes=np.array(
        [
            [221.52, 405.8, 344.98, 857.54]
        ]
    )
)

model.predict('ultralytics/assets/bus.jpg', save=True, prompts=visuals, predictor=YOLOEVPPredictor)