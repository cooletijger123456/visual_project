from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
import os

os.environ["PYTHONHASHSEED"] = "0"

data = dict(
    train=dict(
        yolo_data=["Objects365v1.yaml"],
        grounding_data=[
            dict(
                img_path="../datasets/flickr/full_images/",
                json_file="../datasets/flickr/annotations/final_flickr_separateGT_train.json",
            ),
            dict(
                img_path="../datasets/mixed_grounding/gqa/images",
                json_file="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco.json",
            ),
        ],
    ),
    val=dict(yolo_data=["lvis.yaml"]),
)
model = YOLOWorld("yolov8s-worldv2.yaml")

model.train(data=data, batch=128, epochs=100, close_mosaic=2, \
    optimizer='AdamW', lr0=2e-3, warmup_bias_lr=0.0, \
        weight_decay=0.025, momentum=0.9, \
        trainer=WorldTrainerFromScratch, device='0,1,2,3,4,5,6,7')