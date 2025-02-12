# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.world.train_world_seg import WorldSegTrainerFromScratch
from .val import WorldSegValidator
from copy import copy
from .train_vp import WorldVPTrainer

class WorldSegVPTrainer(WorldVPTrainer, WorldSegTrainerFromScratch):
    pass
