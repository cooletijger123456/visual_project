# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.world.train_world_seg import WorldSegTrainerFromScratch
from .val_vp import WorldVPSegValidator
from copy import copy
from .train_vp import WorldVPTrainer

class WorldSegVPTrainer(WorldVPTrainer, WorldSegTrainerFromScratch):
    
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box", "seg", "cls", "dfl"
        return WorldVPSegValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
