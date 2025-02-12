# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
from ultralytics.models import yolo
from .val_vp import WorldVPDetectValidator
from copy import copy

class WorldVPTrainer(WorldTrainerFromScratch):
    
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box", "cls", "dfl"
        return WorldVPDetectValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
    def preprocess_batch(self, batch):
        """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""
        batch = super().preprocess_batch(batch)
        batch["visuals"] = batch["visuals"].to(self.device)
        return batch
