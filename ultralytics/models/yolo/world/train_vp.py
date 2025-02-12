# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
from ultralytics.models import yolo
from copy import copy

class WorldVPTrainer(WorldTrainerFromScratch):
    
    def preprocess_batch(self, batch):
        """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""
        batch = super().preprocess_batch(batch)
        batch["visuals"] = batch["visuals"].to(self.device)
        return batch
