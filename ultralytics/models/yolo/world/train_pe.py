# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch, WorldTrainer
from ultralytics.models.yolo.detect import DetectionTrainer
from copy import deepcopy
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from copy import copy
from ultralytics.nn.tasks import WorldModel, WorldSegModel
from ultralytics.utils import DEFAULT_CFG, RANK

class WorldPETrainer(DetectionTrainer):
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return WorldModel initialized with specified config and weights."""
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = WorldModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=3,
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        
        model.eval()
        pe_state = torch.load(self.args.train_pe_path)
        model.set_classes(pe_state["names"], pe_state["pe"])
        model.model[-1].fuse(model.pe)
        model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2])
        model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2])
        model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2])
        del model.pe
        model.train()
        
        return model

class WorldPEFreeTrainer(WorldPETrainer, WorldTrainerFromScratch):
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box", "cls", "dfl"
        return DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
    def preprocess_batch(self, batch):
        """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""
        batch = super(WorldTrainer, self).preprocess_batch(batch)
        return batch