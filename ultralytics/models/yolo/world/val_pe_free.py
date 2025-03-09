from ultralytics.models.yolo.detect import DetectionValidator

class WorldPEFreeValidatorMixin:
    def eval_json(self, stats):
        return stats

class WorldPEFreeDetectValidator(WorldPEFreeValidatorMixin, DetectionValidator):
    pass