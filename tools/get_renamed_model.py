import os
from ultralytics import YOLOE

for pt in os.listdir('pretrain_backup'):
    file = f'pretrain_backup/{pt}'
    model = YOLOE(file)
    model.model.model[-1].reprta = model.model.model[-1].gc
    if hasattr(model.model.model[-1], "vpe"):
        model.model.model[-1].savpe = model.model.model[-1].vpe
        del model.model.model[-1].vpe
    
    del model.model.model[-1].gc
    model.save(f'pretrain/{pt}')