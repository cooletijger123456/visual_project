import os
import random
import string

from tqdm import tqdm

from ultralytics import YOLOWorld
from pathlib import Path


def generate_random_phrases(n, min_words=1, max_words=2, min_length=3, max_length=5):
    phrases = []
    for _ in range(n):
        word_count = random.randint(min_words, max_words)  # Randomize word count
        phrase = " ".join(
            ''.join(random.choices(string.ascii_letters, k=random.randint(min_length, max_length)))
            for _ in range(word_count)
        )
        phrases.append(phrase)
    return phrases

for label_num in tqdm([100, 500, 1000]):
    output_dir = f"exports"
    os.makedirs(output_dir, exist_ok=True)
    for model_name in ["yolov8m-worldv2-pan.yaml"]:
        model = YOLOWorld(model_name)

        names = generate_random_phrases(label_num)
        model.set_classes(names)

        onnx_path = model.export(format='onnx', half=True, opset=13, simplify=True, device="0")
        coreml_path = model.export(format='coreml', half=True, nms=False, device="0")
        
        save_name = f"{Path(model_name).stem}-cls{label_num}"
        os.rename(onnx_path, os.path.join(output_dir, f'{save_name}.onnx'))
        os.rename(coreml_path, os.path.join(output_dir, f'{save_name}.mlpackage'))

