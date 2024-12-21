set -e
set -x

conda create -n yoloworld python=3.10 -y
conda activate yoloworld

pip install -r requirements.txt
pip install -e .
pip install -e CLIP
pip install -e lvis-api

# Generate data
# python tools/generate_objects365v1.py

# Generate grounding cache
# python tools/generate_grounding_cache.py --img-path ../datasets/flickr/full_images/ --json-path ../datasets/flickr/annotations/final_flickr_separateGT_train.json
# python tools/generate_grounding_cache.py --img-path ../datasets/mixed_grounding/gqa/images --json-path ../datasets/mixed_grounding/annotations/final_mixed_train_no_coco.json

# Verify data
# python tools/verify_objects365.py
# python tools/verify_lvis.py

# Generate train label embeddings
# python tools/generate_label_embedding.py
# python tools/generate_global_neg_cat.py
