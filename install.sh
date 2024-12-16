set -e
set -x

conda create -n yoloworld python=3.10

pip install -r requirements.txt
pip install -e .
pip install -e CLIP
pip install -e lvis-api

# Generate data
# python tools/generate_objects365v1.py

# Verify data
# python tools/verify_objects365.py
# python tools/verify_lvis.py