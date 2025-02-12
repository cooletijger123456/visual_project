set -e
set -x

export PYTHONPATH=`pwd`:$PYTHONPATH

python yololm/eval/lvis_eval.py --model exp12/tiny-llm --json lvis_val_1k_category.json
python yololm/eval/lvis_eval.py  --model exp12/tiny-llm --json paco_val_1k_category.json
