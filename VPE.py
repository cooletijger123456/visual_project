import os
import json
import torch
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
import torch.nn.functional as F
from torchvision.ops import box_iou, nms
from ultralytics.trackers import register_tracker

# =========== CONFIG ===========
frames_dir       = "/test_docker/yoloe-main/yoloe-main/data/highway/frames_blur_dark_ramp"
gt_json_path     = "/test_docker/yoloe-main/yoloe-main/data/highway/annotations/ground_truths.json"
model_path       = "yoloe-v8l-seg.pt"
MAX_PE           = 100
class_name       = ["car"]
TRACKING         = True
device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========== SAMPLING ===========
sample_frames = []
sample_ranges = [(0, 30000)]

# =========== UTILS ===========
def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec

def match_predictions(pred_cls, true_cls, iou_matrix, iouv):
    correct = np.zeros((pred_cls.shape[0], iouv.numel()), dtype=bool)
    if true_cls.numel() == 0 or pred_cls.numel() == 0:
        return torch.from_numpy(correct)
    cls_mask = true_cls[:, None] == pred_cls
    ious = iou_matrix.cpu().numpy() * cls_mask.cpu().numpy().T
    for ti, thr in enumerate(iouv.cpu().tolist()):
        matches = (np.stack(np.nonzero(ious >= thr), axis=1)
                   if ious.size else np.zeros((0,2), int))
        if matches.size:
            iou_vals = ious[matches[:,0], matches[:,1]]
            order = iou_vals.argsort()[::-1]
            matches = matches[order]
            _, gt_idx = np.unique(matches[:,1], return_index=True)
            matches = matches[gt_idx]
            _, pred_idx = np.unique(matches[:,0], return_index=True)
            matches = matches[pred_idx]
            correct[matches[:,0], ti] = True
    return torch.from_numpy(correct)

# =========== LOAD DATA ===========
with open(gt_json_path, "r") as f:
    gt_data = {e["image_id"]: e for e in json.load(f)}

image_files = sorted([
    fn for fn in os.listdir(frames_dir)
    if fn.lower().endswith(('.png','jpg','jpeg')) and fn in gt_data
])

# =========== INIT TEXT PE & BANK ===========
bootstrap = YOLOE(model_path).to(device)
tpe = bootstrap.get_text_pe(class_name).to(device)
pe_bank = torch.zeros((1, MAX_PE, 512), device=device)
pe_bank[:,0,:] = tpe[:,0,:]

fifo_pos = 0
buffer_size = MAX_PE - 1
assert buffer_size >= 0, "MAX_PE must be >= 1"
inserted_vpes = 0

# =========== METRICS ===========
stats_correct = []
stats_confs = []
stats_predcls = []
num_targets = 0

# =========== BASE MODELS ===========
base_model = YOLOE(model_path).to(device)
if TRACKING:
    register_tracker(base_model, persist=True)
base_model.predictor = None
base_model.set_classes(class_name * MAX_PE, pe_bank)

vpe_model = YOLOE(model_path).to(device)
vpe_model.predictor = None

# =========== RUN INFERENCE ===========
print(f"Sampling frames: {sample_frames}")

for idx, fname in enumerate(image_files, start=0):
    img_path = os.path.join(frames_dir, fname)
    print(f"\n→ Frame {idx}/{len(image_files)}: {fname}")

    base_model.predictor = None
    base_model.set_classes(class_name * MAX_PE, pe_bank)

    result = base_model.predict(img_path, conf=0.001, fuse=False, verbose=False)[0]

    boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.empty((0,4))
    confs = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.empty((0,))
    pred_cls = np.zeros_like(result.boxes.cls.cpu().numpy(), dtype=int) if len(result.boxes) else np.empty((0,), int)

    if boxes.shape[0]:
        keep = nms(torch.tensor(boxes), torch.tensor(confs), 0.5).cpu().numpy()
        boxes, confs, pred_cls = boxes[keep], confs[keep], pred_cls[keep]

    gt_boxes = np.array(gt_data[fname]["boxes"], dtype=np.float32).reshape(-1,4)
    num_targets += gt_boxes.shape[0]

    iou_m = box_iou(torch.tensor(boxes), torch.tensor(gt_boxes)) if boxes.size and gt_boxes.size else torch.zeros((boxes.shape[0], gt_boxes.shape[0]))
    correct = match_predictions(torch.from_numpy(pred_cls), torch.zeros(gt_boxes.shape[0], dtype=int), iou_m, torch.linspace(0.5, 0.95, 10))
    if boxes.size:
        stats_correct.append(correct)
        stats_confs.append(torch.from_numpy(confs))
        stats_predcls.append(torch.from_numpy(pred_cls))

    in_list = idx in sample_frames
    in_range = any(start <= idx <= end for start, end in sample_ranges)
    do_sampling = in_list or in_range

    if buffer_size > 0 and do_sampling:
        print(f"   → Sampling for VPE on frame {idx}")
        valid_conf_mask = confs > 0.1
        if np.any(valid_conf_mask):
            print("      predicted confidences (conf > 0.1):", ", ".join(f"{c:.4f}" for c in confs[valid_conf_mask]))
        else:
            print("      no predicted boxes with conf > 0.1")

        prompt_mask = (confs > 0.73) & (confs < 1)  # or any max threshold you want

        prompt_boxes = boxes[prompt_mask]

        if len(prompt_boxes) == 0:
            print("      No predicted boxes with to use for VPE.")
        else:
            for box in prompt_boxes:
                vpe_model.predictor = None
                vp = {"bboxes": [box.reshape(1, 4)], "cls": [np.array([0])]}
                vpe_model.predict(
                    img_path,
                    prompts=vp,
                    predictor=YOLOEVPSegPredictor,
                    fuse=False,
                    return_vpe=True,
                    verbose=False,
                    save=False
                )
                new_vpe = vpe_model.predictor.vpe  # shape: [1, 1, 512]
                new_vpe = F.normalize(new_vpe, dim=2)

                # Mask zero vectors in bank
                nonzero_mask = pe_bank.abs().sum(dim=2) != 0  # [1, MAX_PE]
                existing_vpes = pe_bank[0, nonzero_mask[0]]  # shape: [N, 512]

                # Cosine similarity check
                similarities = F.cosine_similarity(new_vpe[0, 0].unsqueeze(0), existing_vpes, dim=1)
                max_similarity = similarities.max().item()
                similarity_threshold = 0.7

                if max_similarity >= similarity_threshold:
                    print(f"      → Skipping VPE: max cosine similarity {max_similarity:.4f} (threshold = {similarity_threshold})")
                else:
                    slot = 1 + (inserted_vpes % buffer_size)
                    pe_bank[:, slot, :] = new_vpe[:, 0, :]
                    inserted_vpes += 1
                    print(f"      → Inserted VPE at slot {slot}, similarity = {max_similarity:.4f}")


            num_vpes = min(inserted_vpes, buffer_size)
            print(f"   → Bank now holds 1 TEXT + {num_vpes} VPE(s)")
    else:
        print(f"   → Skipping VPE insertion for frame {idx}")

# =========== FINAL METRICS ===========
if stats_correct:
    corr_cat = torch.cat(stats_correct, dim=0)
    conf_cat = torch.cat(stats_confs, dim=0).numpy()
    predcls_cat = torch.cat(stats_predcls, dim=0).numpy()
else:
    corr_cat = torch.zeros((0, len(torch.linspace(0.5, 0.95, 10))), dtype=bool)
    conf_cat = np.array([])
    predcls_cat = np.array([])

order = np.argsort(-conf_cat)
corr_cat = corr_cat[order]
conf_cat = conf_cat[order]

aps = []
for ti in range(len(torch.linspace(0.5, 0.95, 10))):
    c = corr_cat[:, ti].numpy() if corr_cat.shape[0] else np.array([])
    if c.size:
        fp = (1 - c).cumsum()
        tp = c.cumsum()
        rc = tp / (num_targets + 1e-16)
        pr = tp / (tp + fp + 1e-16)
        ap, _, _ = compute_ap(rc, pr)
    else:
        ap = 0.0
    aps.append(ap)

mAP50 = aps[0]
mAP50_95 = float(np.mean(aps))
nonzero_mask = pe_bank.abs().sum(dim=2) != 0
count_nonzero = int(nonzero_mask.sum().item())

print("\n====== FINAL RESULTS ======")
print(f"pe_bank size = {count_nonzero} tracking = {TRACKING}")
print(f"dataset: {frames_dir}")
print(f"frames:        {len(image_files)}")
print(f"detections:    {corr_cat.shape[0]}")
print("IoU thresholds:", ", ".join(f"{t:.2f}" for t in torch.linspace(0.5, 0.95, 10).tolist()))
print("AP per IoU:   ", ", ".join(f"{x:.4f}" for x in aps))
print(f"mAP@0.50:     {mAP50:.4f}")
print(f"mAP@0.50:0.95: {mAP50_95:.4f}")

import os
import re
import json
from PIL import Image, ImageDraw, ImageFont

# --- Paths ---
frames_dir = "/test_docker/yoloe-main/yoloe-main/data/dog17/frames_clean"
groundtruth_path = "/test_docker/yoloe-main/yoloe-main/data/dog17/groundtruth.txt"
annotations_dir = "/test_docker/yoloe-main/yoloe-main/data/dog17/annotations"
visuals_dir = os.path.join(annotations_dir, "visuals")

os.makedirs(annotations_dir, exist_ok=True)
os.makedirs(visuals_dir, exist_ok=True)

# --- Settings ---
class_id = 0  # single class
class_name = "dog"

# --- Load ground truth boxes ---
with open(groundtruth_path, "r") as f:
    gt_lines = [line.strip() for line in f if line.strip()]

# Parse into list of [x1, y1, x2, y2]
gt_boxes_per_frame = []
for line in gt_lines:
    parts = line.split(",")
    if len(parts) != 4:
        continue
    x, y, w, h = map(float, parts)
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    gt_boxes_per_frame.append([x1, y1, x2, y2])

# --- Helper for numeric sort ---
def extract_index(fname):
    m = re.search(r'(\d+)', fname)
    return int(m.group(1)) if m else -1

# --- Collect annotations ---
annotations = []

# Gather and sort frame filenames (jpg/jpeg/png) numerically
frame_files = [
    f for f in os.listdir(frames_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]
frame_files = sorted(frame_files, key=extract_index)

if len(frame_files) != len(gt_boxes_per_frame):
    print(f"Warning: number of frames ({len(frame_files)}) != number of GT lines ({len(gt_boxes_per_frame)}). "
          f"Using min of both.")
count = min(len(frame_files), len(gt_boxes_per_frame))

for idx in range(count):
    img_name = frame_files[idx]
    img_path = os.path.join(frames_dir, img_name)
    img = Image.open(img_path).convert("RGB")

    box = gt_boxes_per_frame[idx]
    x1, y1, x2, y2 = box
    # Clamp to image bounds
    w_img, h_img = img.size
    x1 = max(0, min(x1, w_img))
    y1 = max(0, min(y1, h_img))
    x2 = max(0, min(x2, w_img))
    y2 = max(0, min(y2, h_img))
    box_clamped = [x1, y1, x2, y2]

    # Save annotation entry
    annotations.append({
        "image_id": img_name,
        "boxes": [box_clamped],
        "classes": [class_id]
    })

    # Visualization
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text_pos = (x1, max(y1 - 10, 0))
    draw.text(text_pos, class_name, fill="green", font=font)
    img.save(os.path.join(visuals_dir, img_name))

# --- Save all annotations to JSON ---
output_json = os.path.join(annotations_dir, "ground_truths.json")
with open(output_json, "w") as f:
    json.dump(annotations, f, indent=2)

print(f"\n✅ Done! Saved {len(annotations)} ground truth entries to:")
print(f"   → {output_json}")
print(f" Visualizations saved to:")
print(f"   → {visuals_dir}/")



import os
import json
from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLOE
from ultralytics.trackers import register_tracker

# --- Paths ---
frames_dir = "/test_docker/yoloe-main/yoloe-main/data/person4/frames_clean"
annotations_dir = "/test_docker/yoloe-main/yoloe-main/data/person4/annotations"
visuals_dir = os.path.join(annotations_dir, "visuals")

os.makedirs(annotations_dir, exist_ok=True)
os.makedirs(visuals_dir, exist_ok=True)

# --- Load model ---
model = YOLOE('yoloe-v8l-seg.pt')
register_tracker(model, persist=True) 
model.to('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['person']
model.set_classes(class_names, model.get_text_pe(class_names))
car_class_id = 0  # 'car' is index 0

# --- Collect pseudo ground truth annotations ---
annotations = []

for img_name in sorted(os.listdir(frames_dir)):
    if not img_name.lower().endswith('.png'):
        continue

    img_path = os.path.join(frames_dir, img_name)
    img = Image.open(img_path).convert("RGB")

    # --- Predict ---
    result = model.predict(img, conf=0.3, iou=0.9, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    # --- Filter to 'car' class ---
    car_boxes = boxes[classes == car_class_id]
    car_scores = scores[classes == car_class_id]

    if len(car_boxes) == 0:
        continue  # skip images with no car prediction

    # --- Apply NMS to remove overlapping boxes ---
    car_boxes_tensor = torch.tensor(car_boxes, dtype=torch.float32)
    car_scores_tensor = torch.tensor(car_scores, dtype=torch.float32)
    keep = nms(car_boxes_tensor, car_scores_tensor, iou_threshold=0.5)
    car_boxes_nms = car_boxes_tensor[keep].numpy()

    # --- Save annotation ---
    annotations.append({
        "image_id": img_name,
        "boxes": car_boxes_nms.tolist(),
        "classes": [car_class_id] * len(car_boxes_nms)
    })

    # --- Save visualization ---
    draw = ImageDraw.Draw(img)
    for box in car_boxes_nms:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1 - 10), "person", fill="green")
    img.save(os.path.join(visuals_dir, img_name))

# --- Save all annotations to JSON ---
output_json = os.path.join(annotations_dir, "ground_truths.json")
with open(output_json, "w") as f:
    json.dump(annotations, f, indent=2)

print(f"\n✅ Done! Saved {len(annotations)} pseudo-ground truth entries to:")
print(f"   → {output_json}")
print(f" Visualizations saved to:")
print(f"   → {visuals_dir}/")
