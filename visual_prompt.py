import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from ultralytics.utils.torch_utils import smart_inference_mode
from gradio_image_prompter import ImagePrompter
import torch, supervision as sv, os
from pathlib import Path

# ───────────────────────── GLOBALS ─────────────────────────
selected_weights = "pretrain/yoloe-v8l-seg.pt"
conf_threshold   = 0.25    # default confidence
iou_threshold    = 0.70    # default IoU 

# ─────────────────── MODEL HELPERS ─────────────────────────
def load_model(weights_path=selected_weights):
    m = YOLOE(weights_path)
    m.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return m

model = load_model(selected_weights)

def reset_model_predictor(tag="", weights_path=None):
    """Reload model and clear predictor/VPE buffers."""
    global model
    print(f"[{tag}] reset — weights = {weights_path or selected_weights}")
    model = load_model(weights_path or selected_weights)
    try:
        if getattr(model, "predictor", None) and hasattr(model.predictor, "vpe"):
            del model.predictor.vpe
        model.predictor = None
        if hasattr(model, "_classes"): del model._classes
        if hasattr(model.model, "model") and hasattr(model.model.model, "vpe"):
            del model.model.model.vpe
    except Exception:  # pragma: no cover
        import traceback; traceback.print_exc()

# ───────────────── SINGLE-IMAGE INFERENCE ─────────────────
@smart_inference_mode()
def run_prediction(*inputs):
    reset_model_predictor("single_predict", selected_weights)

    results, bboxes_list, cls_list = [], [], []
    source_names, source_imgs = [], []

    for i, inp in enumerate(inputs):
        if not inp or "image" not in inp or "points" not in inp:
            results.append(Image.new("RGB", (512, 512), "gray")); continue

        img, pts = inp["image"], inp["points"]
        if img is None or pts is None:
            results.append(Image.new("RGB", img.size, "gray")); continue

        source_names.append(getattr(img, "filename", f"in_{i+1}.jpg"))
        source_imgs.append(img.copy())

        boxes = np.array([p[[0, 1, 3, 4]] for p in np.array(pts) if p[2] == 2])
        if len(boxes) == 0:
            results.append(Image.new("RGB", img.size, "gray"))
            bboxes_list.append(None); cls_list.append(None); continue

        bboxes_list.append(boxes)
        cls_list.append(np.zeros(len(boxes), int))
        prompts = {"bboxes": boxes, "cls": np.zeros(len(boxes), int)}

        preds = model.predict(
            img,
            prompts=prompts,
            predictor=YOLOEVPSegPredictor,
            conf=conf_threshold,
            iou=iou_threshold                           # ← NEW
        )

        det = sv.Detections.from_ultralytics(preds[0])
        annotated = img.copy()
        if len(det):
            wh = img.size
            labels = [f"{n} {c:.2f}" for n, c in zip(det['class_name'], det.confidence)]
            annotated = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4)\
                         .annotate(annotated, det)
            annotated = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX,
                                        thickness=sv.calculate_optimal_line_thickness(wh))\
                         .annotate(annotated, det)
            annotated = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX,
                                          text_scale=sv.calculate_optimal_text_scale(wh))\
                         .annotate(annotated, det, labels)
        else:
            ImageDraw.Draw(annotated).text((10, 10), "No detections", fill="red")
        results.append(annotated)

    # build visuals_code for VPE
    vcode  = "import numpy as np\n\nvisuals = dict(\n    bboxes=[\n"
    for b in bboxes_list:
        vcode += f"        np.array({repr(b.tolist())}),\n" if b is not None else "        None,\n"
    vcode += "    ],\n    cls=[\n"
    for c in cls_list:
        vcode += f"        np.array({repr(c.tolist())}),\n" if c is not None else "        None,\n"
    vcode += "    ]\n)\n\n"
    for idx, n in enumerate(source_names):
        vcode += f"source_image{idx+1} = '{os.path.basename(n)}'\n"

    return results + [vcode, source_imgs]

# ─────────────── ACCUMULATED-VPE INFERENCE ───────────────
@smart_inference_mode()
def predict_with_accumulated_vpe(target_img, visuals_code, source_imgs):
    reset_model_predictor("accum_vpe", selected_weights)
    try:
        ns = {}; exec(visuals_code, {}, ns)
        visuals = ns.get("visuals", {})
        refs = [im for im in source_imgs if im is not None]
        if not refs: raise ValueError("No reference images")

        # build VPE
        model.predict(
            refs,
            prompts=visuals,
            predictor=YOLOEVPSegPredictor,
            return_vpe=True,
            conf=0.10,
            iou=iou_threshold                          # ← NEW
        )
        mean_vpe = torch.nn.functional.normalize(model.predictor.vpe.mean(0, keepdim=True),
                                                 p=2, dim=-1)
        model.set_classes([f"object{i}" for i in range(mean_vpe.shape[0])], mean_vpe)
        model.predictor = None

        res = model.predict(target_img, conf=conf_threshold, iou=iou_threshold)[0]  # ← NEW
        det = sv.Detections.from_ultralytics(res)
        out = target_img.copy()
        if len(det):
            wh = target_img.size
            labels = [f"{n} {c:.2f}" for n, c in zip(det['class_name'], det.confidence)]
            out = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4)\
                  .annotate(out, det)
            out = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX,
                                  thickness=sv.calculate_optimal_line_thickness(wh))\
                  .annotate(out, det)
            out = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX,
                                    text_scale=sv.calculate_optimal_text_scale(wh))\
                  .annotate(out, det, labels)
        else:
            ImageDraw.Draw(out).text((10, 10), "No detections", fill="red")
        return out
    finally:
        reset_model_predictor("accum_vpe:end", selected_weights)

# ─────────────────────── BATCH PREDICT ────────────────────
@smart_inference_mode()
def batch_predict_from_folder(folder_path, output_folder,
                              visuals_code, source_imgs):
    reset_model_predictor("batch", selected_weights)
    try:
        ns = {}; exec(visuals_code, {}, ns)
        visuals = ns.get("visuals", {})
        refs = [im for im in source_imgs if im is not None]
        if not refs: return "No refs."

        model.predict(refs, prompts=visuals,
                      predictor=YOLOEVPSegPredictor,
                      return_vpe=True, conf=0.10, iou=iou_threshold)   # ← NEW
        mean_vpe = torch.nn.functional.normalize(model.predictor.vpe.mean(0, keepdim=True),
                                                 p=2, dim=-1)
        model.set_classes([f"object{i}" for i in range(mean_vpe.shape[0])], mean_vpe)
        model.predictor = None

        paths = [str(Path(folder_path)/f) for f in sorted(os.listdir(folder_path))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not paths: return "No images."
        os.makedirs(output_folder, exist_ok=True)

        for p in paths:
            model.predict(p, conf=conf_threshold, iou=iou_threshold,   # ← NEW
                          save=True, project=output_folder,
                          name="", exist_ok=True, stream=False)
            print("Saved:", Path(p).name)
        return f"Processed {len(paths)} images."
    finally:
        reset_model_predictor("batch:end", selected_weights)

# ───────────────────────────── UI ──────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("## YOLOE Visual Prompt – Side-by-Side Viewer")

    # weight, confidence, IoU controls
    with gr.Row():
        weight_opts = sorted([f"pretrain/{f}" for f in os.listdir("pretrain")
                              if f.endswith(".pt")])
        weights_dd  = gr.Dropdown(weight_opts, value=selected_weights,
                                  label="YOLOE Weights")
        conf_slider = gr.Slider(0.05, 0.90, step=0.05, value=conf_threshold,
                                label="Confidence Threshold")
        iou_slider  = gr.Slider(0.10, 0.95, step=0.05, value=iou_threshold,     # ← NEW
                                label="IoU Threshold")

    # callbacks
    weights_dd.change(lambda w: (reset_model_predictor("weights_changed", w), None),
                      weights_dd, [])
    conf_slider.change(lambda c: (globals().__setitem__('conf_threshold', c),
                                  reset_model_predictor("conf_changed")), conf_slider, [])
    iou_slider.change(lambda t: (globals().__setitem__('iou_threshold', t),            # ← NEW
                                 reset_model_predictor("iou_changed")), iou_slider, [])

    # reference-prompt inputs
    MAX_INPUTS, idx_state, src_imgs_state = 5, gr.State(1), gr.State()
    prompters, preds, rows = [], [], []
    for i in range(MAX_INPUTS):
        with gr.Row(visible=i == 0) as r:
            with gr.Column():
                prompters.append(
                    ImagePrompter(label=f"Input {i+1}", type="pil", interactive=True))
            with gr.Column():
                preds.append(gr.Image(label=f"Prediction {i+1}", type="pil"))
            rows.append(r)

    # add/remove rows
    toggle = lambda idx, add=True: [gr.update(visible=i < (min(MAX_INPUTS, idx+1)
                                    if add else max(1, idx-1)))
                                    for i in range(MAX_INPUTS)] + \
                                    [min(MAX_INPUTS, idx+1) if add else max(1, idx-1)]
    gr.Button("+ Add Image").click(toggle, idx_state, rows + [idx_state])
    gr.Button("- Remove Image").click(lambda i: toggle(i, False),
                                      idx_state, rows + [idx_state])

    vis_code = gr.Code(label="Collected Visual Prompt Info", language="python")
    gr.Button("Run Inference").click(run_prediction, prompters,
                                     preds + [vis_code, src_imgs_state])

    # Accumulated-prompt UI
    gr.Markdown("### Predict on a New Image Using Accumulated Prompts")
    with gr.Row(equal_height=True):
        tgt_img = gr.Image(label="Target Image", type="pil")
        tgt_out = gr.Image(label="→ Prediction", type="pil")
    gr.Button("Run Inference").click(
        predict_with_accumulated_vpe,
        [tgt_img, vis_code, src_imgs_state],
        tgt_out
    )

    # batch UI
    gr.Markdown("### Batch Predict on Folder")
    inp_folder = gr.Textbox(label="Folder Path")
    out_folder = gr.Textbox(label="Output Folder")
    batch_txt  = gr.Textbox(label="Batch Result")
    gr.Button("Run Batch").click(
        batch_predict_from_folder,
        [inp_folder, out_folder, vis_code, src_imgs_state],
        batch_txt
    )

page = demo
if __name__ == "__main__":
    page.launch()
