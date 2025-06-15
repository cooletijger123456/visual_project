import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from ultralytics.utils.torch_utils import smart_inference_mode
from gradio_image_prompter import ImagePrompter
import torch
import supervision as sv
import os

# Load model
model = YOLOE("pretrain/yoloe-v8l-seg.pt")
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load model wrapper for clean reinitialization
def load_model():
    model = YOLOE("pretrain/yoloe-v8l-seg.pt")
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

model = load_model()

def reset_model_predictor(debug_tag=""):
    global model
    print(f"[{debug_tag}] Reinitializing model...")
    model = load_model()
    try:
        if hasattr(model, "predictor") and model.predictor is not None:
            if hasattr(model.predictor, "vpe"):
                print(f"[{debug_tag}] Deleting predictor.vpe: shape={getattr(model.predictor.vpe, 'shape', 'N/A')}")
                del model.predictor.vpe
            else:
                print(f"[{debug_tag}] predictor.vpe not found.")
        else:
            print(f"[{debug_tag}] model.predictor is None or not set.")

        model.predictor = None
        print(f"[{debug_tag}] model.predictor set to None.")

        if hasattr(model, "_classes"):
            print(f"[{debug_tag}] Deleting model._classes")
            del model._classes
        else:
            print(f"[{debug_tag}] model._classes not set, skipping.")

        # Try removing low-level VPE embedding if it exists
        if hasattr(model.model, 'model') and hasattr(model.model.model, 'vpe'):
            print(f"[{debug_tag}] Deleting model.model.model.vpe")
            del model.model.model.vpe
        else:
            print(f"[{debug_tag}] model.model.model.vpe not set or unavailable.")

    except Exception as e:
        print(f"[{debug_tag}] Error during reset_model_predictor: {e}")
        import traceback
        traceback.print_exc()

# rest of the code remains unchanged

@smart_inference_mode()
def run_prediction(*inputs):
    reset_model_predictor("run_prediction:init")

    results = []
    bboxes_list = []
    cls_list = []
    source_images = []
    source_images_data = []

    for i, inp in enumerate(inputs):
        try:
            if not inp or "image" not in inp or "points" not in inp:
                results.append(Image.new("RGB", (512, 512), "gray"))
                continue

            img = inp["image"]
            pts = inp["points"]
            if not img or not pts:
                results.append(Image.new("RGB", img.size, "gray"))
                continue

            filename = getattr(img, "filename", f"input_image_{i+1}.jpg")
            source_images.append(filename)
            source_images_data.append(img.copy())

            boxes = np.array([p[[0, 1, 3, 4]] for p in np.array(pts) if p[2] == 2])
            if len(boxes) == 0:
                results.append(Image.new("RGB", img.size, "gray"))
                bboxes_list.append(None)
                cls_list.append(None)
                continue

            bboxes_list.append(boxes)
            cls_list.append(np.zeros(len(boxes), dtype=int))

            prompts = {"bboxes": boxes, "cls": np.zeros(len(boxes), dtype=int)}
            preds = model.predict(source=img, prompts=prompts, predictor=YOLOEVPSegPredictor, conf=0.25)

            detections = sv.Detections.from_ultralytics(preds[0])
            annotated = img.copy()

            if len(detections) == 0:
                draw = ImageDraw.Draw(annotated)
                draw.text((10, 10), "No detections", fill="red")
                results.append(annotated)
                continue

            resolution_wh = img.size
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

            labels = [
                f"{class_name} {conf:.2f}"
                for class_name, conf in zip(detections['class_name'], detections.confidence)
            ]

            annotated = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4).annotate(annotated, detections)
            annotated = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness).annotate(annotated, detections)
            annotated = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale).annotate(annotated, detections, labels)

            results.append(annotated)

        except Exception as e:
            print(f"[ERROR in input {i}] {e}")
            import traceback; traceback.print_exc()
            fallback = Image.new("RGB", img.size if 'img' in locals() else (512, 512), "red")
            draw = ImageDraw.Draw(fallback)
            draw.text((10, 10), "Error", fill="white")
            results.append(fallback)
            bboxes_list.append(None)
            cls_list.append(None)
            source_images.append(f"input_image_{i+1}.jpg")
            source_images_data.append(None)

    visual_str = "import numpy as np\n\nvisuals = dict(\n    bboxes=[\n"
    for b in bboxes_list:
        visual_str += f"        np.array({repr(b.tolist())}),\n" if b is not None else "        None,\n"
    visual_str += "    ],\n    cls=[\n"
    for c in cls_list:
        visual_str += f"        np.array({repr(c.tolist())}),\n" if c is not None else "        None,\n"
    visual_str += "    ]\n)\n\n"

    for idx, name in enumerate(source_images):
        visual_str += f"source_image{idx + 1} = '{os.path.basename(name)}'\n"

    return results + [visual_str, source_images_data]

@smart_inference_mode()
def predict_with_accumulated_vpe(target_image, visuals_code, source_images_data):
    reset_model_predictor("vpe:before_exec")

    try:
        local_vars = {}
        exec(visuals_code, {}, local_vars)

        visuals = local_vars.get("visuals", {})
        source_image_keys = sorted([k for k in local_vars if k.startswith("source_image")])

        image_inputs = []
        for i, key in enumerate(source_image_keys):
            if i < len(source_images_data):
                image = source_images_data[i]
                if image is not None:
                    image_inputs.append(image)

        if not image_inputs:
            raise ValueError("No valid source images in memory.")

        _ = model.predict(
            source=image_inputs,
            prompts=visuals,
            predictor=YOLOEVPSegPredictor,
            return_vpe=True,
            conf=0.1
        )

        if hasattr(model.predictor, "vpe") and model.predictor.vpe is not None:
            print(f"[vpe:accum] VPE shape: {model.predictor.vpe.shape}")
            mean_vpe = torch.nn.functional.normalize(model.predictor.vpe.mean(dim=0, keepdim=True), dim=-1, p=2)
            model.set_classes(["object" + str(i) for i in range(mean_vpe.shape[0])], mean_vpe)
        else:
            raise ValueError("VPE not generated correctly from source images.")

        model.predictor = None
        result = model.predict(source=target_image, conf=0.25)[0]
        detections = sv.Detections.from_ultralytics(result)
        annotated = target_image.copy()

        if len(detections) > 0:
            resolution_wh = target_image.size
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
            labels = [
                f"{class_name} {conf:.2f}"
                for class_name, conf in zip(detections['class_name'], detections.confidence)
            ]

            annotated = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4).annotate(annotated, detections)
            annotated = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness).annotate(annotated, detections)
            annotated = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale).annotate(annotated, detections, labels)
        else:
            draw = ImageDraw.Draw(annotated)
            draw.text((10, 10), "No detections", fill="red")

        return annotated

    except Exception as e:
        print(f"[ERROR in VPE prediction] {e}")
        import traceback; traceback.print_exc()
        fallback = Image.new("RGB", target_image.size if target_image else (512, 512), "red")
        draw = ImageDraw.Draw(fallback)
        draw.text((10, 10), "Error", fill="white")
        return fallback

    finally:
        reset_model_predictor("vpe:finally")

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## YOLOE Visual Prompt – Side-by-Side Viewer")

    MAX_INPUTS = 5
    index = gr.State(value=1)
    source_image_data_state = gr.State()

    input_prompters = []
    output_images = []
    visibility_controls = []

    for i in range(MAX_INPUTS):
        with gr.Row(visible=(i == 0)) as row:
            with gr.Column():
                img_input = ImagePrompter(label=f"Input Image {i+1}", type="pil", interactive=True)
            with gr.Column():
                output_img = gr.Image(label=f"Prediction {i+1}", type="pil")
            input_prompters.append(img_input)
            output_images.append(output_img)
            visibility_controls.append(row)

    with gr.Row():
        add_btn = gr.Button("+ Add Image")
        remove_btn = gr.Button("- Remove Image")
        run_btn = gr.Button("Run Inference")

    visuals_output = gr.Code(label="Collected Visual Prompt Info", language="python")

    add_btn.click(fn=lambda idx: [gr.update(visible=(i < idx + 1)) for i in range(MAX_INPUTS)] + [min(idx + 1, MAX_INPUTS)],
                  inputs=index, outputs=visibility_controls + [index])
    remove_btn.click(fn=lambda idx: [gr.update(visible=(i < idx - 1)) for i in range(MAX_INPUTS)] + [max(idx - 1, 1)],
                     inputs=index, outputs=visibility_controls + [index])
    run_btn.click(fn=run_prediction, inputs=input_prompters, outputs=output_images + [visuals_output, source_image_data_state])

    gr.Markdown("### Predict on a New Image Using Accumulated Prompts")
    with gr.Row():
        target_image_input = gr.Image(label="Target Image", type="pil")
        predict_vpe_btn = gr.Button("Predict with Accumulated Visual Prompts")
    target_output_image = gr.Image(label="Target Prediction", type="pil")

    predict_vpe_btn.click(
        fn=predict_with_accumulated_vpe,
        inputs=[target_image_input, visuals_output, source_image_data_state],
        outputs=target_output_image
    )

demo.launch()
