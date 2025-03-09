import PIL.Image
import gradio as gr
import cv2
import torch
import numpy as np
import scipy
import tempfile
from scipy.ndimage import binary_fill_holes
from ultralytics import YOLOE
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from gradio_image_prompter import ImagePrompter


def init_model(model_id, is_pf=False):
    model = YOLOE(f"pretrain/{model_id}-seg.pt")
    if is_pf:
        model.load(f"pretrain/{model_id}-seg-pf.pt") # TODO: add segmentation modules into pf
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

@smart_inference_mode()
def yoloe_inference(image, prompts, target_image, model_id, image_size, conf_thresh, iou_thresh, input_type):
    model = init_model(model_id)
    kwargs = {}
    if input_type == "Text":
        texts = prompts["texts"]
        model.set_classes(texts, model.get_text_pe(texts))
    elif input_type == "Visual":
        model.model.names = ["target"]
        prompts.pop("texts")
        kwargs = dict(
            prompts=prompts,
            predictor=YOLOEVPSegPredictor
        )
        if target_image:
            model.predict(source=image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, **kwargs)
            model.set_classes(["target"], model.predictor.vpe)
            model.predictor = None  # remove VPPredictor
            image = target_image
            kwargs = {}
    elif input_type == "Prompt-free":
        vocab = model.get_vocab(prompts["texts"])
        model = init_model(model_id, is_pf=True)
        model.set_vocab(vocab, names=prompts["texts"])
        model.model.model[-1].is_fused = True
        model.model.model[-1].conf = 0.001
        model.model.model[-1].max_det = 1000
    
    results = model.predict(source=image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, **kwargs)
    annotated_image = results[0].plot()
    return annotated_image[:, :, ::-1]


def yoloe_inference_for_examples(image, model_path, image_size, conf_thresh, iou_thresh, input_type):
    annotated_image = yoloe_inference(image, model_path, image_size, conf_thresh, iou_thresh, input_type)
    return annotated_image


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    pb_image = ImagePrompter(type="pil", label="Image", visible=True)
                    mask_image = gr.ImageEditor(type="pil", label="Image", value=None, visible=False, layers=False)
                    target_image = gr.Image(type="pil", label="Target Image", value=None, visible=False)
                input_type = gr.Radio(
                    choices=["Text", "Visual", "Prompt-free"],
                    value="Text",
                    label="Input Type",
                )
                text_input = gr.Textbox(label="Please Input Texts", placeholder='dog,person,hat', visible=True, interactive=True)
                with gr.Row():
                    visual_type = gr.Dropdown(choices=["points", "bboxes", "masks"], value="bboxes", show_label=False, visible=False, interactive=True)
                    visual_input = gr.Radio(choices=["Within Image", "Cross Image"], value="Within Image", show_label=False, visible=False, interactive=True)

                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yoloe-v8s",
                        "yoloe-v8m",
                        "yoloe-v8l",
                        "yoloe-11s",
                        "yoloe-11m",
                        "yoloe-11l",
                    ],
                    value="yoloe-v8l",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_thresh = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                iou_thresh = gr.Slider(
                    label="IoU Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.70,
                )
                yoloe_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)

        def update_visibility(input_type):
            if input_type == "Text":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            elif input_type == "Visual":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
            else:
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[text_input, visual_type, visual_input],
        )
        
        def update_visual_type(visual_type):
            if visual_type == "points" or visual_type == "bboxes":
                return gr.update(value=None, visible=True), gr.update(value=None, visible=False)
            elif visual_type == "masks":
                return gr.update(value=None, visible=False), gr.update(value=None, visible=True)

        visual_type.change(
            fn=update_visual_type,
            inputs=[visual_type],
            outputs=[pb_image, mask_image]
        )
        
        def update_visual_input(visual_input):
            if visual_input == "Within Image":
                return gr.update(value=None, visible=False)
            elif visual_input == "Cross Image":
                return gr.update(value=None, visible=True)
        
        visual_input.change(
            fn=update_visual_input,
            inputs=[visual_input],
            outputs=[target_image]
        )

        def run_inference(pb_image, mask_image, target_image, texts, model_id, image_size, conf_thresh, iou_thresh, input_type):
            # add text prompts
            if input_type == "Prompt-free":
                with open('tools/ram_tag_list.txt', 'r') as f:
                    texts = [x.strip() for x in f.readlines()]
            else:
                texts = [text.strip() for text in texts.split(',')]
            prompts = {
                "texts": texts
            }

            # add visual prompts
            if pb_image is not None:
                image, points = pb_image["image"], pb_image["points"]
                points = np.array(points)
                prompts.update({
                    "points": np.array([p[[0, 1]] for p in points if p[2] == 1]),
                    "bboxes": np.array([p[[0, 1, 3, 4]] for p in points if p[2] == 2]),
                })
            elif mask_image is not None:
                image, masks = mask_image["background"], mask_image["layers"][0]
                image = image.convert("RGB")
                masks = np.array(masks.convert("L"))
                masks = binary_fill_holes(masks).astype(np.uint8)
                masks[masks > 0] = 1
                prompts.update({
                    "masks": masks[None]
                })
            
            return yoloe_inference(image, prompts, target_image, model_id, image_size, conf_thresh, iou_thresh, input_type)

        yoloe_infer.click(
            fn=run_inference,
            inputs=[pb_image, mask_image, target_image, text_input, model_id, image_size, conf_thresh, iou_thresh, input_type],
            outputs=[output_image],
        )

        # gr.Examples(
        #     examples=[
        #         [
        #             "ultralytics/assets/bus.jpg",
        #             "yolov10s",
        #             640,
        #             0.25,
        #             0.7,
        #         ],
        #         [
        #             "ultralytics/assets/zidane.jpg",
        #             "yolov10s",
        #             640,
        #             0.25,
        #             0.7,
        #         ],
        #     ],
        #     fn=yoloe_inference_for_examples,
        #     inputs=[
        #         image,
        #         model_id,
        #         image_size,
        #         conf_thresh,
        #         iou_thresh
        #     ],
        #     outputs=[output_image],
        #     cache_examples='lazy',
        # )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOE: Real-Time Seeing Anything
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='' target='_blank'>arXiv</a> | <a href='' target='_blank'>github</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()

if __name__ == '__main__':
    gradio_app.launch()