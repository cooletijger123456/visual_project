import argparse
import copy
import os
import uuid
from functools import partial

import numpy as np
import torch
from yololm.utils import process_image_box, add_pad_token, add_spatial_token

import gradio as gr
import gradio.themes.base as ThemeBase
from gradio_image_prompter import ImagePrompter
from PIL import Image

from transformers import AutoTokenizer
from yololm.models.yololm import YOLOForCausalLM
from yololm.utils import preprocess, make_mask
from yololm.utils import disable_torch_init
from yololm.models.utils import KeywordsStoppingCriteria
from ultralytics.data.augment import LetterBox

class ConversationBot:
    def __init__(self, model_name):
        self.build_model(model_name)
        self.scale_factor = 1/8
        self.imgsz = 800
        self.transform = LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)

    def build_model(self, model_name):
        ########################  base model define ########################
        print('Start loading model...')
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = YOLOForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            use_cache=True,
        ).cuda()

        self.tokenizer = add_pad_token(self.tokenizer, self.model)
        self.tokenizer = add_spatial_token(self.tokenizer)
        self.model.model.tokenizer = self.tokenizer

        vision_tower = self.model.get_model().vision_tower
        vision_tower.to(device='cuda', dtype=torch.float16)
        
        # init inputs: img, inputs ids, texts
        self.last_source = dict()

    def init_inputs(self, input_dict, question_str, history_cache):
        image = input_dict['image']
        bboxes = torch.tensor(input_dict['boxes'])

        image, bboxes = process_image_box(image, bboxes, False, 800, self.transform, "xyxy")

        masksz = self.imgsz * self.scale_factor
        masks = make_mask(bboxes * self.scale_factor, masksz, masksz)
        
        texts = ["<bbox>" + question_str for _ in range(len(bboxes))]

        data = preprocess(texts, "", self.tokenizer)
        data['image'] = image
        data['visuals'] = masks
        
        return data, history_cache

    def run(self, text, image, chat_history, state, history_cache):
        path = 'image/{}.png'.format(uuid.uuid4().hex)
        Image.fromarray(image['image']).convert('RGB').save(path)
        points = np.array(image['points']).reshape(-1, 3)
        image['boxes'] = points[:, :2].reshape(-1, 4).tolist()
        
        chat_history.append(((path,), None))
        text = text.strip()
        assert(text == "<roc>")

        print("text:", text, "boxes:", image['boxes'], "image shape:", image['image'].shape)
        this_round_input = copy.deepcopy(text)

        init_inputs, history_cache = self.init_inputs(dict(image=path, boxes=image['boxes']), text, history_cache)

        input_ids = init_inputs['input_ids'].cuda()
        image = init_inputs['image'][None].half().cuda()
        visuals = [init_inputs['visuals'].half().cuda()]

        stop_str = self.tokenizer.eos_token
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            self.model.orig_forward = self.model.forward
            self.model.forward = partial(
                self.model.orig_forward,
                visuals=visuals
            )

            with torch.amp.autocast(device_type='cuda'):
                output_ids = self.model.generate(
                    input_ids,
                    images=image,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])
            self.model.forward = self.model.orig_forward

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
                input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True)

        for output in outputs:
            output = output.strip()
            if output.endswith(stop_str):
                output = outputs[:-len(stop_str)]
            output = output.strip()
        outputs = "; ".join(outputs)
        init_outputs = outputs

        chat_history.append(('{}'.format(this_round_input.replace('<', '&lt;').replace('>', '&gt;')), init_outputs))
        return None, chat_history, state, history_cache

css = '''
#image_upload {align-items: center; max-width: 640px}
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=20012)
    parser.add_argument('--https', action='store_true')
    parser.add_argument('--model', type=str, default='exp12/tiny-llm')
    args = parser.parse_args()

    os.makedirs('image', exist_ok=True)

    bot = ConversationBot(model_name=args.model)
    with gr.Blocks(theme=ThemeBase.Base(), css=css) as demo:
        state = gr.State([])
        history_cache = gr.State([])
        with gr.Row(visible=True) as user_interface:

            with gr.Column(elem_id='Image', scale=0.5) as img_part:
                with gr.Tab('Image(Please draw the boxes here👇👇👇)', elem_id='image_tab') as img_tab:
                    click_img = ImagePrompter(show_label=False)
                    with gr.Row() as img_btn:
                        clear_btn = gr.Button(value='Clear All', elem_id='clear_btn')

            with gr.Column(scale=0.5, elem_id='text_input') as chat_part:
                chatbot = gr.Chatbot(elem_id='chatbot', label='GPT4RoI')
                with gr.Row(visible=True) as input_row:
                    with gr.Column(min_width=0) as text_col:
                        txt = gr.Textbox(show_label=False,
                                         placeholder="Enter your question here and press 'Enter' to send.")
            txt.submit(
                lambda: gr.update(visible=False), [], [img_btn]
            ).then(
                lambda: gr.update(visible=True), [], [txt]
            ).then(
                bot.run, [txt, click_img, chatbot, state, history_cache], [click_img, chatbot, state, history_cache]
            ).then(
                lambda: gr.update(value=''), None, [txt, ]
            ).then(
                lambda: gr.update(visible=True), [], [txt]
            ).then(
                lambda: gr.update(visible=True), [], [img_btn]
            ).then(
                lambda: gr.update(visible=True), [], [chat_part]
            )

            clear_btn.click(
                lambda: None, [], [click_img]
            ).then(
                lambda: None, None, None
            ).then(
                lambda: [], None, state
            ).then(
                lambda: [], None, history_cache
            ).then(
                lambda: None, None, chatbot
            ).then(
                lambda: os.system('rm image/*.png'), None, []
            ).then(
                lambda: '', None, []
            )
        
    demo.queue(api_open=False).launch(server_name='127.0.0.1', server_port=args.port)

