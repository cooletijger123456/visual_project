import gradio as gr

from visual_prompt import demo as vp_demo   # your full visual-prompt page
from preprocess     import demo as pre_demo # placeholder or real page
from guide          import demo as guide_demo

with gr.Blocks() as site:
    with gr.Tabs():
        with gr.Tab("Visual Prompting"):
            vp_demo.render()                # page 1
        with gr.Tab("Pre-processing"):
            pre_demo.render()               # page 2
        with gr.Tab("Guide"):
            guide_demo.render()             # page 3

if __name__ == "__main__":
    site.launch()

