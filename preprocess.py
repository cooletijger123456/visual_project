import gradio as gr

def dummy_preprocess(x):
    # placeholder logic – just echo the image back
    return x

with gr.Blocks(title="Pre-processing") as demo:
    gr.Markdown("## Pre-processing (placeholder)")
    with gr.Row():
        inp = gr.Image(label="Input", type="pil")
        out = gr.Image(label="Output", type="pil")
    inp.change(dummy_preprocess, inp, out)
