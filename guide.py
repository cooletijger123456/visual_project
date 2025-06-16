import gradio as gr

with gr.Blocks(title="Guide") as demo:
    gr.Markdown("""
# Guide (placeholder)

This page will contain usage instructions, tips, and FAQs for your app.

* **Visual Prompting** – draw boxes, run inference.  
* **Pre-processing** – resize or blur images before prompting.  
* **Batch Mode** – drop a folder to run on many images.

_Add more sections here as needed._
""")
