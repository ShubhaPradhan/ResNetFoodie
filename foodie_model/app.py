import gradio as gr

from inference import classify

gr.Interface(
    fn=classify,
    inputs=["image"],
    outputs=["text"],
    title="ResNetFoodie",
    description="Transfer Learning of ResNet50 for Nepali Food Classification",
    allow_flagging=False
).launch()
