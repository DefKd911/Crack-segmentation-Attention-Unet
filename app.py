import gradio as gr
from model import predict

title = "🧠 UNet Crack Segmentation"
description = "Upload a concrete surface image and get the predicted crack mask."

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(type="pil", label="Predicted Mask"),
    title=title,
    description=description,
    examples=[]
)

if __name__ == "__main__":
    iface.launch()
