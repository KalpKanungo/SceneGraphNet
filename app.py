import gradio as gr
import cv2
import numpy as np

from src.pipeline import run_pipeline
from src.scene_graph import build_graph
from src.visualization import visualize_graph
from src.text_generation import graph_to_text   


def process_image(image):
    try:
        
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        
        relations = run_pipeline(image_cv)

        if not relations or len(relations) == 0:
            return image, None, "No relationships detected."

     
        G = build_graph(relations)

       
        fig = visualize_graph(G)

     
        caption = graph_to_text(relations)

        return image, fig, caption

    except Exception as e:
        print("Error:", e)
        return image, None, "Error processing image."


demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(label="Input Image"),
        gr.Plot(label="Scene Graph"),
        gr.Textbox(label="Generated Description")  
    ],
    title="Scene Graph Generator",
    description="Upload an image → Detect objects → Predict relationships → Generate scene graph + description",
    theme="soft"
)


if __name__ == "__main__":
    demo.launch()