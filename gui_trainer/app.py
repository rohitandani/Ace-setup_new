import gradio as gr
from .dataset_tab import create_dataset_tab
from .training_tab import create_training_tab
from .monitoring_tab import create_monitoring_tab
from .inference_tab import create_inference_tab
from .settings_tab import create_settings_tab

def create_interface():
    with gr.Blocks() as interface:
        with gr.Tabs():
            with gr.TabItem("Dataset Creation"):
                create_dataset_tab()
            with gr.TabItem("Training Configuration"):
                create_training_tab()
            with gr.TabItem("Monitoring"):
                create_monitoring_tab()
            with gr.TabItem("Inference"):
                create_inference_tab()
            with gr.TabItem("Settings"):
                create_settings_tab()
    return interface

if __name__ == "__main__":
    app = create_interface()
    app.launch()
